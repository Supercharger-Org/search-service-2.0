import os
import json
import logging
from typing import Dict, List, Tuple, Optional
import vertexai
from vertexai.generative_models import GenerationConfig, GenerativeModel
from vertexai.preview.tokenization import get_tokenizer_for_model
from datetime import datetime
import sys
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import re
import asyncio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

INPUT_COST_PER_MILLION = 0.0375
OUTPUT_COST_PER_MILLION = 0.15

def get_next_test_number(base_path: str) -> int:
    """
    Determine the next test number in the specified path
    Args:
        base_path: Base directory path to check for existing test folders
    Returns:
        int: Next available test number
    """
    try:
        os.makedirs(base_path, exist_ok=True)
        
        existing_dirs = [d for d in os.listdir(base_path) 
                        if os.path.isdir(os.path.join(base_path, d)) 
                        and d.startswith('test-')]
        
        if not existing_dirs:
            return 1
            
        existing_numbers = [int(d.split('-')[1]) for d in existing_dirs]
        return max(existing_numbers) + 1
        
    except Exception as e:
        logger.error(f"Error determining next test number: {str(e)}")
        raise

def build_input_path(base_folder: str, subfolder: Optional[str], test_number: int) -> str:
    """
    Build the input file path based on provided folder structure
    Args:
        base_folder: Base directory for input
        subfolder: Optional subdirectory
        test_number: Test number to use
    Returns:
        str: Complete path to the input JSON file
    """
    if subfolder:
        path = os.path.join(base_folder, subfolder, f'test-{test_number}', 'results.json')
    else:
        path = os.path.join(base_folder, f'test-{test_number}', 'results.json')
    return path

def build_output_path(base_folder: str, output_subfolder: str) -> Tuple[str, int]:
    """
    Build the output directory path and get next test number
    Args:
        base_folder: Base directory for output
        output_subfolder: Subdirectory for output
    Returns:
        Tuple[str, int]: Complete output path and next test number
    """
    output_base = os.path.join(base_folder, output_subfolder)
    next_test = get_next_test_number(output_base)
    output_dir = os.path.join(output_base, f'test-{next_test}')
    os.makedirs(output_dir, exist_ok=True)
    return output_dir, next_test

class BrowserHeaders:
    """Class to provide progressively more sophisticated browser headers"""
    
    BASIC_HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    STANDARD_HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1'
    }
    
    ENHANCED_HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Cache-Control': 'max-age=0'
    }
    
    FULL_HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Cache-Control': 'max-age=0',
        'sec-ch-ua': '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Windows"',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Sec-Fetch-User': '?1',
        'Upgrade-Insecure-Requests': '1'
    }
    
    FIREFOX_HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/115.0',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Sec-Fetch-User': '?1'
    }
    
    @classmethod
    def get_header_sequence(cls) -> List[Dict[str, str]]:
        """Return list of headers to try in sequence"""
        return [
            cls.BASIC_HEADERS,
            cls.STANDARD_HEADERS,
            cls.ENHANCED_HEADERS,
            cls.FULL_HEADERS,
            cls.FIREFOX_HEADERS
        ]
    
class TextExtractor:
    def extract_page_meta(self, html_content: str) -> str:
        """Extract metadata and initial content from HTML"""
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Extract meta tags
        meta_content = []
        for meta in soup.find_all('meta'):
            if meta.get('content'):
                name = meta.get('name') or meta.get('property') or ''
                content = meta.get('content')
                if name and content:
                    meta_content.append(f"{name}: {content}")
        
        # Get initial page content
        content = soup.get_text(separator=' ')
        initial_content = ' '.join(content.split())[:20000]
        
        return "\n".join(meta_content + [initial_content])

    def extract_text_segments(self, html_content: str) -> List[Dict[str, str]]:
        """Extract relevant text segments from HTML content"""
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style elements
        for element in soup(["script", "style"]):
            element.decompose()
        
        # Get cleaned text
        text = ' '.join(soup.get_text(separator=' ').split())
        
        segments = []
        # Add initial page meta and content
        segments.append({
            "type": "page_meta",
            "content": self.extract_page_meta(html_content)
        })
        
        # Function to extract segments for a given term
        def extract_term_segments(term: str, text: str, max_segments: int = 2) -> List[str]:
            found_segments = []
            start_pos = 0
            last_end = -3000  # Initialize to allow first match
            
            while len(found_segments) < max_segments:
                pos = text.lower().find(term, start_pos)
                if pos == -1 or pos - last_end < 3000:
                    break
                
                # Extract surrounding text
                seg_start = max(0, pos - 500)
                seg_end = min(len(text), pos + 3000)
                segment = text[seg_start:seg_end].strip()
                
                found_segments.append(segment)
                last_end = seg_end
                start_pos = pos + 1
            
            return found_segments
        
        # Extract abstract segments
        for segment in extract_term_segments('abstract', text):
            segments.append({
                "type": "abstract",
                "content": segment
            })
        
        # Extract summary segments
        for segment in extract_term_segments('summary', text):
            segments.append({
                "type": "summary",
                "content": segment
            })
        
        return segments

class ScholarExtractor:
    def __init__(self, project_id: str, scraper_api_key: str, location: str = "us-central1"):
        """Initialize the ScholarExtractor with project details"""
        self.project_id = project_id
        self.location = location
        self.text_extractor = TextExtractor()
        self.scraper_api_requests = 0
        self.scraper_api_key = scraper_api_key
        
        try:
            vertexai.init(project=project_id, location=location)
            self.model = GenerativeModel("gemini-1.5-flash")
            
            # Updated response schema to include additional_snippets
            self.response_schema = {
                "type": "object",
                "properties": {
                    "publication_date": {
                        "type": "string",
                        "description": "Publication date in YYYY-MM-DD format"
                    },
                    "abstract": {
                        "type": "string",
                        "description": "Full abstract text"
                    },
                    "abstract_source": {
                        "type": "string",
                        "description": "Part of text that abstract was extracted from"
                    },
                    "additional_snippets": {
                        "type": "string",
                        "description": "Additional relevant article content, excluding the abstract/summary, metadata, or citations"
                    },
                    "media_type": {
                        "type": "string",
                        "enum": ["Book or Print Media", "Academic Publication", "Online Article or Blog Post"],
                        "description": "Classification of the media type"
                    }
                },
                "required": ["publication_date", "abstract", "abstract_source", "media_type", "additional_snippets"]
            }
            
            logger.info("Successfully initialized Vertex AI and Gemini model")
        except Exception as e:
            logger.error(f"Error initializing Vertex AI: {str(e)}")
            raise

    def load_scholar_results(self, input_path: str) -> Dict:
        """Load scholar results from the specified path"""
        try:
            with open(input_path, 'r') as f:
                data = json.load(f)
                logger.info(f"Successfully loaded results from {input_path}")
                return data
        except Exception as e:
            logger.error(f"Error loading scholar results from {input_path}: {str(e)}")
            raise

    def _extract_content(self, response) -> str:
        """Extract content from Cohere response"""
        if hasattr(response, 'text'):
            return response.text
        if hasattr(response, 'message') and hasattr(response.message, 'content'):
            return response.message.content
        return None

    def count_tokens(self, text: str) -> int:
        """Count tokens for a given text using Gemini's built-in counter"""
        try:
            response = self.model.count_tokens(text)
            return response.total_tokens
        except Exception as e:
            logger.error(f"Error counting tokens: {str(e)}")
            return 0

    async def extract_with_llm(self, segments: List[Dict[str, str]]) -> Dict:
        """Extract publication info using Gemini"""
        # Prepare context text with clear section separation
        context_parts = []
        
        # Organize segments by type
        meta_segments = []
        abstract_segments = []
        article_segments = []
        summary_segments = []
        
        for segment in segments:
            if segment['type'] == 'page_meta':
                meta_segments.append(segment['content'])
            elif segment['type'] == 'abstract':
                abstract_segments.append(segment['content'])
            elif segment['type'] == 'article_content':
                article_segments.append(segment['content'])
            elif segment['type'] == 'summary':
                summary_segments.append(segment['content'])
        
        # Build organized context
        context_parts.extend([
            "=== PAGE METADATA ===",
            *meta_segments,
            "",
            "=== ABSTRACT SECTIONS ===",
            *abstract_segments,
            "",
            "=== ARTICLE CONTENT ===",
            *article_segments,
            "",
            "=== SUMMARY SECTIONS ===",
            *summary_segments
        ])
        
        combined_text = "\n\n".join(filter(None, context_parts))
        
        prompt = f"""Extract publication information from these webpage segments:

{combined_text}

Requirements:

1. Publication date:
- Must be in YYYY-MM-DD format
- Use YYYY-01-01 if only year is available
- Use YYYY-MM-01 if only year and month are available

2. Abstract:
- Must be extracted directly from the webpage segments, represents the actual Abstract or "Summary"
- Must be complete and coherent

3. Additional Snippets:
- Extract 3-5 paragraphs of additional relevant content that meets ALL these criteria:
  * Directly relates to the main focus of the article based on the extracted Abstract and metadata of the page
  * Is NOT part of citations, author bios, or metadata
  * Is NOT already included in the abstract
  * Contains substantive article content
- Leave this field empty if no content meets ALL criteria
- Focus on capturing introductory or key content that appears before any paywall that can supplement the abstract

4. Media Type:
- Classify as exactly one of:
  * "Book or Print Media" - for books, textbooks, printed publications
  * "Academic Publication" - for peer-reviewed papers, journal articles, conference proceedings
  * "Online Article or Blog Post" - for web articles, blog posts, online news
- Base classification on content structure, source, and formatting"""

        try:
            # Generate response using Gemini
            input_tokens = self.count_tokens(prompt)
            
            response = self.model.generate_content(
                prompt,
                generation_config=GenerationConfig(
                    response_mime_type="application/json",
                    response_schema=self.response_schema
                )
            )
            
            usage_metadata = response.usage_metadata
            output_tokens = usage_metadata.candidates_token_count
            
            # Parse and validate response
            result = json.loads(response.text)
            
            # Validate media_type
            valid_media_types = ["Book or Print Media", "Academic Publication", "Online Article or Blog Post"]
            if result.get("media_type") not in valid_media_types:
                result["media_type"] = "Online Article or Blog Post"
            
            # Store token information
            token_info = {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": usage_metadata.total_token_count,
                "cost": 0.0
            }

            return {
                "publication_date": result["publication_date"],
                "abstract": result["abstract"],
                "abstract_source": result["abstract_source"],
                "additional_snippets": result.get("additional_snippets", ""),
                "media_type": result["media_type"],
                "extract_tokens": token_info
            }
            
        except Exception as e:
            logger.error(f"Error in LLM extraction: {str(e)}", exc_info=True)
            return {
                "publication_date": "",
                "abstract": "",
                "abstract_source": "",
                "additional_snippets": "",
                "media_type": "",
                "extract_tokens": {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "cost": 0.0
                },
                "error": str(e)
            }

    def create_error_response(self, method: str, error_msg: str = "") -> Dict:
        """Create a standardized error response"""
        return {
            "publication_date": "",
            "abstract": "",
            "media_type": "",  # Added media_type field to error response
            "extraction_successful": False,
            "method": method,
            "error": error_msg,
            "extract_tokens": {"input_tokens": 0, "output_tokens": 0, "cost": 0.0}
        }

    async def process_content(self, html_content: str, method: str) -> Dict:
        """Process HTML content regardless of source"""
        try:
            # Extract text segments
            segments = self.text_extractor.extract_text_segments(html_content)
            if not segments:
                return self.create_error_response(f"{method}_failed", "No segments extracted")
            
            # Extract information using LLM
            extraction_result = await self.extract_with_llm(segments)
            if not extraction_result:
                return self.create_error_response(f"{method}_failed", "LLM extraction failed")
            
            # Check if we got meaningful content
            has_abstract = bool(extraction_result.get("abstract", "").strip())
            has_date = bool(extraction_result.get("publication_date", "").strip())
            has_media_type = bool(extraction_result.get("media_type", "").strip())
            
            result = {
                **extraction_result,
                "extraction_successful": (has_abstract or has_date) and has_media_type,  # Success requires media type
                "method": method,
                "abstract_confidence": 1.0 if has_abstract else 0.0
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing content with {method}: {str(e)}", exc_info=True)
            return self.create_error_response(f"{method}_failed", str(e))
        
    async def fetch_with_requests(self, url: str) -> Optional[str]:
        """Attempt to fetch URL content using requests with progressive header attempts"""
        
        for i, headers in enumerate(BrowserHeaders.get_header_sequence(), 1):
            try:
                logger.info(f"Attempting request {i} to {url} with header set...")
                response = requests.get(url, headers=headers, timeout=30)
                response.raise_for_status()
                return response.text
                
            except requests.RequestException as e:
                # Enhanced error logging
                if isinstance(e, requests.HTTPError):
                    logger.error(f"HTTP Error for {url}: {e.response.status_code} - {e.response.reason}")
                    if e.response.status_code == 403:
                        logger.error(f"Access forbidden. Response headers: {dict(e.response.headers)}")
                elif isinstance(e, requests.ConnectionError):
                    logger.error(f"Connection Error for {url}: {str(e)}")
                elif isinstance(e, requests.Timeout):
                    logger.error(f"Timeout Error for {url}: {str(e)}")
                else:
                    logger.error(f"Request Error for {url}: {str(e)}", exc_info=True)
                continue
                
        logger.error(f"All header attempts failed for {url}")
        return None

    async def fetch_with_scraper_api(self, url: str) -> Optional[str]:
        """Fetch URL content using ScraperAPI with single retry and 120s timeout"""
        MAX_RETRIES = 2  # One initial attempt + one retry
        TIMEOUT = 120    # 120 seconds timeout
        
        for attempt in range(MAX_RETRIES):
            try:
                self.scraper_api_requests += 1
                
                params = {
                    'api_key': self.scraper_api_key,
                    'url': url,
                    'autoparse': 'true',
                    'premium': 'false',
                    'render': 'true',
                    'device': 'desktop',
                    'country_code': 'us'
                }
                
                logger.info(f"ScraperAPI attempt {attempt + 1}/{MAX_RETRIES} for {url}")
                
                # Use a session for better connection reuse
                with requests.Session() as session:
                    session.mount('https://', requests.adapters.HTTPAdapter(
                        max_retries=0,  # No connection-level retries, we're handling retries ourselves
                        pool_connections=10,
                        pool_maxsize=10
                    ))
                    
                    response = session.get(
                        'https://api.scraperapi.com/',
                        params=params,
                        timeout=TIMEOUT  # Single 120s timeout
                    )
                    
                    logger.info(f"ScraperAPI response status: {response.status_code}")
                    
                    response.raise_for_status()
                    return response.text
                    
            except requests.Timeout as e:
                logger.warning(f"ScraperAPI timeout on attempt {attempt + 1}/{MAX_RETRIES} for {url}: {str(e)}")
                if attempt == MAX_RETRIES - 1:
                    logger.error(f"Both ScraperAPI attempts timed out for {url}")
                    return None
                    
            except requests.RequestException as e:
                if hasattr(e, 'response') and e.response is not None:
                    logger.error(f"ScraperAPI error response: Status {e.response.status_code}")
                    
                    # Don't retry on certain status codes
                    if e.response.status_code in [401, 403, 404]:
                        logger.error(f"Non-retriable status code {e.response.status_code} for {url}")
                        return None
                        
                if attempt == MAX_RETRIES - 1:
                    logger.error(f"All ScraperAPI attempts failed for {url}: {str(e)}")
                    return None
                    
            except Exception as e:
                logger.error(f"Unexpected error with ScraperAPI for {url}: {str(e)}", exc_info=True)
                return None
                
            # Simple delay before retry
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(2)  # 2 second delay before retry
        
        return None

    async def process_content(self, html_content: str, method: str) -> Dict:
        """Process HTML content regardless of source"""
        try:
            # Extract text segments
            segments = self.text_extractor.extract_text_segments(html_content)
            if not segments:
                return {
                    "abstract": "",
                    "abstract_source": "",
                    "abstract_validated": False,
                    "publication_date": "",
                    "publication_date_validated": False,
                    "extraction_successful": False,
                    "extraction_method": method,
                    "extract_tokens": {
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "cost": 0.0
                    }
                }
            
            # Mark extraction as successful since we got segments
            extraction_successful = True
            
            # Extract information using LLM
            extraction_result = await self.extract_with_llm(segments)
            
            # Calculate token costs
            input_tokens = extraction_result.get("extract_tokens", {}).get("input_tokens", 0)
            output_tokens = extraction_result.get("extract_tokens", {}).get("output_tokens", 0)
            
            input_cost = (input_tokens / 1_000_000) * INPUT_COST_PER_MILLION
            output_cost = (output_tokens / 1_000_000) * OUTPUT_COST_PER_MILLION
            total_cost = input_cost + output_cost
            
            # If LLM extraction failed but we got segments, keep extraction_successful as True
            if not extraction_result:
                return {
                    "abstract": "",
                    "abstract_source": "",
                    "abstract_validated": False,
                    "publication_date": "",
                    "publication_date_validated": False,
                    "extraction_successful": extraction_successful,
                    "extraction_method": method,
                    "extract_tokens": {
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "cost": round(total_cost, 6)
                    }
                }
            
            # Get values from LLM response
            abstract = extraction_result.get("abstract", "").strip()
            abstract_source = extraction_result.get("abstract_source", "").strip()
            publication_date = extraction_result.get("publication_date", "").strip()
            
            # Validate fields
            abstract_validated = bool(abstract and abstract_source)
            publication_date_validated = bool(publication_date and len(publication_date.split('-')) == 3)
            
            result = {
                "abstract": abstract,
                "abstract_source": abstract_source,
                "abstract_validated": abstract_validated,
                "publication_date": publication_date,
                "publication_date_validated": publication_date_validated,
                "extraction_successful": extraction_successful,
                "extraction_method": method,
                "extract_tokens": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "cost": round(total_cost, 6)
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing content with {method}: {str(e)}", exc_info=True)
            return {
                "abstract": "",
                "abstract_source": "",
                "abstract_validated": False,
                "publication_date": "",
                "publication_date_validated": False,
                "extraction_successful": False,
                "extraction_method": method,
                "extract_tokens": {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cost": 0.0
                }
            }

    async def process_article_url(self, url: str) -> Dict:
        """Process a single article URL to extract information"""
        try:
            # Try direct requests first
            html_content = await self.fetch_with_requests(url)
            if html_content:
                result = await self.process_content(html_content, "direct")
                if result["extraction_successful"]:
                    logger.info(f"Successfully extracted content via direct request for {url}")
                    return result
                logger.debug(f"Direct request failed for {url}")

            # Try ScraperAPI if needed
            html_content = await self.fetch_with_scraper_api(url)
            if html_content:
                result = await self.process_content(html_content, "scraperAPI")
                if result["extraction_successful"]:
                    logger.info(f"Successfully extracted content via ScraperAPI for {url}")
                    return result
                logger.debug(f"ScraperAPI failed for {url}")

            # If both methods failed
            logger.error(f"All extraction methods failed for {url}")
            return {
                "abstract": "",
                "abstract_source": "",
                "abstract_validated": False,
                "publication_date": "",
                "publication_date_validated": False,
                "extraction_successful": False,
                "extraction_method": "failed",
                "extract_tokens": {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cost": 0.0
                }
            }
                
        except Exception as e:
            logger.error(f"Unexpected error processing {url}: {str(e)}", exc_info=True)
            return {
                "abstract": "",
                "abstract_source": "",
                "abstract_validated": False,
                "publication_date": "",
                "publication_date_validated": False,
                "extraction_successful": False,
                "extraction_method": "error",
                "extract_tokens": {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cost": 0.0
                }
            }

    async def process_results(self, input_path: str, output_path: str):
        """Process all scholar results and extract abstracts"""
        try:
            data = self.load_scholar_results(input_path)
            
            results = []
            successful_extractions = 0
            direct_request_successes = 0
            
            for result in data.get("results", []):
                url = result.get("link")
                if not url:
                    continue
                    
                logger.info(f"Processing URL: {url}")
                extraction_result = await self.process_article_url(url)
                
                if extraction_result["extraction_successful"]:
                    successful_extractions += 1
                    if extraction_result["extraction_method"] == "direct":  # Changed from "method" to "extraction_method"
                        direct_request_successes += 1
                
                enriched_result = {**result, **extraction_result}
                results.append(enriched_result)
            
            # Calculate statistics
            total_processed = len(results)
            success_rate = successful_extractions / total_processed if total_processed > 0 else 0
            direct_request_percent = direct_request_successes / total_processed if total_processed > 0 else 0
            scraper_api_percent = 1 - direct_request_percent
            
            consolidated_data = {
                "metadata": {
                    "total_results": total_processed,
                    "success_rate": success_rate,
                    "no_api_percent": direct_request_percent,
                    "scrape_api_percent": scraper_api_percent,
                    "scrape_api_credits": self.scraper_api_requests
                },
                "extract_token_total": {
                    "input_tokens": sum(r["extract_tokens"]["input_tokens"] for r in results),
                    "output_tokens": sum(r["extract_tokens"]["output_tokens"] for r in results),
                    "cost": sum(r["extract_tokens"]["cost"] for r in results)
                },
                "queries": data.get("queries", []),
                "results": results
            }
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(consolidated_data, f, indent=2)
                
            logger.info(f"Saved consolidated results to {output_path}")
            logger.info(f"Success rate: {success_rate:.2%}")
            logger.info(f"ScraperAPI usage: {self.scraper_api_requests} requests")
                
        except Exception as e:
            logger.error(f"Error processing results: {str(e)}")
            raise


def calculate_token_cost(token_count: int, cost_per_million: float) -> float:
    """Calculate cost for tokens based on per-million rate"""
    return (token_count / 1_000_000) * cost_per_million

def main():
    # Configuration
    input_base_folder = "scholar_results"
    input_subfolder = None
    test_number = 6
    
    output_base = "final_extracted_results"
    output_subfolder = "scholar_results"
    
    # Build input and output paths
    input_path = build_input_path(input_base_folder, input_subfolder, test_number)
    output_dir, next_test = build_output_path(output_base, output_subfolder)
    output_path = os.path.join(output_dir, 'consolidated_results.json')
    
    # Project Settings
    project_id = ""  # Your actual project ID
    location = ""
    scraper_api_key = ""
    
    extractor = ScholarExtractor(project_id, scraper_api_key, location)
    
    try:
        import asyncio
        asyncio.run(extractor.process_results(input_path, output_path))
        print(f"Extraction completed. Results saved to test-{next_test}")
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
