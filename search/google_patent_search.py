import os
import json
import logging
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Optional, Tuple
import sys
import math
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_next_test_folder_number(base_path: str) -> int:
        """
        Find the next available test folder number by checking existing test-{number} folders
        Returns the next available number that doesn't exist yet
        """
        try:
            # Get all folders that match the pattern 'test-{number}'
            existing_folders = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d)) and d.startswith('test-')]
            
            # Extract numbers from folder names
            folder_numbers = []
            for folder in existing_folders:
                try:
                    number = int(folder.split('-')[1])
                    folder_numbers.append(number)
                except (IndexError, ValueError):
                    continue
            
            # If no existing folders found, start with 1
            if not folder_numbers:
                return 1
                
            # Get the next number after the highest existing number
            next_number = max(folder_numbers) + 1
            return next_number
            
        except FileNotFoundError:
            # If the base path doesn't exist, start with 1
            return 1
        except Exception as e:
            logger.error(f"Error determining next test folder number: {str(e)}")
            raise

class PatentSearchExecutor:
    def __init__(self, serp_api_key: str, base_folder: str = 'queries_refined/patent_queries', 
                 max_total_results: int = 1000, results_per_page: int = 100, 
                 country_codes: Optional[List[str]] = None,
                 assignees: Optional[List[str]] = None,
                 max_retries: int = 3,
                 initial_backoff: int = 60):
        self.serp_api_key = serp_api_key
        self.base_folder = base_folder
        self.max_total_results = max_total_results
        self.results_per_page = min(results_per_page, 100)
        self.processed_patents = {}
        self.duplicate_count = 0
        self.country_codes = [code.upper() for code in country_codes] if country_codes else None
        self.assignees = assignees  # Now an array of assignees
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff

    def retry_with_backoff(self, func, *args, **kwargs):
        """Execute a function with exponential backoff retry logic"""
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except requests.exceptions.RequestException as e:
                if "429" in str(e):  # Rate limit error
                    wait_time = self.initial_backoff * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"Rate limit hit. Waiting {wait_time} seconds before retry {attempt + 1}/{self.max_retries}")
                    time.sleep(wait_time)
                    continue
                raise  # Re-raise other request exceptions
        
        logger.error(f"Max retries ({self.max_retries}) exceeded")
        return None, f"Max retries exceeded after rate limit", 0

    def save_progress(self, results_dir: str, consolidated_data: Dict):
        """Save current results to an intermediate file"""
        try:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            output_path = os.path.join(results_dir, f'patent_results_{timestamp}.json')
            with open(output_path, 'w') as f:
                json.dump(consolidated_data, f, indent=2)
            logger.info(f"Progress saved to: {output_path}")
        except Exception as e:
            logger.error(f"Error saving progress: {str(e)}")
        
    def matches_country_code(self, publication_number: str) -> bool:
        """Check if publication number matches any of the specified country codes"""
        if not self.country_codes:
            return True  # Accept all if no country codes specified
            
        # Extract country code from start of publication number
        # Look for 2-character country codes (most common)
        country_code = publication_number[:2].upper()
        if country_code in self.country_codes:
            return True
            
        # If no match found with 2 chars, try 3-character codes
        country_code = publication_number[:3].upper()
        return country_code in self.country_codes
        
    def load_queries(self, test_number: int) -> List[str]:
        """Load queries from any JSON file in the specified test folder"""
        folder_path = os.path.join(self.base_folder, f'test-{test_number}')
        
        try:
            json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
            
            if not json_files:
                raise FileNotFoundError(f"No JSON files found in {folder_path}")
            
            json_file = json_files[0]
            query_path = os.path.join(folder_path, json_file)
            
            with open(query_path, 'r') as f:
                data = json.load(f)
                final_queries = data.get('final_queries', {})
                queries = final_queries.get('queries', [])
                
                if not queries:
                    raise KeyError("No queries found in final_queries section of JSON file")
                    
                logger.info(f"Loaded {len(queries)} queries from {query_path}")
                return queries
                
        except Exception as e:
            logger.error(f"Error loading queries from test-{test_number} folder: {str(e)}")
            raise


    def extract_first_claim(self, soup: BeautifulSoup) -> str:
        """
        Extract up to the first 6 claims from the patent.
        Despite the method name being kept as 'extract_first_claim' for compatibility,
        it now extracts multiple claims and combines them into a single string.
        """
        claims_section = soup.find("claims") or soup.find("div", class_="claims")
        claims = []
        
        if claims_section:
            # Try both modern and legacy claim formats
            claim_elements = claims_section.select(":not(claim-text) > claim-text")
            
            if not claim_elements:
                claim_elements = claims_section.select(":not(.claim-text) > div.claim-text")
            
            # Get up to 6 claims
            for i, claim_element in enumerate(claim_elements[:6]):
                claim_text = ' '.join(claim_element.stripped_strings)
                if claim_text:  # Only add non-empty claims
                    claims.append(f"Claim {i+1}: {claim_text}")
        
        # Join all claims with newlines between them
        return '\n\n'.join(claims) if claims else ""

    def fetch_patent_details(self, publication_number: str) -> Dict:
        """Fetch specific patent details using SerpAPI"""
        try:
            if publication_number.startswith("/patent/"):
                publication_number = publication_number.split("/patent/")[1].split("/en")[0]
            clean_number = publication_number.replace('-', '').replace(' ', '')
            
            # First check if we've already processed this patent
            if clean_number in self.processed_patents:
                self.duplicate_count += 1
                logger.info(f"Skipping duplicate patent: {clean_number}")
                return None
                
            # Then check country code match before making API request
            if not self.matches_country_code(clean_number):
                logger.info(f"Skipping patent with non-matching country code: {clean_number}")
                return None
            
            params = {
                "engine": "google_patents_details",
                "patent_id": f"patent/{clean_number}/en",
                "output": "html",
                "api_key": self.serp_api_key
            }
            
            logger.info(f"Fetching details for patent: {clean_number}")
            
            def _fetch():
                response = requests.get("https://serpapi.com/search", params=params)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.text, 'html.parser')
                
                title_elem = soup.find("span", itemprop="title")
                abstract_elem = soup.find("abstract") or soup.find("div", class_="abstract")
                pub_num_elem = soup.find("dd", itemprop="publicationNumber")
                first_claim = self.extract_first_claim(soup)
                
                patent_details = {
                    "title": title_elem.get_text(strip=True) if title_elem else "",
                    "publication_number": pub_num_elem.get_text(strip=True) if pub_num_elem else clean_number,
                    "abstract": abstract_elem.get_text(strip=True) if abstract_elem else "",
                    "claim": first_claim,
                    "link": f"https://patents.google.com/patent/{clean_number}/en"
                }
                
                return patent_details, None, None  # Match retry_with_backoff return format
            
            patent_details, error, _ = self.retry_with_backoff(_fetch)
            
            if error:
                logger.error(f"Error fetching patent details after retries: {error}")
                return None
            
            if patent_details:
                self.processed_patents[clean_number] = patent_details
            
            return patent_details
            
        except Exception as e:
            logger.error(f"Error fetching patent details for {publication_number}: {str(e)}")
            logger.debug("Exception details:", exc_info=True)
            return None

    def save_query_results(self, results_dir: str, query: str, query_data: Dict):
        """Save results from a single query to its own file"""
        try:
            # Create a clean filename from the query
            # Replace special characters and trim length
            safe_query = "".join(c for c in query if c.isalnum() or c in (' ', '-', '_'))[:100]
            safe_query = safe_query.replace(' ', '_')
            
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f'query_results_{safe_query}_{timestamp}.json'
            output_path = os.path.join(results_dir, filename)
            
            with open(output_path, 'w') as f:
                json.dump(query_data, f, indent=2)
            logger.info(f"Saved query results to: {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving query results: {str(e)}")


    def execute_search(self, test_number: int):
        """Execute patent searches for all queries with pagination"""
        # Determine the next available test folder number
        base_results_path = 'patent_results'
        next_test_number = get_next_test_folder_number(base_results_path)
        results_dir = os.path.join(base_results_path, f'test-{next_test_number}')
        
        # Create the new directory
        os.makedirs(results_dir, exist_ok=False)  # Will raise error if directory exists
        
        logger.info(f"Creating new results directory: {results_dir}")
        
        try:
            queries = self.load_queries(test_number)
            logger.info(f"\n{'='*80}")
            logger.info(f"Starting search execution for test-{test_number}")
            logger.info(f"Saving results to test-{next_test_number}")
            logger.info(f"Found {len(queries)} queries to process")
            logger.info(f"Results per page: {self.results_per_page}")
            logger.info(f"{'='*80}\n")
            
            # Rest of the existing execute_search code remains the same...
            
            for i, query in enumerate(queries, 1):
                logger.info(f"\n{'*'*80}")
                logger.info(f"Processing query {i}/{len(queries)}")
                logger.info(f"Query: {query}")
                logger.info(f"{'*'*80}\n")
                
                self.processed_patents = {}
                self.duplicate_count = 0
                
                query_data = {
                    "query_info": {
                        "query": query,
                        "total_available_results": 0,
                        "results_collected": 0,
                        "pages_fetched": 0,
                        "duplicate_count": 0
                    },
                    "results": []
                }
                
                try:
                    # Get first page and total results
                    initial_results, error, total_available = self.fetch_patent_results(query, page=1)
                    if error:
                        logger.error(f"Error processing query: {error}")
                        query_data["query_info"]["error"] = error
                        self.save_query_results(results_dir, query, query_data)
                        continue
                    
                    query_data["query_info"]["total_available_results"] = total_available
                    results_to_fetch = min(total_available, self.max_total_results)
                    pages_needed = math.ceil(results_to_fetch / self.results_per_page)
                    
                    logger.info(f"Query Statistics:")
                    logger.info(f"- Total results available: {total_available}")
                    logger.info(f"- Results to fetch: {results_to_fetch}")
                    logger.info(f"- Pages needed: {pages_needed}")
                    logger.info(f"- Results per page: {self.results_per_page}\n")
                    
                    # Process first page results
                    self.process_page_results(initial_results, query_data["results"], query_data["query_info"])
                    query_data["query_info"]["duplicate_count"] = self.duplicate_count
                    
                    # Save after processing first page
                    self.save_query_results(results_dir, query, query_data)
                    
                    # Process remaining pages
                    for current_page in range(2, pages_needed + 1):
                        if query_data["query_info"]["results_collected"] >= self.max_total_results:
                            logger.info("Reached maximum results limit. Stopping pagination.")
                            break
                            
                        logger.info(f"Fetching page {current_page}/{pages_needed}")
                        page_results, error, _ = self.fetch_patent_results(query, page=current_page)
                        
                        if error:
                            logger.error(f"Error on page {current_page}: {error}")
                            self.save_query_results(results_dir, query, query_data)
                            continue
                            
                        if not page_results:
                            logger.warning(f"No results found on page {current_page}. Stopping pagination.")
                            break
                            
                        self.process_page_results(page_results, query_data["results"], query_data["query_info"])
                        query_data["query_info"]["duplicate_count"] = self.duplicate_count
                        
                        # Save after each page
                        self.save_query_results(results_dir, query, query_data)
                        time.sleep(1)  # Rate limiting
                    
                    # Final save for this query
                    self.save_query_results(results_dir, query, query_data)
                    
                except Exception as e:
                    logger.error(f"Error processing query: {str(e)}")
                    query_data["query_info"]["error"] = str(e)
                    self.save_query_results(results_dir, query, query_data)
                    continue
            
            logger.info(f"\n{'='*80}")
            logger.info("Search Execution Complete")
            logger.info(f"{'='*80}\n")
            
        except Exception as e:
            logger.error(f"Error executing search for test-{test_number}: {str(e)}")
            raise


    def fetch_patent_results(self, query: str, page: int = 1) -> Tuple[List[Dict], Optional[str], int]:
        """Fetch patent search results from SERP API with retry logic"""
        params = {
            "engine": "google_patents",
            "q": query,
            "api_key": self.serp_api_key,
            "page": page,
            "num": self.results_per_page,
            "dups": "language"
        }
        
        # Add assignee parameter if specified
        if self.assignees:
            params["assignee"] = ",".join(self.assignees)
        
        def _fetch():
            response = requests.get("https://serpapi.com/search", params=params)
            response.raise_for_status()
            data = response.json()
            
            if 'error' in data:
                raise requests.exceptions.RequestException(data['error'])
            
            total_results = int(data.get('search_information', {}).get('total_results', 0))
            
            results = []
            if 'organic_results' in data:
                results = data['organic_results']
            elif 'patents' in data:
                results = data['patents']
            
            if not results:
                return [], "No results found in API response", 0
            
            return results, None, total_results
            
        return self.retry_with_backoff(_fetch)
    
    def process_page_results(self, page_results: List[Dict], query_results: List[Dict], query_object: Dict):
        """Process results from a single page"""
        for result in page_results:
            if query_object["results_collected"] >= self.max_total_results:
                break
                
            if 'publication_number' in result:
                details = self.fetch_patent_details(result['publication_number'])
                if details:  # Will be None if it was a duplicate or non-matching country code
                    details.update({
                        'snippet': result.get('snippet', ''),
                        'priority_date': result.get('priority_date', ''),
                        'filing_date': result.get('filing_date', ''),
                        'grant_date': result.get('grant_date', ''),
                        'publication_date': result.get('publication_date', ''),
                        'inventor': result.get('inventor', ''),
                        'assignee': result.get('assignee', '')
                    })
                    query_results.append(details)
                    query_object["results_collected"] += 1
        
        query_object["pages_fetched"] += 1
        
def main():
    # Define your variables here
    test_number = 7  
    serp_api_key = ""
    max_total_results = 750
    results_per_page = 100  
    base_folder = 'patent_queries'  
    
    # Country codes added into the Array will filter Patents based on their code value. Leave None for no filter.
  
    country_codes = ["US", "EP", "JP", "CN"]  
    
    assignees = None  # Can be None for no assignee filtering
    
    executor = PatentSearchExecutor(
        serp_api_key=serp_api_key,
        base_folder=base_folder,
        max_total_results=max_total_results,
        results_per_page=results_per_page,
        country_codes=country_codes,
        assignees=assignees  # Now passing array of assignees
    )
    
    try:
        executor.execute_search(test_number)
        print(f"Search execution completed for test-{test_number}")
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
