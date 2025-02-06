import os
import json
import logging
import requests
from typing import List, Dict
import sys
import math
from collections import defaultdict
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ScholarSearchExecutor:
    def __init__(self, api_key: str, main_folder: str, max_total_results: int = 1000, results_per_page: int = 20):
        """
        Initialize the search executor
        
        Args:
            api_key: SerpAPI key
            main_folder: Main folder containing test subfolders
            max_total_results: Maximum total results to fetch across all pages for each query
            results_per_page: Number of results to fetch per page (max 100 for SerpAPI)
        """
        self.api_key = api_key
        self.base_url = "https://serpapi.com/search"
        self.main_folder = main_folder
        self.max_total_results = max_total_results
        self.results_per_page = min(results_per_page, 20)  # SerpAPI max is 100
        
    def get_next_test_number(self) -> int:
        """
        Determine the next test number by scanning existing directories
        Returns the next available test number
        """
        results_dir = 'scholar_results'
        if not os.path.exists(results_dir):
            return 1
            
        existing_tests = [d for d in os.listdir(results_dir) 
                         if os.path.isdir(os.path.join(results_dir, d)) 
                         and d.startswith('test-')]
        
        if not existing_tests:
            return 1
            
        test_numbers = [int(d.split('-')[1]) for d in existing_tests]
        return max(test_numbers) + 1

    def load_queries(self, subfolder_number: int) -> List[str]:
        """Load queries from the specified folder structure"""
        folder_path = os.path.join(self.main_folder, f'test-{subfolder_number}')
        
        try:
            json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
            if not json_files:
                raise FileNotFoundError(f"No JSON files found in {folder_path}")
            
            json_file = json_files[0]
            query_path = os.path.join(folder_path, json_file)
            
            with open(query_path, 'r') as f:
                data = json.load(f)
                queries = data.get('final_queries', {}).get('queries', [])
                
                if not queries:
                    raise KeyError("No queries found in final_queries.queries section of JSON file")
                    
                logger.info(f"Loaded {len(queries)} queries from {query_path}")
                return queries
                
        except Exception as e:
            logger.error(f"Error loading queries from {folder_path}: {str(e)}")
            raise
            
    def _fetch_search_results(self, query: str, start: int = 0) -> Dict:
        """
        Fetch search results from SERP API with pagination support
        
        Args:
            query: Search query string
            start: Starting index for pagination
        """
        remaining_results = self.max_total_results - start
        
        params = {
            "engine": "google_scholar",
            "q": query,
            "api_key": self.api_key,
            "num": min(self.results_per_page, remaining_results),  # Using full results_per_page
            "start": start  # This will be multiples of results_per_page
        }
        
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"API request failed for query '{query}' at offset {start}: {str(e)}")
            raise

    def process_results(self, raw_results: Dict) -> List[Dict]:
        """Process raw results into desired format"""
        processed_results = []
        
        if 'organic_results' not in raw_results:
            logger.warning("No organic results found in response")
            return processed_results
            
        for result in raw_results['organic_results']:
            processed_result = {
                "title": result.get('title', ''),
                "snippet": result.get('snippet', ''),
                "link": result.get('link', ''),
                "publication_info": result.get('publication_info', {}).get('summary', ''),
            }
            
            processed_results.append(processed_result)
            
        return processed_results
        
    def remove_duplicates(self, results: List[Dict]) -> tuple[List[Dict], int]:
        """
        Remove duplicate results based on the 'link' field
        
        Args:
            results: List of result dictionaries
            
        Returns:
            Tuple of (deduplicated results list, number of duplicates removed)
        """
        seen_links = {}
        unique_results = []
        duplicates_count = 0
        
        for result in results:
            link = result.get('link')
            if not link:
                unique_results.append(result)
                continue
                
            if link not in seen_links:
                seen_links[link] = result
                unique_results.append(result)
            else:
                duplicates_count += 1
                
        logger.info(f"Removed {duplicates_count} duplicate results")
        return unique_results, duplicates_count

    def execute_search(self, subfolder_number: int):
        """Execute searches using queries from specified subfolder with pagination"""
        results_test_number = self.get_next_test_number()
        results_dir = os.path.join('scholar_results', f'test-{results_test_number}')
        os.makedirs(results_dir, exist_ok=True)
        
        try:
            queries = self.load_queries(subfolder_number)
            logger.info(f"\n{'='*80}")
            logger.info(f"Starting search execution for test-{results_test_number}")
            logger.info(f"Found {len(queries)} queries to process")
            logger.info(f"Results per page: {self.results_per_page}")
            logger.info(f"{'='*80}\n")
            
            consolidated_data = {
                "total_results": 0,
                "duplicates_removed": 0,
                "queries": [],
                "results": []
            }
            
            for query_index, query in enumerate(queries, 1):
                logger.info(f"\n{'*'*80}")
                logger.info(f"Processing query {query_index}/{len(queries)}")
                logger.info(f"Query: {query}")
                logger.info(f"{'*'*80}\n")
                
                try:
                    # Get first page and total results
                    first_page = self._fetch_search_results(query, 0)
                    total_available = int(first_page.get('search_information', {}).get('total_results', 0))
                    results_to_fetch = min(total_available, self.max_total_results)
                    pages_needed = math.ceil(results_to_fetch / self.results_per_page)
                    
                    logger.info(f"Query Statistics:")
                    logger.info(f"- Total results available: {total_available}")
                    logger.info(f"- Results to fetch: {results_to_fetch}")
                    logger.info(f"- Pages needed: {pages_needed}")
                    logger.info(f"- Results per page: {self.results_per_page}\n")
                    
                    query_results = []
                    query_object = {
                        "query": query,
                        "total_available_results": total_available,
                        "results_collected": 0,
                        "pages_fetched": 0
                    }
                    
                    # Process all pages
                    for page in range(pages_needed):
                        if query_object["results_collected"] >= self.max_total_results:
                            logger.info("Reached maximum results limit. Stopping pagination.")
                            break
                        
                        start_index = page * self.results_per_page  # Using proper offset based on results_per_page
                        
                        # Use existing results for first page
                        if page == 0:
                            page_data = first_page
                        else:
                            logger.info(f"Fetching page {page + 1}/{pages_needed} (start index: {start_index})")
                            page_data = self._fetch_search_results(query, start_index)
                        
                        processed_page = self.process_results(page_data)
                        if not processed_page:
                            logger.warning(f"No results found on page {page + 1}. Stopping pagination.")
                            break
                        
                        results_before = len(query_results)
                        
                        # Add query index to results and collect them
                        for result in processed_page:
                            if query_object["results_collected"] >= self.max_total_results:
                                break
                            result["query_index"] = query_index
                            query_results.append(result)
                            query_object["results_collected"] += 1
                        
                        new_results = len(query_results) - results_before
                        query_object["pages_fetched"] += 1
                        
                        logger.info(f"Page {page + 1} complete:")
                        logger.info(f"- New results on this page: {new_results}")
                        logger.info(f"- Total results so far: {query_object['results_collected']}")
                        logger.info("")
                        
                        # Add delay to avoid rate limiting
                        time.sleep(1)
                    
                    logger.info(f"\nQuery {query_index}/{len(queries)} Complete:")
                    logger.info(f"- Final results collected: {query_object['results_collected']}")
                    logger.info(f"- Pages processed: {query_object['pages_fetched']}\n")
                    
                    consolidated_data["queries"].append(query_object)
                    consolidated_data["results"].extend(query_results)
                    
                except Exception as e:
                    logger.error(f"Error processing query '{query}': {str(e)}")
                    query_object = {
                        "query": query,
                        "total_available_results": 0,
                        "results_collected": 0,
                        "pages_fetched": 0,
                        "error": str(e)
                    }
                    consolidated_data["queries"].append(query_object)
                    continue
            
            # Remove duplicates before saving
            logger.info("\nRemoving duplicates...")
            deduplicated_results, duplicates_count = self.remove_duplicates(consolidated_data["results"])
            consolidated_data["results"] = deduplicated_results
            consolidated_data["duplicates_removed"] = duplicates_count
            consolidated_data["total_results"] = len(deduplicated_results)
            
            # Save consolidated results
            if consolidated_data["results"]:
                output_path = os.path.join(results_dir, 'results.json')
                with open(output_path, 'w') as f:
                    json.dump(consolidated_data, f, indent=2)
                
                logger.info(f"\n{'='*80}")
                logger.info("Search Execution Complete")
                logger.info(f"- Results saved to: {output_path}")
                logger.info(f"- Total queries processed: {len(consolidated_data['queries'])}")
                logger.info(f"- Total unique results: {consolidated_data['total_results']}")
                logger.info(f"- Total duplicates removed: {consolidated_data['duplicates_removed']}")
                
                # Per-query summary
                logger.info("\nPer-query Summary:")
                for query_obj in consolidated_data["queries"]:
                    if "error" in query_obj:
                        logger.info(f"Query '{query_obj['query']}': Failed - {query_obj['error']}")
                    else:
                        logger.info(f"Query '{query_obj['query']}': {query_obj['results_collected']} results from {query_obj['pages_fetched']} pages")
                logger.info(f"{'='*80}\n")
            else:
                logger.warning("No results to save")
                
        except Exception as e:
            logger.error(f"Error executing search for test-{results_test_number}: {str(e)}")
            raise

def main():
    main_folder = "google_queries"
    subfolder_number = 13
    api_key = ""
    max_total_results = 50 # MAX TOTAL RESULTS PER QUERY
    results_per_page = 20
    
    executor = ScholarSearchExecutor(
        api_key=api_key,
        main_folder=main_folder,
        max_total_results=max_total_results,
        results_per_page=results_per_page
    )
    
    try:
        executor.execute_search(subfolder_number)
        print(f"Search execution completed successfully using queries from {main_folder}/test-{subfolder_number}")
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
