import os
import json
import logging
from typing import Dict, List
import anthropic
import statistics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

INPUT_COST_PER_MILLION = 3.0  # Claude 3.5 Sonnet input cost
OUTPUT_COST_PER_MILLION = 15.0  # Claude 3.5 Sonnet output cost

def calculate_token_cost(token_count: int, cost_per_million: float) -> float:
    """Calculate cost for tokens based on per-million rate"""
    return (token_count / 1_000_000) * cost_per_million

class GoogleResultsAnalyzer:
    def __init__(self, anthropic_api_key: str, target_text: str, testing_abstract: bool = True):
        self.client = anthropic.Client(api_key=anthropic_api_key)
        self.model = "claude-3-5-sonnet-20241022"
        self.target_text = target_text
        self.testing_abstract = testing_abstract  # Add testing_abstract flag

    def get_next_test_number(self, output_base: str, subfolder: str) -> int:
        """Determine the next test number based on existing folders"""
        folder_path = os.path.join(output_base, subfolder)
        try:
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
                return 1
                
            test_folders = [d for d in os.listdir(folder_path) if d.startswith('test-')]
            if not test_folders:
                return 1
                
            # Extract numbers from folder names and find max
            numbers = [int(folder.split('-')[1]) for folder in test_folders]
            return max(numbers) + 1
        except Exception as e:
            logger.error(f"Error determining next test number: {str(e)}")
            raise

    def load_consolidated_results(self, base_folder: str, test_number: int) -> Dict:
        """Load JSON results from the specified test folder"""
        folder_path = os.path.join(base_folder, f'test-{test_number}')
        try:
            # Get the first JSON file in the test folder
            json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
            if not json_files:
                raise FileNotFoundError(f"No JSON file found in {folder_path}")
            
            results_path = os.path.join(folder_path, json_files[0])
            with open(results_path, 'r') as f:
                data = json.load(f)
                return data
        except Exception as e:
            logger.error(f"Error loading results from {folder_path}: {str(e)}")
            raise

    def load_criteria(self, base_folder: str, criteria_test_number: int) -> Dict:
        """Load criteria from the specified test folder"""
        folder_path = os.path.join("criteria", f'test-{criteria_test_number}')
        try:
            # Get the first JSON file in the test folder
            json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
            if not json_files:
                raise FileNotFoundError(f"No JSON file found in {folder_path}")
            
            criteria_path = os.path.join(folder_path, json_files[0])
            with open(criteria_path, 'r') as f:
                data = json.load(f)
                # Return the exact structure we need
                return {
                    'initial_criteria': {
                        'criteria': data.get('initial_criteria', {}).get('criteria', [])
                    },
                    'final_criteria': {
                        'requirements': data.get('final_criteria', {}).get('requirements', [])
                    }
                }
        except Exception as e:
            logger.error(f"Error loading criteria from {folder_path}: {str(e)}")
            raise

    def analyze_result(self, result: Dict, criteria: Dict) -> Dict:
        """Analyze a single search result using Claude with criteria and target text"""
        system_prompt = """You are an expert in technical analysis and relevance assessment. You are aiding an automated application in conducting an initial analysis of search result potential relevance."""

prompt = f"""Analyze the relevance and technical similarity between this search result and the target text, considering the provided criteria.

Target Text:
{self.target_text}

Technical Criteria:
General Criteria: {json.dumps(criteria['initial_criteria']['criteria'], indent=2)}
Technical Requirements: {json.dumps(criteria['final_criteria']['requirements'], indent=2)}

Search Result Information:
Title: {result.get('title', 'No title')}
Abstract: {result.get('abstract', 'No abstract')}
Snippet: {result.get('snippet', 'No snippet')}

Please analyze this result's technical relevance considering the technical criteria and target text."""

        format_instructions = """Provide your response as a JSON object with this structure:
{
    "analysis": "Brief analysis of technical relevance",
    "score": float (0.0 to 1.0, where 0 = technically irrelevant, 0.5 = some technical overlap, 1.0 = high technical relevance)
}"""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1000,
                temperature=0,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": f"{prompt}\n\n{format_instructions}"}
                ]
            )

            # Parse JSON response
            response_text = response.content[0].text
            analysis_data = json.loads(response_text)

            # Calculate token costs
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            
            input_cost = calculate_token_cost(input_tokens, INPUT_COST_PER_MILLION)
            output_cost = calculate_token_cost(output_tokens, OUTPUT_COST_PER_MILLION)
            
            # Add token information to analysis data
            analysis_data["analysis_token_totals"] = {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_cost": round(input_cost + output_cost, 6)
            }

            return analysis_data

        except Exception as e:
            logger.error(f"Error analyzing result: {str(e)}")
            return None

    def process_results(self, base_folder: str, search_test_number: int, criteria_test_number: int):
        """Process all search results using both search results and criteria analysis"""
        try:
            # Load all necessary data
            consolidated_data = self.load_consolidated_results(base_folder, search_test_number)
            criteria = self.load_criteria(base_folder, criteria_test_number)
            
            base_name = os.path.basename(base_folder.rstrip('/'))
            
            output_base = "analyzed_results"
            next_test_num = self.get_next_test_number(output_base, base_name)
            output_dir = os.path.join(output_base, base_name, f'test-{next_test_num}')
            os.makedirs(output_dir, exist_ok=True)
            
            total_input_tokens = 0
            total_output_tokens = 0
            
            # Filter results based on testing_abstract flag
            filtered_results = []
            for result in consolidated_data["results"]:
                # Skip results without validated abstracts if testing_abstract is False
                if not self.testing_abstract and not result.get('abstract_validated', False):
                    logger.info(f"Skipping result {result.get('title', 'Untitled')} - abstract not validated")
                    continue
                
                logger.info(f"Processing result {result.get('title', 'Untitled')}")
                analysis_data = self.analyze_result(result, criteria)
                
                if analysis_data:
                    # Update the result with analysis data while preserving existing fields
                    result.update({
                        "analysis": analysis_data["analysis"],
                        "score": analysis_data["score"],
                        "analysis_token_totals": analysis_data["analysis_token_totals"]
                    })
                    
                    # Add to running totals
                    total_input_tokens += analysis_data["analysis_token_totals"]["input_tokens"]
                    total_output_tokens += analysis_data["analysis_token_totals"]["output_tokens"]
                    
                    filtered_results.append(result)

            consolidated_data["results"] = filtered_results

            consolidated_data["results"].sort(key=lambda x: x.get('score', 0), reverse=True)

            top_20_scores = [r.get('score', 0) for r in consolidated_data["results"][:20]]
            median_score = statistics.median(top_20_scores) if top_20_scores else 0
      
            total_input_cost = calculate_token_cost(total_input_tokens, INPUT_COST_PER_MILLION)
            total_output_cost = calculate_token_cost(total_output_tokens, OUTPUT_COST_PER_MILLION)

            consolidated_data = {
                "median_score": median_score,
                "analysis_token_totals": {
                    "input_tokens": total_input_tokens,
                    "output_tokens": total_output_tokens,
                    "total_cost": round(total_input_cost + total_output_cost, 6)
                },
                "criteria_test_number": criteria_test_number,
                **consolidated_data  # This spreads all the original data after our new fields
            }

            # Save results
            output_path = os.path.join(output_dir, 'analyzed_results.json')
            with open(output_path, 'w') as f:
                json.dump(consolidated_data, f, indent=2)

            logger.info(f"Analysis completed")
            logger.info(f"Median score of top 20 results: {median_score}")
            logger.info(f"Total input tokens: {total_input_tokens}")
            logger.info(f"Total output tokens: {total_output_tokens}")
            logger.info(f"Total token cost: ${round(total_input_cost + total_output_cost, 6)}")
            logger.info(f"Total results analyzed: {len(consolidated_data['results'])}")
            if not self.testing_abstract:
                logger.info("Note: Only processed results with validated abstracts")

        except Exception as e:
            logger.error(f"Error processing results: {str(e)}")
            raise

def main():
    # Define your variables here
    base_folder = "final_extracted_results/scholar_results"  # Base folder containing test folders
    search_test_number = 3  # The test number for search results
    criteria_test_number = 1  # The test number for criteria analysis
    anthropic_api_key = ""
    testing_abstract = False
    
    # Define your target text here
    target_text = """I am requesting a comprehensive global state of the art search regarding a novel room temperature phosphorescent wood hydrogel technology. The invention centers on a unique material that combines natural wood's structural properties with phosphorescent capabilities, creating a versatile biomaterial with switchable mechanical and optical properties.
The core technology utilizes partially delignified wood as a structural skeleton, with polyacrylamide polymerized in situ within the wood structure. The base material starts with basswood that undergoes a specific delignification process, resulting in a composition of approximately 98.3% holocellulose (comprising 88.1% cellulose and 10.2% hemicellulose) and 1.7% lignin. This delignified wood framework is then integrated with polyacrylamide through an in-situ polymerization process, creating a hydrogel with approximately 47.8% water content.
The manufacturing process involves two key stages. First, the wood undergoes delignification through treatment with a solution of 2.5 M NaOH and 0.4 M Na2SO3, followed by exposure to 30% hydrogen peroxide under xenon lamp illumination. After washing with hot deionized water and ethanol, the delignified wood is soaked in an acrylamide solution (0.34 g/mL) and thermally polymerized at 60°C for two hours to form the final hydrogel.
The resulting material exhibits remarkable properties that distinguish it from existing hydrogels. It demonstrates a base tensile strength of 38.4 MPa, which can be increased to 153.8 MPa through ethanol treatment. The material shows room temperature phosphorescence centered at 490 nm with a lifetime of 32.5 ms, which can be extended to 69.7 ms following ethanol treatment. Furthermore, the hydrogel can serve as an energy donor when combined with rhodamine B, producing red afterglow emission at 600 nm with energy transfer efficiency reaching 77.8%.
A particularly notable feature is the material's reversible property modification through alternating ethanol and water treatments. This allows switching between flexible and rigid states while maintaining phosphorescent properties. The material can be processed into various forms including threads, textiles, and complex 3D structures, suggesting potential applications in medical sutures and other biomedical applications.
In conducting this search, particular attention should be paid to existing technologies involving wood-based hydrogels with phosphorescent properties, methods for creating mechanically strong hydrogels using natural wood components, systems for reversible property modification of hydrogels through solvent treatment, and luminescent hydrogel materials intended for medical applications.
The material operates at room temperature and maintains its properties under normal atmospheric conditions in both wet and ethanol-treated states. Please include any relevant prior art that might address similar approaches to creating multi-functional biomaterials with switchable properties."""
    
    analyzer = GoogleResultsAnalyzer(anthropic_api_key, target_text, testing_abstract)
    
    try:
        analyzer.process_results(
            base_folder=base_folder,
            search_test_number=search_test_number,
            criteria_test_number=criteria_test_number
        )
        print(f"Analysis completed for test-{search_test_number} using criteria from test-{criteria_test_number}")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
