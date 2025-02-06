import os
import json
import logging
import time
from typing import Dict, List, Optional
import anthropic
import statistics
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('patent_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

INPUT_COST_PER_MILLION = 3.0  # Claude 3.5 Sonnet input cost
OUTPUT_COST_PER_MILLION = 15.0  # Claude 3.5 Sonnet output cost

def calculate_token_cost(token_count: int, cost_per_million: float) -> float:
    """Calculate cost for tokens based on per-million rate"""
    return (token_count / 1_000_000) * cost_per_million

class PatentAnalyzer:
    def __init__(self, anthropic_api_key: str, target_text: str):
        self.client = anthropic.Client(api_key=anthropic_api_key)
        self.model = "claude-3-5-sonnet-20241022"
        self.target_text = target_text
        self.current_output_dir = None

    def get_next_test_number(self, output_base: str) -> int:
        """Determine the next test number based on existing folders"""
        try:
            if not os.path.exists(output_base):
                os.makedirs(output_base)
                return 1
                
            test_folders = [d for d in os.listdir(output_base) if d.startswith('test-')]
            if not test_folders:
                return 1
                
            numbers = [int(folder.split('-')[1]) for folder in test_folders]
            return max(numbers) + 1
        except Exception as e:
            logger.error(f"Error determining next test number: {str(e)}")
            raise

    def save_intermediate_results(self, consolidated_data: Dict, output_dir: str) -> None:
        """Save intermediate results to avoid data loss"""
        try:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            output_path = os.path.join(output_dir, f'intermediate_results_{timestamp}.json')
            
            with open(output_path, 'w') as f:
                json.dump(consolidated_data, f, indent=2)
            
            logger.info(f"Saved intermediate results to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save intermediate results: {str(e)}")
            # Continue execution even if saving fails

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((anthropic.APIError, anthropic.APIConnectionError, anthropic.RateLimitError)),
        after=lambda retry_state: logger.info(f"Retrying API call after {retry_state.attempt_number} attempts...")
    )
    def load_patent_results(self, test_number: int) -> Dict:
        """Load patent results with retry logic"""
        results_path = os.path.join('patent_results', f'test-{test_number}', 'patent_results.json')
        try:
            with open(results_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading patent results from {results_path}: {str(e)}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((anthropic.APIError, anthropic.APIConnectionError, anthropic.RateLimitError)),
        after=lambda retry_state: logger.info(f"Retrying API call after {retry_state.attempt_number} attempts...")
    )
    def load_criteria(self, criteria_test_number: int) -> List[str]:
        """Load criteria analysis with retry logic"""
        criteria_path = os.path.join('criteria', f'test-{criteria_test_number}', 'criteria_analysis.json')
        try:
            with open(criteria_path, 'r') as f:
                data = json.load(f)
                return data['final_criteria']['potential_relevance_indicators']
        except Exception as e:
            logger.error(f"Error loading criteria from {criteria_path}: {str(e)}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((anthropic.APIError, anthropic.APIConnectionError, anthropic.RateLimitError)),
        after=lambda retry_state: logger.info(f"Retrying API call after {retry_state.attempt_number} attempts...")
    )
    def analyze_patent(self, patent: Dict, criteria: List[str]) -> Optional[Dict]:
        """Analyze a single patent with retry logic and error handling"""
        try:
            system_prompt = """You are an expert in technical patent analysis and prior art assessment. Your task is to:
1. Analyze the given patent information in relation to the target text and general criteria
2. Provide an analysis of potential relevance based on the available information
3. Provide a potential relevance score based on your analysis

The Patent Information being sent is a result of a prior art search that was created for the target text. We are looking to gauge the potential relevance of the results"""

prompt = f"""You are tasked with analyzing the potential technical relevance of search result snippets in relation to a target description of a product, invention, technology, or patent. Your goal is to evaluate these snippets against provided technical relevance criteria and assign a relevance score based on the information available.

### Context:
You are provided with three key inputs to perform your analysis:
1. **User Input**: This is a detailed description of an item, product, invention, technology, or patent for which we are conducting a prior art search. This serves as the core reference point for your assessment.
2. **Criteria Outline**: A list of potential relevance indicators that has been developed based on the user's input. These indicators define what constitutes potential relevance to the target text.
3. **Search Result Snippets**: These are short excerpts (e.g., abstracts or snippets) extracted from search results obtained during the prior art search process. These snippets represent incomplete portions of the result, and you must base your analysis solely on the information provided in the snippet and abstract.

The ultimate goal is to determine the potential relevance of the snippet by evaluating its alignment with the target text and the criteria. This is not a binary process of declaring something fully relevant or irrelevant. Instead, your task is to indicate the degree of potential relevance using a nuanced approach:
- Highlight overlaps, even partial ones, between the patent and the criteria.
- Recognize synonyms, alternative phrasing, or related concepts as potential indicators of relevance.

### Goals:
- Provide a balanced analysis that identifies potential overlaps or gaps between the snippet and the target text or criteria.
- Assign a relevance score to reflect the degree of alignment:
  - If the patent does not indicate any potential relevance, score it low.
  - If the patent addresses the same subject matter broadly but does not explicitly match the target, score it in the low-mid-range.
  - If the patent addresses the same subject matter and some of the same facets but does not explicity provide enough detail to showcase guaranteed relevance, score it in the high mid-range.
  - If the patent strongly aligns with the criteria and subject matter of the target text, score it high.
- The analysis and scoring should help guide further examination of the results by identifying items with meaningful potential relevance.

### Output:
Provide your response as a JSON object containing:
- **analysis**: A concise description of the snippet's potential relevance, noting specific overlaps or areas of alignment with the target text and criteria.
- **score**: A relevance score between 0.0 and 1.0, where:
  - 0.0 = Technically irrelevant
  - 0.5 = Some overlap or potential relevance
  - 1.0 = High technical relevance

Target Text:
{self.target_text}

Potential Relevance Indicators:
{json.dumps(criteria, indent=2)}

Patent Information:
Title: {patent.get('title', 'No title')}
Abstract: {patent.get('abstract', 'No abstract')}
First Claim: {patent.get('claim', 'No claim')}
Search Snippet: {patent.get('snippet', 'No snippet')}

Please analyze this patent's technical relevance and provide a relevance score."""

            format_instructions = """Provide your response as a JSON object with this structure:
{
    "analysis": "Detailed analysis of technical similarities and differences",
    "score": float (0.0 to 1.0, where 0 = technically irrelevant, 0.5 = some technical overlap, 1.0 = high technical relevance)
}"""

            response = self.client.messages.create(
                model=self.model,
                max_tokens=1000,
                temperature=0,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": f"{prompt}\n\n{format_instructions}"}
                ]
            )

            response_text = response.content[0].text
            analysis_data = json.loads(response_text)

            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            
            input_cost = calculate_token_cost(input_tokens, INPUT_COST_PER_MILLION)
            output_cost = calculate_token_cost(output_tokens, OUTPUT_COST_PER_MILLION)

            patent.update({
                "analysis": analysis_data["analysis"],
                "score": analysis_data["score"],
                "analysis_token_totals": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_cost": round(input_cost + output_cost, 6)
                }
            })

            return patent

        except Exception as e:
            logger.error(f"Error analyzing patent {patent.get('publication_number', 'unknown')}: {str(e)}")
            # Return a partial result instead of None to preserve data
            patent.update({
                "analysis": "Error during analysis",
                "score": 0.0,
                "analysis_error": str(e),
                "analysis_token_totals": {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_cost": 0
                }
            })
            return patent

    def process_results(self, patent_test_number: int, criteria_test_number: int):
        """Process all patents with comprehensive error handling and state preservation"""
        consolidated_data = None
        criteria = None
        processed_count = 0
        
        try:
            # Load all necessary data
            consolidated_data = self.load_patent_results(patent_test_number)
            criteria = self.load_criteria(criteria_test_number)
            
            # Setup output directory
            output_base = "analyzed_patents"
            next_test_num = self.get_next_test_number(output_base)
            self.current_output_dir = os.path.join(output_base, f'test-{next_test_num}')
            os.makedirs(self.current_output_dir, exist_ok=True)
            
            # Initialize tracking
            total_input_tokens = 0
            total_output_tokens = 0
            total_cost = 0
            
            total_results = len(consolidated_data["results"])
            save_frequency = max(1, min(50, total_results // 10)) 
            
            for i, result in enumerate(consolidated_data["results"]):
                try:
                    analyzed_patent = self.analyze_patent(result, criteria)
                    if analyzed_patent:
                        token_totals = analyzed_patent["analysis_token_totals"]
                        total_input_tokens += token_totals["input_tokens"]
                        total_output_tokens += token_totals["output_tokens"]
                        total_cost += token_totals["total_cost"]
                        processed_count += 1


                    if (i + 1) % save_frequency == 0:
                        logger.info(f"Processed {i + 1}/{total_results} patents")
                        self.save_intermediate_results(consolidated_data, self.current_output_dir)

                except Exception as e:
                    logger.error(f"Error processing patent {i}: {str(e)}")
                    continue  # Continue with next patent even if one fails

            # Sort results by score
            consolidated_data["results"].sort(key=lambda x: x.get('score', 0), reverse=True)

            # Calculate median score of top 20 results
            top_20_scores = [r.get('score', 0) for r in consolidated_data["results"][:20]]
            median_score = statistics.median(top_20_scores) if top_20_scores else 0

            # Prepare final output
            final_output = {
                "median_score": median_score,
                "analysis_token_totals": {
                    "input_tokens": total_input_tokens,
                    "output_tokens": total_output_tokens,
                    "total_cost": round(total_cost, 6)
                },
                "criteria_test_number": criteria_test_number,
                "processed_patents": processed_count,
                "total_patents": total_results,
                **consolidated_data
            }

            # Save final results
            output_path = os.path.join(self.current_output_dir, 'analyzed_results.json')
            with open(output_path, 'w') as f:
                json.dump(final_output, f, indent=2)

            logger.info(f"Analysis completed")
            logger.info(f"Median score of top 20 results: {median_score}")
            logger.info(f"Total input tokens: {total_input_tokens}")
            logger.info(f"Total output tokens: {total_output_tokens}")
            logger.info(f"Total cost: ${round(total_cost, 6)}")
            logger.info(f"Successfully processed {processed_count}/{total_results} patents")

        except Exception as e:
            logger.error(f"Catastrophic error in process_results: {str(e)}")
            if consolidated_data:
                try:
                    # Save whatever we have in case of catastrophic failure
                    error_output = {
                        "error": str(e),
                        "processed_patents": processed_count,
                        "analysis_token_totals": {
                            "input_tokens": total_input_tokens,
                            "output_tokens": total_output_tokens,
                            "total_cost": round(total_cost, 6) if 'total_cost' in locals() else 0
                        },
                        **consolidated_data
                    }
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    error_path = os.path.join(self.current_output_dir or "error_output", f'error_results_{timestamp}.json')
                    os.makedirs(os.path.dirname(error_path), exist_ok=True)
                    with open(error_path, 'w') as f:
                        json.dump(error_output, f, indent=2)
                    logger.error(f"Saved error state to {error_path}")
                except Exception as save_error:
                    logger.critical(f"Failed to save error state: {save_error}")
            raise  # Re-raise the original error after saving state

def main():
    # Define your variables here
    patent_test_number = 21 
    criteria_test_number = 6  
    anthropic_api_key = ""
    
    # Define your target text here
    target_text = """I am requesting a comprehensive global state of the art search regarding a novel room temperature phosphorescent wood hydrogel technology. The invention centers on a unique material that combines natural wood's structural properties with phosphorescent capabilities, creating a versatile biomaterial with switchable mechanical and optical properties.
The core technology utilizes partially delignified wood as a structural skeleton, with polyacrylamide polymerized in situ within the wood structure. The base material starts with basswood that undergoes a specific delignification process, resulting in a composition of approximately 98.3% holocellulose (comprising 88.1% cellulose and 10.2% hemicellulose) and 1.7% lignin. This delignified wood framework is then integrated with polyacrylamide through an in-situ polymerization process, creating a hydrogel with approximately 47.8% water content.
The manufacturing process involves two key stages. First, the wood undergoes delignification through treatment with a solution of 2.5 M NaOH and 0.4 M Na2SO3, followed by exposure to 30% hydrogen peroxide under xenon lamp illumination. After washing with hot deionized water and ethanol, the delignified wood is soaked in an acrylamide solution (0.34 g/mL) and thermally polymerized at 60°C for two hours to form the final hydrogel.
The resulting material exhibits remarkable properties that distinguish it from existing hydrogels. It demonstrates a base tensile strength of 38.4 MPa, which can be increased to 153.8 MPa through ethanol treatment. The material shows room temperature phosphorescence centered at 490 nm with a lifetime of 32.5 ms, which can be extended to 69.7 ms following ethanol treatment. Furthermore, the hydrogel can serve as an energy donor when combined with rhodamine B, producing red afterglow emission at 600 nm with energy transfer efficiency reaching 77.8%.
A particularly notable feature is the material's reversible property modification through alternating ethanol and water treatments. This allows switching between flexible and rigid states while maintaining phosphorescent properties. The material can be processed into various forms including threads, textiles, and complex 3D structures, suggesting potential applications in medical sutures and other biomedical applications.
In conducting this search, particular attention should be paid to existing technologies involving wood-based hydrogels with phosphorescent properties, methods for creating mechanically strong hydrogels using natural wood components, systems for reversible property modification of hydrogels through solvent treatment, and luminescent hydrogel materials intended for medical applications.
The material operates at room temperature and maintains its properties under normal atmospheric conditions in both wet and ethanol-treated states. Please include any relevant prior art that might address similar approaches to creating multi-functional biomaterials with switchable properties."""
    
    try:
        analyzer = PatentAnalyzer(anthropic_api_key, target_text)
        analyzer.process_results(patent_test_number, criteria_test_number)
        logger.info(f"Analysis completed for patent test-{patent_test_number} using criteria from test-{criteria_test_number}")
    except KeyboardInterrupt:
        logger.warning("Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error in main: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
