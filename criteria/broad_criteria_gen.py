import anthropic
import os
import json
import logging
from typing import Dict, List
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PromptTemplate:
    title: str
    instructions: str
    output_format: str
    system_prompt: str
    temperature: float = 0
    json_schema: Dict = None

CRITERIA_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "potential_relevance_indicators": {
            "type": "array",
            "items": {"type": "string"}
        }
    },
    "required": ["potential_relevance_indicators"]
}

CRITERIA_SYSTEM_PROMPT = """
You are a technical analysis specialist identifying broad topics or areas that may indicate potential relevance to the target technology.
Your role:
1. List straightforward points or keywords that might show overlapping relevance.
2. Use general terms rather than strict parameters or thresholds.
3. Ensure the list is clear and easy to scan quickly.
4. Avoid overly technical detail; focus on capturing possible subject-matter overlaps.
"""

CRITERIA_REVIEW_SYSTEM_PROMPT = """
You are an expert reviewer ensuring the bullet-point criteria fully capture potential relevance without unnecessary detail.
Your role:
1. Check that each point remains broad and indicative of relevance.
2. Remove or simplify any overly detailed points.
3. Add any missing high-level topics or concepts.
4. Maintain a concise list for quick relevance scanning.
"""

CRITERIA_PROMPT = PromptTemplate(
    title="priorArtCriteriaGeneration",
    instructions="""
### PRIOR ART CRITERIA GENERATION

# Instructions:
Given a technology description, list in bullet form any mentions, references, or topics that might indicate potential relevance to the subject. Focus on high-level topics, not strict technical requirements.

1. **High-Level Topics**  
   - Any conceptual area, process, or material that could overlap with the target.

2. **Concise Bullet Points**  
   - Do not include detailed measurements or thresholds.

### Target Text:
{target_text}
""",
    output_format="""Return a JSON object containing a single array of bullet points:
{
    "explanation": "Short explanation of why these topics might indicate relevance.",
    "potential_relevance_indicators": [
        "bullet point 1",
        "bullet point 2"
    ]
}""",
    system_prompt=CRITERIA_SYSTEM_PROMPT,
    json_schema=CRITERIA_JSON_SCHEMA
)

CRITERIA_REVIEW_PROMPT = PromptTemplate(
    title="priorArtCriteriaReview",
    instructions="""
### PRIOR ART CRITERIA REVIEW

# Instructions:
Review and refine the bullet-point list of relevance indicators.

Original Target Text:
{target_text}

Generated Indicators:
{prev_response}

# Validation Steps:
1. **Ensure Broad Coverage**  
   - Confirm points capture general or indirect references.

2. **Trim Over-Specifics**  
   - Remove or adjust any overly detailed items.

3. **Add Missing Topics**  
   - Include any areas or concepts missing but potentially relevant.

""",
    output_format="""Return a JSON object in the same format:
{
    "potential_relevance_indicators": [
        "updated bullet point 1",
        "updated bullet point 2"
    ]
}""",
    system_prompt=CRITERIA_REVIEW_SYSTEM_PROMPT,
    json_schema=CRITERIA_JSON_SCHEMA
)


class CriteriaAnalyzer:
    def __init__(self, anthropic_api_key: str):
        self.client = anthropic.Client(api_key=anthropic_api_key)
        self.model = "claude-3-5-sonnet-20241022"
        self.test_dir = self._setup_test_directory()

    def _get_next_test_number(self) -> int:
        """Determine the next test number by scanning the criteria directory"""
        base_dir = "criteria"
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
            return 1
        
        existing_tests = [d for d in os.listdir(base_dir) 
                         if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("test-")]
        if not existing_tests:
            return 1
        
        test_numbers = [int(d.split("-")[1]) for d in existing_tests]
        return max(test_numbers) + 1

    def _setup_test_directory(self) -> str:
        """Create and return the path for the next test directory"""
        test_number = self._get_next_test_number()
        test_dir = os.path.join("criteria", f"test-{test_number}")
        os.makedirs(test_dir, exist_ok=True)
        return test_dir

    def _validate_json_response(self, response_text: str, schema: Dict) -> Dict:
        """Parse JSON response with basic cleaning"""
        try:
            # Clean and normalize the text
            cleaned_text = ' '.join(response_text.split())
            start_idx = cleaned_text.find('{')
            end_idx = cleaned_text.rfind('}') + 1
            
            if start_idx >= 0 and end_idx > 0:
                cleaned_text = cleaned_text[start_idx:end_idx]
            
            return json.loads(cleaned_text)
                
        except Exception as e:
            logger.error(f"Error parsing JSON: {e}")
            raise

    def analyze_criteria(self, target_text: str) -> Dict:
        """Generate and review criteria for a target text"""
        try:
            # Initial criteria generation
            formatted_instructions = CRITERIA_PROMPT.instructions.format(target_text=target_text)
            initial_response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                temperature=CRITERIA_PROMPT.temperature,
                system=CRITERIA_PROMPT.system_prompt,
                messages=[{
                    "role": "user", 
                    "content": formatted_instructions + "\n\n" + CRITERIA_PROMPT.output_format
                }]
            )
            
            initial_criteria = self._validate_json_response(
                initial_response.content[0].text,
                CRITERIA_PROMPT.json_schema
            )
            
            # Criteria review
            review_vars = {
                "prev_instructions": CRITERIA_PROMPT.instructions,
                "target_text": target_text,
                "prev_response": json.dumps(initial_criteria, indent=2)
            }
            
            formatted_review = CRITERIA_REVIEW_PROMPT.instructions.format(**review_vars)
            review_response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                temperature=CRITERIA_REVIEW_PROMPT.temperature,
                system=CRITERIA_REVIEW_PROMPT.system_prompt,
                messages=[{
                    "role": "user", 
                    "content": formatted_review + "\n\n" + CRITERIA_REVIEW_PROMPT.output_format
                }]
            )
            
            final_criteria = self._validate_json_response(
                review_response.content[0].text,
                CRITERIA_REVIEW_PROMPT.json_schema
            )
            
            return {
                "initial_criteria": initial_criteria,
                "final_criteria": final_criteria
            }
            
        except Exception as e:
            logger.error(f"Error in criteria analysis: {e}")
            raise

    def save_results(self, results: Dict) -> None:
        """Save the criteria results to a JSON file"""
        try:
            json_path = os.path.join(self.test_dir, "criteria_analysis.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to: {json_path}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            raise

def run_criteria_analysis(anthropic_api_key: str, target_texts: List[str]) -> None:
    """Run criteria analysis for multiple target texts"""
    for target_text in target_texts:
        try:
            analyzer = CriteriaAnalyzer(anthropic_api_key=anthropic_api_key)
            results = analyzer.analyze_criteria(target_text)
            analyzer.save_results(results)
            logger.info("Criteria analysis completed successfully")
        except Exception as e:
            logger.error(f"Criteria analysis failed: {e}")


if __name__ == "__main__":
    # Example usage
    anthropic_api_key = ""

    # Example target text
    target_texts = ["""I am requesting a comprehensive global state of the art search regarding a novel room temperature phosphorescent wood hydrogel technology. The invention centers on a unique material that combines natural wood's structural properties with phosphorescent capabilities, creating a versatile biomaterial with switchable mechanical and optical properties.
The core technology utilizes partially delignified wood as a structural skeleton, with polyacrylamide polymerized in situ within the wood structure. The base material starts with basswood that undergoes a specific delignification process, resulting in a composition of approximately 98.3% holocellulose (comprising 88.1% cellulose and 10.2% hemicellulose) and 1.7% lignin. This delignified wood framework is then integrated with polyacrylamide through an in-situ polymerization process, creating a hydrogel with approximately 47.8% water content.
The manufacturing process involves two key stages. First, the wood undergoes delignification through treatment with a solution of 2.5 M NaOH and 0.4 M Na2SO3, followed by exposure to 30% hydrogen peroxide under xenon lamp illumination. After washing with hot deionized water and ethanol, the delignified wood is soaked in an acrylamide solution (0.34 g/mL) and thermally polymerized at 60°C for two hours to form the final hydrogel.
The resulting material exhibits remarkable properties that distinguish it from existing hydrogels. It demonstrates a base tensile strength of 38.4 MPa, which can be increased to 153.8 MPa through ethanol treatment. The material shows room temperature phosphorescence centered at 490 nm with a lifetime of 32.5 ms, which can be extended to 69.7 ms following ethanol treatment. Furthermore, the hydrogel can serve as an energy donor when combined with rhodamine B, producing red afterglow emission at 600 nm with energy transfer efficiency reaching 77.8%.
A particularly notable feature is the material's reversible property modification through alternating ethanol and water treatments. This allows switching between flexible and rigid states while maintaining phosphorescent properties. The material can be processed into various forms including threads, textiles, and complex 3D structures, suggesting potential applications in medical sutures and other biomedical applications.
In conducting this search, particular attention should be paid to existing technologies involving wood-based hydrogels with phosphorescent properties, methods for creating mechanically strong hydrogels using natural wood components, systems for reversible property modification of hydrogels through solvent treatment, and luminescent hydrogel materials intended for medical applications.
The material operates at room temperature and maintains its properties under normal atmospheric conditions in both wet and ethanol-treated states. Please include any relevant prior art that might address similar approaches to creating multi-functional biomaterials with switchable properties."""]
    
    run_criteria_analysis(anthropic_api_key, target_texts)
