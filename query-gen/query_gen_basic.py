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
    temperature: float = 0.1
    json_schema: Dict = None

PATENT_SYSTEM_PROMPT = """You are an expert patent search query strategist specialized in technical prior art searches.
Your role is to generate comprehensive, precise patent search queries that:
1. Identify and capture critical technical components, implied relationships, and synonyms
2. Use appropriate patent search operators and syntax
3. Cover the full scope of potential prior art, including subtle or inferred elements
4. Maintain logical coherence while exploring varied angles
Provide only valid search queries formatted according to the specified structure."""

PATENT_QUERY_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "queries": {
            "type": "array",
            "items": {"type": "string"}
        },
        "explanation": {"type": "string"}
    },
    "required": ["queries", "explanation"]
}

PATENT_QUERY_PROMPT = PromptTemplate(
    title="patentSearchQuery",
    instructions="""
### CONTEXT AND OVERVIEW
The user is providing a description of a product, invention, patent, chemical, process, or other form of technology. This description is used by you to generate search queries that will be used to locate relevant prior art in both patent and non-patent literature.

### Step 1: Core Terms
- Identify exactly one critical term that are central to the invention.
- This terms should have the highest probability of relevance; if a result does not contain at least one of these core terms, it is most likely irrelevant.
- Ensure that the chosen core term represent the main subject, rather than secondary attributes. (e.g., if the invention described a main subject that is comprised of multiple attributes, the attributes are not core terms, the main subject is.) Example: If the user describes a wood-based hydrogel, hydrogel is the core term. Wood is only an attribute.

### Step 2: Supporting Terms
- Create multiple groups of supporting terms, each reflecting a different component, property, or implied concept of the invention (e.g., composition details, mechanical properties, structural layers, applications, products, methods, etc).
- Include synonyms, alternative phrasings, or related concepts (e.g., “switchable” or “tunable” for variable properties, “multi-layer” for layered structures).
- The goal is to diversify coverage across different aspects.

### Step 3: Inferring Domain-Specific Terms
- Dynamically infer additional terms that may not appear explicitly but are commonly associated with the invention’s domain (e.g., “crosslink” in the context of hydrogels).
- Use domain knowledge to identify processes, subcomponents, or descriptors relevant to the invention category or application.
- Incorporate these inferred terms selectively to expand the potential scope of relevant results.

### Step 4: Query Construction
- Generate at least 10 queries, up to a maximum of 16, covering the full range of identified components and angles.
- Each query should have a main focus on a specific aspect of the described subject.
- Ensure that you are making optimal use of words and word counts. Do not use both singular and plural versions of words. For example, hydrogel OR hydrogels is unnessecary, since hydrogel covers both versions.
- Ensure that each query has a balance between specifity and full-scope (e.g., if very specific words or long keyphrases are used, ensure that they have OR groupings and are tied to other variations of the word, or other aspects of the focus of the query)
- Create two types of queries:

  1. **Long Queries (32 words, excluding logical operators AND, OR, and punctuation):**  
     - Start each query with the core term, combining it with multiple supporting groups or deeper variations.
     - Ensure that supporting attributes contain OR groupings to link together different aspects or synonyms and alternative verbiage
     - Connect one or more supporting attribute groups using OR to link synonyms or categorically related terms, using AND to connect each group
     - Example: (primary term OR primary term) AND (attribute OR attribute synonym OR similar attribute OR attribute synonym) AND (attribute two OR attribute two synonym OR attribute two related attribute) AND (primary NEAR attribute)
  2. **Short Queries (around 20 words, excluding logical operators AND, OR, and punctuation):**  
     - Focus on the core term plus exactly one individual supporting group.  
     - Vary the chosen supporting group across these short queries to isolate specific facets and avoid redundancy.
     - Ensure that supporting attributes contain OR groupings to link together different aspects or synonyms and alternative verbiage
     - Example: (primary term OR primary term) AND (attribute OR attribute synonym OR similar attribute OR attribute synonym) AND (primary NEAR attribute)

- Use only AND and OR as logical operators; do not use NEAR or other operators.
- Double-check that all queries adhere to their respective word requirements (32 words for long queries, ~20 words for short queries).

### Output Requirements
1. Return a valid JSON object with two keys:
   - **"queries"**: An array of query strings.
   - **"explanation"**: A concise text explaining how you identified the core term(s), supporting terms, domain-specific inferences, and ensured query diversity.
2. Long queries must strictly contain 32 words (excluding AND, OR, and punctuation).
3. Short queries must be ~20 words (excluding AND, OR, and punctuation).
4. Ensure the minimum number of queries is 10, with no more than 16 total.
5. Avoid near-duplicate queries or overly repetitive terms.

### Target Text Provided by User
{target_text}
""",
    output_format="""
Return a JSON object with the structure:
{
  "queries": [
    "query_string_1",
    ...
    "query_string_N"
  ],
  "explanation": "Short text explaining how you applied the instructions, chose the core term, identified components, and ensured coverage."
}
""",
    system_prompt=PATENT_SYSTEM_PROMPT,
    json_schema=PATENT_QUERY_JSON_SCHEMA
)


class GoogleQueryGenerator:
    def __init__(self, anthropic_api_key: str):
        self.client = anthropic.Client(api_key=anthropic_api_key)
        self.model = "claude-3-5-sonnet-20241022"
        self.test_dir = self._setup_test_directory()

    def _get_next_test_number(self) -> int:
        """Determine the next test number by scanning the queries directory"""
        base_dir = "google_queries"
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
        test_dir = os.path.join("google_queries", f"test-{test_number}")
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

    def generate_queries(self, target_text: str) -> Dict:
        """Generate queries for a target text"""
        try:
            # Query generation
            formatted_instructions = PATENT_QUERY_PROMPT.instructions.format(target_text=target_text)
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                temperature=PATENT_QUERY_PROMPT.temperature,
                system=PATENT_QUERY_PROMPT.system_prompt,
                messages=[
                    {"role": "user", "content": f"{formatted_instructions}\n\n{PATENT_QUERY_PROMPT.output_format}"}
                ]
            )
            
            queries = self._validate_json_response(
                response.content[0].text,
                PATENT_QUERY_PROMPT.json_schema
            )
            
            # Create the results dictionary maintaining the same structure
            results = {
                "instructions": PATENT_QUERY_PROMPT.instructions,
                "target_text": target_text,
                "final_queries": queries     # Same as initial since no review step
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error in query generation: {e}")
            raise

    def save_results(self, results: Dict) -> None:
        """Save the query results to a JSON file"""
        try:
            json_path = os.path.join(self.test_dir, "queries.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to: {json_path}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            raise

def run_google_query_generation(anthropic_api_key: str, target_texts: List[str]) -> None:
    """Run query generation for multiple target texts"""
    for target_text in target_texts:
        try:
            generator = GoogleQueryGenerator(anthropic_api_key=anthropic_api_key)
            results = generator.generate_queries(target_text)
            generator.save_results(results)
            logger.info("Google query generation completed successfully")
        except Exception as e:
            logger.error(f"Google query generation failed: {e}")

if __name__ == "__main__":

    target_texts = [
        """I am requesting a comprehensive global state of the art search regarding a novel room temperature phosphorescent wood hydrogel technology. The invention centers on a unique material that combines natural wood's structural properties with phosphorescent capabilities, creating a versatile biomaterial with switchable mechanical and optical properties.
The core technology utilizes partially delignified wood as a structural skeleton, with polyacrylamide polymerized in situ within the wood structure. The base material starts with basswood that undergoes a specific delignification process, resulting in a composition of approximately 98.3% holocellulose (comprising 88.1% cellulose and 10.2% hemicellulose) and 1.7% lignin. This delignified wood framework is then integrated with polyacrylamide through an in-situ polymerization process, creating a hydrogel with approximately 47.8% water content.
The manufacturing process involves two key stages. First, the wood undergoes delignification through treatment with a solution of 2.5 M NaOH and 0.4 M Na2SO3, followed by exposure to 30% hydrogen peroxide under xenon lamp illumination. After washing with hot deionized water and ethanol, the delignified wood is soaked in an acrylamide solution (0.34 g/mL) and thermally polymerized at 60°C for two hours to form the final hydrogel.
The resulting material exhibits remarkable properties that distinguish it from existing hydrogels. It demonstrates a base tensile strength of 38.4 MPa, which can be increased to 153.8 MPa through ethanol treatment. The material shows room temperature phosphorescence centered at 490 nm with a lifetime of 32.5 ms, which can be extended to 69.7 ms following ethanol treatment. Furthermore, the hydrogel can serve as an energy donor when combined with rhodamine B, producing red afterglow emission at 600 nm with energy transfer efficiency reaching 77.8%.
A particularly notable feature is the material's reversible property modification through alternating ethanol and water treatments. This allows switching between flexible and rigid states while maintaining phosphorescent properties. The material can be processed into various forms including threads, textiles, and complex 3D structures, suggesting potential applications in medical sutures and other biomedical applications.
In conducting this search, particular attention should be paid to existing technologies involving wood-based hydrogels with phosphorescent properties, methods for creating mechanically strong hydrogels using natural wood components, systems for reversible property modification of hydrogels through solvent treatment, and luminescent hydrogel materials intended for medical applications.
The material operates at room temperature and maintains its properties under normal atmospheric conditions in both wet and ethanol-treated states. Please include any relevant prior art that might address similar approaches to creating multi-functional biomaterials with switchable properties."""
    ]

    anthropic_api_key = ""
    
    run_google_query_generation(anthropic_api_key, target_texts)
