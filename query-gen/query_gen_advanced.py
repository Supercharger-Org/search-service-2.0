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
2. Use appropriate patent search operators and syntax, including AND, OR, and NEAR
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
    title="expandedPatentSearchQuery",
    instructions="""
### OVERVIEW AND CONTEXT
The user is providing a description of a product, invention, patent, chemical, process, or other form of technology. This description is used by you to generate search queries that will be used to locate relevant prior art in both patent and non-patent literature.

### Step 1: Core Terms
- Identify one critical term that are central to the invention.
- This core term should have the highest probability of relevance; if a result does not contain at least one of these, it is most likely irrelevant.
- Ensure you select the actual main subject (e.g., if a wood-based hydrogel is described, hydrogel is the core term, not wood. Wood would be considered an attribute)

### Step 2: Supporting Terms
- Create multiple groups of supporting terms, each reflecting a distinct component, property, or implied concept of the invention (e.g., composition details, mechanical properties, structural layers, applications).
- Consider synonyms, alternative phrasings, or implicit references (e.g., “switchable” or “tunable” for variable properties, “multi-layer” for layered structures).
- Use these groups strategically to diversify query coverage without universally repeating any term that is not a core term.

### Step 3: Inferring Domain-Specific Terms
- Dynamically infer additional terms relevant to the invention’s domain that may not appear explicitly in the target text (e.g., “crosslink” for hydrogels).
- Use subject-matter knowledge to identify related mechanisms, processes, or descriptors.
- Incorporate these inferred terms selectively into queries to broaden the scope of potentially relevant results.

### Step 4: Query Construction
1. Generate at least 10 queries (minimum), up to a maximum of 20, ensuring comprehensive coverage of all identified components, angles, and domain inferences.
- Each query should have a main focus on a specific aspect of the described subject.
- Ensure that you are making optimal use of words and word counts. Do not use both singular and plural versions of words. For example, hydrogel OR hydrogels is unnessecary, since hydrogel covers both versions.
- Ensure that each query has a balance between specifity and full-scope (e.g., if very specific words or long keyphrases are used, ensure that they have OR groupings and are tied to other variations of the word, or other aspects of the focus of the query)
2. Create two types of queries (with some NEAR-focused queries):
   - **Long Queries (e.g., 35 words, excluding AND, OR, NEAR, and punctuation):**
     - Use the core term to start each query
     - Connect one or more supporting attribute groups using OR to link synonyms or categorically related terms, using AND to connect each group
     - Ensure that supporting attributes contain OR groupings to link together different aspects or synonyms and alternative verbiage
     - Incorporate NEAR selectively if it is beneficial for capturing closely related terms.
     - Example: (primary term OR primary term) AND (attribute OR attribute synonym OR similar attribute OR attribute synonym) AND (attribute two OR attribute two synonym OR attribute two related attribute) AND (primary NEAR attribute)
   - **Short Queries (around 20–25 words, excluding AND, OR, NEAR, and punctuation):**
     - Use the core term to start each query
     - Connect one singular supporting attribute and provide a wide range words that fall into this category using OR ruling
     - Incorporate NEAR selectively if it is beneficial for capturing closely related terms.
     - Ensure that supporting attributes contain OR groupings to link together different aspects or synonyms and alternative verbiage
     - Example: (primary term OR primary term) AND (attribute OR attribute synonym OR similar attribute OR attribute synonym) AND (primary NEAR attribute)
3. Begin each query with the chosen core term; avoid building all queries around multiple core terms.
4. Vary synonyms and query structures; do not use the same supporting or inferred terms in all queries.
5. Use only AND, OR, and NEAR as logical operators.

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
  "explanation": "Short text explaining how you applied the instructions, chose the core term, identified components, and leveraged NEAR."
}
""",
    system_prompt=PATENT_SYSTEM_PROMPT,
    json_schema=PATENT_QUERY_JSON_SCHEMA
)

class PatentQueryGenerator:
    def __init__(self, anthropic_api_key: str):
        self.client = anthropic.Client(api_key=anthropic_api_key)
        self.model = "claude-3-5-sonnet-20241022"
        self.test_dir = self._setup_test_directory()

    def _get_next_test_number(self) -> int:
        """Determine the next test number by scanning the queries directory"""
        base_dir = "patent_queries"
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
        test_dir = os.path.join("patent_queries", f"test-{test_number}")
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
                "initial_queries": queries,  # Keep for structure consistency
                "final_queries": queries     # Same as initial since no review step
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error in query generation: {e}")
            raise

    def save_results(self, results: Dict) -> None:
        """Save the query results to a JSON file"""
        try:
            json_path = os.path.join(self.test_dir, "patent_queries.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to: {json_path}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            raise

def run_patent_query_generation(anthropic_api_key: str, target_texts: List[str]) -> None:
    """Run patent query generation for multiple target texts"""
    for target_text in target_texts:
        try:
            generator = PatentQueryGenerator(anthropic_api_key=anthropic_api_key)
            results = generator.generate_queries(target_text)
            generator.save_results(results)
            logger.info("Patent query generation completed successfully")
        except Exception as e:
            logger.error(f"Patent query generation failed: {e}")

if __name__ == "__main__":
    target_texts = [
        """Title: Engineered microbial population
Abstract: Provided herein are genetically engineered bacteria that are native to a host insect microbiome. Further provided are methods of inducing RNA interference in an insect, such as a bee, by administering the genetically engineered bacteria.
Claims:
What is claimed is:1. A bee-ingestible microbial composition comprising one or more bacteria genetically engineered to express at least one heterologous nucleic acid, wherein the heterologous nucleic acid is heterologous to a host bee, and further wherein the one or more bacteria are selected from the group consisting of Snodgrassella alvi, Bartonella apis, Gilliamella apicola, and Parasaccharibacter apium.
2. The composition of claim 1, wherein the host bee is selected from the group consisting of a honey bee and a bumble bee.
3. The composition of claim 1, wherein the composition comprises 2, 3, 4, or 5 bacterial species selected from the group consisting of Snodgrassella alvi, Bartonella apis, Gilliamella apicola, Serratia marcescens, and Parasaccharibacter apium.
4. The composition of claim 1, wherein the one or more bacteria express at least two heterologous nucleic acids.
5. The composition of claim 1, wherein the heterologous nucleic acid encodes a polypeptide.
6. The composition of claim 5, wherein the heterologous nucleic acid encodes a pesticide degrading polypeptide or a cytochrome.
7. The composition of claim 1, wherein the heterologous nucleic acid is an inhibitory nucleic acid.
8. The composition of claim 7, wherein the inhibitory nucleic acid is selected from the group consisting of an antisense DNA, dsRNA, siRNA, shRNA, sgRNA and a miRNA.
9. The composition of claim 1, wherein the composition comprises a broad host range plasmid that either comprises or can express the at least one heterologous nucleic acid.
10. The composition of claim 9, wherein the broad host range plasmid comprises at least one regulatory sequence selected from the group consisting of an RSF 1010 origin of replication, a PA1 promoter sequence, a PA2 promoter sequence, a PA3 promoter sequence, a cp25 promoter sequence, and a detectable marker.
11. The composition of claim 1, wherein the composition comprises at least one selected from the group consisting of:
a) a live suspension comprising the bacteria,
b) a lyophilized powder comprising the bacteria,
c) a solid comprising the bacteria,
d) a liquid comprising the bacteria,
e) protein,
f) pollen,
g) a sucrose solution comprising the bacteria, and
h) a corn syrup solution comprising the bacteria.
12. The composition of claim 11, further comprising a carbohydrate or sugar supplement.
13. A bee comprising the composition of claim 1.
14. The bee of claim 13, wherein the bee is a honey bee."""
    ]
    anthropic_api_key = ""
    
    run_patent_query_generation(anthropic_api_key, target_texts)
