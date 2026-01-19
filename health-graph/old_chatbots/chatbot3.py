import os
import re
from langchain_community.graphs import Neo4jGraph
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings
from langchain_experimental.llms.ollama_functions import OllamaFunctions
load_dotenv()

# --- Configuration ---
NEO4J_URI = os.environ.get("NEO4J_URI")
NEO4J_USER = os.environ.get("NEO4J_USER")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

# 1. Initialize Graph Connection
graph = Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USER,
    password=NEO4J_PASSWORD
)

print("="*70)
print("📊 Graph Schema:")
print("="*70)
print(graph.schema)
print("="*70)

# 2. Initialize LLM
# llm = ChatGoogleGenerativeAI(
#     model="gemini-2.0-flash-exp",
#     temperature=0, 
#     google_api_key=GOOGLE_API_KEY
# )
llm = OllamaFunctions(model="llama3.1", temperature=0, format="json")



# ============================================
# Entity Extraction
# ============================================

def extract_entities_from_question(question: str) -> dict:
    """
    Extract key entities from the question using LLM.
    Returns: {
        'topics': list of medical/audit concepts,
        'frameworks': list of frameworks mentioned,
        'query_type': 'comparison' | 'search' | 'specific'
    }
    """
    
    entity_extraction_prompt = f"""Extract key information from this hospital audit question:

Question: "{question}"

Extract:
1. Topics/Concepts: Medical or audit terms (e.g., "hand hygiene", "medication safety", "patient identification")
2. Frameworks: JCI, SHCC, or both
3. Query Type: 
   - "comparison" if comparing frameworks
   - "search" if looking for requirements
   - "specific" if asking about a specific clause code

Return ONLY a JSON object with this format:
{{
    "topics": ["topic1", "topic2"],
    "frameworks": ["JCI", "SHCC"],
    "query_type": "comparison"
}}

JSON:"""
    
    try:
        response = llm.invoke(entity_extraction_prompt)
        result_text = response.content.strip()
        
        # Clean up response
        result_text = result_text.replace("```json", "").replace("```", "").strip()
        
        # Parse JSON
        import json
        entities = json.loads(result_text)
        
        return entities
    except Exception as e:
        print(f"⚠️  Entity extraction failed: {e}")
        # Fallback: simple keyword extraction
        return {
            "topics": [],
            "frameworks": ["JCI", "SHCC"] if "compare" in question.lower() else ["JCI", "SHCC"],
            "query_type": "comparison" if "compar" in question.lower() else "search"
        }

# ============================================
# Cypher Generation with Entities
# ============================================

def generate_cypher_with_entities(question: str, entities: dict) -> str:
    """Generate optimized Cypher query using extracted entities."""
    
    topics = entities.get('topics', [])
    frameworks = entities.get('frameworks', ['JCI', 'SHCC'])
    query_type = entities.get('query_type', 'search')
    
    # Build topic search conditions
    topic_conditions = []
    for topic in topics:
        # More flexible matching
        topic_lower = topic.lower().replace(' ', '.*')
        topic_conditions.append(f"toLower(co.name) =~ '.*{topic_lower}.*' OR toLower(co.label) =~ '.*{topic_lower}.*'")
    
    topic_where = " OR ".join(topic_conditions) if topic_conditions else "TRUE"
    
    # Generate query based on type
    if query_type == "comparison" and len(frameworks) >= 2:
        # Framework comparison using SIMILAR_TO
        cypher = f"""
// Step 1: Find concepts matching topics
MATCH (co:Concept)
WHERE {topic_where}

// Step 2: Find clauses mentioning these concepts
MATCH (jci:Clause {{framework: '{frameworks[0]}'}})-[:MENTIONS]->(co)
MATCH (shcc:Clause {{framework: '{frameworks[1]}'}})-[:MENTIONS]->(co)

// Step 3: Check if they're similar
OPTIONAL MATCH (jci)-[s:SIMILAR_TO]->(shcc)

// Return everything
RETURN DISTINCT
    jci.code AS jci_code, 
    jci.text AS jci_text,
    shcc.code AS shcc_code,
    shcc.text AS shcc_text,
    s.score AS similarity_score,
    co.name AS concept_name,
    co.label AS concept_label
ORDER BY similarity_score DESC
LIMIT 20
"""
    else:
        # General topic search
        framework_filter = " OR ".join([f"cl.framework = '{fw}'" for fw in frameworks])
        
        cypher = f"""
// Find concepts matching topics
MATCH (co:Concept)
WHERE {topic_where}

// Find clauses mentioning these concepts
MATCH (cl:Clause)-[:MENTIONS]->(co)
WHERE {framework_filter}

// Return results
RETURN DISTINCT
    cl.code AS code,
    cl.text AS text,
    cl.framework AS framework,
    co.name AS concept_name,
    co.label AS concept_label
LIMIT 20
"""
    
    return cypher.strip()

# ============================================
# Execute and Display Results
# ============================================

def execute_and_display_results(cypher: str):
    """Execute Cypher and display formatted results."""
    
    try:
        results = graph.query(cypher)
        
        if not results:
            print("\n❌ No results found.")
            return
        
        print(f"\n✅ Retrieved {len(results)} results:")
        print("="*70)
        
        # Group by framework for comparisons
        jci_results = []
        shcc_results = []
        
        for i, record in enumerate(results, 1):
            # Check if this is a comparison result
            if 'jci_code' in record and 'shcc_code' in record:
                print(f"\n📊 Match #{i}:")
                print("-"*70)
                
                # JCI side
                print(f"\n📘 JCI Clause: {record.get('jci_code', 'N/A')}")
                print(f"   Text: {record.get('jci_text', '')[:300]}...")
                
                # SHCC side
                print(f"\n📗 SHCC Clause: {record.get('shcc_code', 'N/A')}")
                print(f"   Text: {record.get('shcc_text', '')[:300]}...")
                
                # Similarity
                sim_score = record.get('similarity_score')
                if sim_score:
                    print(f"\n🔗 Similarity Score: {sim_score:.3f}")
                    if sim_score > 0.90:
                        print("   → Nearly identical requirements")
                    elif sim_score > 0.80:
                        print("   → Highly similar with minor variations")
                    elif sim_score > 0.75:
                        print("   → Similar core requirements")
                else:
                    print("\n🔗 No direct similarity link found")
                
                # Concept
                concept = record.get('concept_name') or record.get('concept_label')
                if concept:
                    print(f"\n🏷️  Related Concept: {concept}")
                
            else:
                # Single clause result
                print(f"\n📄 Result #{i}:")
                print("-"*70)
                
                framework = record.get('framework', 'N/A')
                code = record.get('code', 'N/A')
                text = record.get('text', '')
                
                emoji = "📘" if framework == "JCI" else "📗"
                print(f"{emoji} {framework} Clause: {code}")
                print(f"   Text: {text[:400]}...")
                
                concept = record.get('concept_name') or record.get('concept_label')
                if concept:
                    print(f"   Related Concept: {concept}")
        
        print("\n" + "="*70)
        
    except Exception as e:
        print(f"\n❌ Query execution error: {str(e)}")

# ============================================
# Main Query Pipeline
# ============================================

def process_query(question: str):
    """Complete retrieval pipeline with entity extraction."""
    
    print(f"\n{'='*70}")
    print(f"❓ Question: {question}")
    print(f"{'='*70}")
    
    # Step 1: Extract entities
    print("\n[Step 1/3] Extracting entities from question...")
    entities = extract_entities_from_question(question)
    
    print(f"\n📝 Extracted Entities:")
    print(f"   Topics: {entities.get('topics', [])}")
    print(f"   Frameworks: {entities.get('frameworks', [])}")
    print(f"   Query Type: {entities.get('query_type', 'unknown')}")
    
    # Step 2: Generate Cypher
    print(f"\n[Step 2/3] Generating optimized Cypher query...")
    cypher = generate_cypher_with_entities(question, entities)
    
    print(f"\n📝 Generated Cypher:")
    print("-"*70)
    print(cypher)
    print("-"*70)
    
    # Step 3: Execute and display
    print(f"\n[Step 3/3] Executing query and retrieving results...")
    execute_and_display_results(cypher)

# ============================================
# Interactive Chatbot
# ============================================

def run_chatbot():
    print("\n" + "="*70)
    print("  🏥 HOSPITAL AUDIT COMPLIANCE ASSISTANT")
    print("  Retrieval-Only Mode with Entity Extraction")
    print("="*70)
    print("\nCommands: 'exit', 'help', 'test'")
    print("="*70)
    
    while True:
        try:
            user_input = input("\n💬 You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            elif user_input.lower() == 'help':
                print("\n📖 Example Questions:")
                print("\n🔄 Comparisons:")
                print("  • Compare JCI and SHCC standards for hand hygiene")
                print("  • What are the differences between JCI and SHCC medication safety?")
                print("\n🔍 Searches:")
                print("  • What are the requirements for patient identification?")
                print("  • Show me infection control standards in JCI")
                print("  • Find all clauses about informed consent")
                continue
            
            elif user_input.lower() == 'test':
                # Run test queries
                test_queries = [
                    "Compare JCI and SHCC standards for hand hygiene",
                    "What are medication safety requirements?",
                    "Show me patient identification requirements in JCI"
                ]
                for query in test_queries:
                    process_query(query)
                    input("\nPress Enter for next test...")
                continue
            
            # Process query
            process_query(user_input)
            
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            import traceback
            traceback.print_exc()

# ============================================
# Main Execution
# ============================================

if __name__ == "__main__":
    run_chatbot()