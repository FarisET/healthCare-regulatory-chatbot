"""
Neo4j Natural Language to Cypher - Implementation using LangChain's ChatGoogleGenerativeAI
Now with an Interactive Chatbot Loop (No History)
"""
import os
from neo4j import GraphDatabase

# --- LangChain Imports ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate

# ----------------------------------------------------------------------
# --- 1. LLM Initialization and Test ---

def test_langchain_llm(api_key: str):
    """Initial test of the LLM using LangChain's ChatGoogleGenerativeAI."""
    print("="*70)
    print("🚀 [TEST] Testing LangChain's ChatGoogleGenerativeAI (gemini-2.5-flash)...")
    try:
        # 1. Initialize the model using LangChain
        test_llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.7,
            google_api_key=api_key 
        )
        
        # 2. Simple invocation
        messages = [
            HumanMessage(content="What is the Cypher keyword to count nodes?")
        ]
        response = test_llm.invoke(messages)

        print("\n✅ Test Successful!")
        print(f"Model: {test_llm.model}")
        print(f"Question: {messages[0].content}")
        print(f"Answer: {response.content.strip()}")
        print("="*70)
        return True
    except Exception as e:
        # Catching the specific error from the LangChain client (e.g., API key issue)
        print(f"\n❌ Test Failed: Could not initialize or invoke LLM. Check GOOGLE_API_KEY. Error: {str(e)}")
        print("="*70)
        return False

# ----------------------------------------------------------------------
# --- 2. NLToCypherQuery Class ---

class NLToCypherQuery:
    """Simple NL to Cypher converter using LangChain's ChatGoogleGenerativeAI."""
    
    def __init__(self, neo4j_uri, neo4j_user, neo4j_password, llm_instance: ChatGoogleGenerativeAI):
        
        self.llm = llm_instance
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        
        self.schema = """
        Nodes:
        - Clause: {code: string, text: string, framework: string, full_text: string}
        - Concept: {name: string}
        
        Relationships:
        - (Clause)-[:SIMILAR_TO {score: float}]->(Clause)
        - (Clause)-[:MENTIONS]->(Concept)
        
        Examples:
        - JCI clauses have codes like: IPSG.3, ACC.1, PFR.5.1
        - SHCC clauses have codes like: 4.1.1, 4.1.2, 5.1.8
        - Frameworks are: 'JCI' or 'SHCC'
        """
        
        print("✓ Connected to Neo4j")
        print("✓ LangChain LLM ready")
    
    def nl_to_cypher(self, question):
        """Convert natural language to Cypher query using the LangChain LLM."""
        
        cypher_generation_template = """You are an expert Neo4j Cypher query generator.

        Database Schema:
        {schema}

        Convert this question to a Cypher query:
        "{question}"

        Rules:
        1. Return ONLY the Cypher query, no explanation or introductory text.
        2. Use proper Cypher syntax.
        3. Use LIMIT when appropriate.
        4. Match exact property names from schema.

        Cypher Query:"""
        
        prompt_content = cypher_generation_template.format(schema=self.schema, question=question)

        # Invoke the LangChain LLM
        response = self.llm.invoke([HumanMessage(content=prompt_content)])
        
        # Extract and clean the query from the response text
        cypher = response.content.strip()
        cypher = cypher.replace("```cypher", "").replace("```", "").strip()
        
        return cypher
    
    def execute_query(self, cypher_query):
        """Execute Cypher query and return results."""
        with self.driver.session() as session:
            result = session.run(cypher_query)
            return [record.data() for record in result]
    
    def query(self, question):
        """Main method: Ask question in natural language."""
        
        print(f"\n{'='*70}")
        print(f"Question: {question}")
        print(f"{'='*70}")
        
        # Step 1: Generate Cypher
        print("\n[1] Generating Cypher query...")
        cypher = self.nl_to_cypher(question)
        print(f"\n[Generated Cypher]")
        print(f"{cypher}")
        
        # Step 2: Execute query
       # print(f"\n[2] Executing query...")
      #  try:
            # For a real run, UNCOMMENT the next line:
            # results = self.execute_query(cypher) 
            
            # --- Placeholder result for a non-live execution example ---
            # if "count" in question.lower():
            #     results = [{'count': 123}]
            # elif "ipsg.3" in question.lower():
            #     results = [{'c.name': 'Patient Safety'}, {'c.name': 'Quality'}]
            # else:
            #      results = [{'code1': 'SHCC.4.1.1', 'code2': 'JCI.IPSG.3', 'score': 0.88}]
            # -----------------------------------------------------------
            
           # print(f"\n[3] Results (Simulated {len(results)} records):")
           # print("-" * 70)
            
            # if not results:
            #     print("No results found.")
            # else:
            #     for i, record in enumerate(results[:20], 1):
            #         print(f"\n{i}. {record}")
            
            # return results
            
        # except Exception as e:
        #     print(f"\n❌ Execution Error (Likely No Live Neo4j Connection): {str(e)}")
        #     return None
    
    def close(self):
        """Close Neo4j connection."""
        self.driver.close()
        print("\n✓ Neo4j connection closed")


# ============================================
# MAIN EXECUTION: CHATBOT LOOP
# ============================================

if __name__ == "__main__":
    
    # ===== CONFIGURATION (Use your actual details) =====
    NEO4J_URI = os.environ.get("NEO4J_URI")
    NEO4J_USER = os.environ.get("NEO4J_USER")
    NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
    # Use GEMINI_API_KEY
    
    # --- Check and Test LLM ---
    # if GOOGLE_API_KEY == "your_gemini_api_key":
    #     print("\n🛑 WARNING: Please replace 'your_gemini_api_key' with your actual API key.")
    
    # if not test_langchain_llm(GOOGLE_API_KEY):
    #     exit()

    # --- Initialize LLM for the main application ---
    main_llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        temperature=0, # Set to 0 for deterministic code generation
        google_api_key=GOOGLE_API_KEY 
    )

    # --- Initialize NLToCypherQuery ---
    print("\n🚀 Initializing NL to Cypher System for Chatbot...")
    
    try:
        nl_query = NLToCypherQuery(
            NEO4J_URI,
            NEO4J_USER,
            NEO4J_PASSWORD,
            main_llm
        )
        
        # ===== INTERACTIVE CHATBOT LOOP =====
        print("\n" + "#"*70)
        print("   CYPHER CHATBOT ACTIVE - Ask your graph questions!")
        print("   Type 'exit' or 'quit' to close.")
        print("#"*70)
        
        while True:
            # Get user input
            user_question = input("\n[USER]: ")
            
            # Check for exit commands
            if user_question.lower() in ['exit', 'quit', 'q']:
                print("\nGoodbye! Closing chatbot.")
                break
            
            # Process the query
            if user_question.strip():
                nl_query.query(user_question)
                
    except Exception as e:
        print(f"\n❌ A fatal error occurred during initialization or runtime: {str(e)}")
    
    finally:
        # Ensure the Neo4j connection is closed on exit
        if 'nl_query' in locals() and hasattr(nl_query, 'close'):
            nl_query.close()