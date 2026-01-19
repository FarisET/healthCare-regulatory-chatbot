"""
GraphRAG Audit Assistant - Clause-Centric with Concepts as Context Enhancement
Pipeline: Question → Cypher → Clauses + SIMILAR_TO → Optional Concepts → Response
"""
import os
from neo4j import GraphDatabase
from typing import List, Dict, Any
from types import SimpleNamespace
from openai import OpenAI as OpenAIClient
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()
# ============================================
# GraphRAG Audit Assistant Class
# ============================================

# Put this near the top of your file (after imports)
from types import SimpleNamespace
from openai import OpenAI as OpenAIClient

class OpenAIInvokeAdapter:
    """
    Adapter to make openai.OpenAI behave like an object with invoke(messages)
    where `messages` is a list of langchain_core.messages.SystemMessage/HumanMessage
    or simple objects with .content and .role semantics.
    Returns SimpleNamespace(content=str).
    """
    def __init__(self, api_key: str = None, base_url: str = None, model: str = "gpt-4o-mini"):
        # Create the underlying OpenAI client
        self.client = OpenAIClient(api_key=api_key)
        self.model = model
        # If you need to set base_url for a custom host, set it on environment or client config:
        if base_url:
            # The OpenAI client doesn't accept base_url param in constructor in all SDKs;
            # If needed, set OPENAI_API_BASE env var before creating client:
            import os
            os.environ["OPENAI_API_BASE"] = base_url

    def _convert_messages(self, messages):
        """
        Accepts list of langchain_core.messages.* or dicts with .content
        and returns a list of {"role": "...", "content": "..."} for the API.
        """
        converted = []
        for m in messages:
            # Langchain message classes often have .type or .role attributes:
            role = getattr(m, "role", None) or getattr(m, "type", None)
            if role is None:
                # try to infer from class name (SystemMessage/HumanMessage/AIMessage)
                role = m.__class__.__name__.replace("Message", "").lower()
                if role == "ai": role = "assistant"
            content = getattr(m, "content", None)
            if content is None and isinstance(m, dict):
                content = m.get("content")
                role = m.get("role", role)
            converted.append({"role": role, "content": content})
        return converted

    def invoke(self, messages):
        """
        messages: list of message objects (langchain SystemMessage/HumanMessage etc.)
        returns: SimpleNamespace(content=str) to match your current usage
        """
        chat_messages = self._convert_messages(messages)

        # call chat completion
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=chat_messages,
            # tune other params if you want: temperature=0.0, max_tokens=1000, etc.
        )

        # extract text -- depends on SDK response shape
        # For openai.OpenAI client: resp.choices[0].message.content
        try:
            content = resp.choices[0].message["content"]
        except Exception:
            # fallback for alternative response shapes
            content = getattr(resp.choices[0].message, "content", None) or str(resp)

        return SimpleNamespace(content=content)

class GraphRAGAuditAssistant:
    """
    Clause-centric GraphRAG for hospital audit compliance queries.
    Core Strategy:
    - Framework comparisons → Use SIMILAR_TO relationships between clauses
    - Topic searches → Start with clauses, optionally use concepts for filtering
    - Concepts are terminology hints, NOT ground truth
    """
    
    def __init__(self, neo4j_uri, neo4j_user, neo4j_password, llm_instance: ChatGoogleGenerativeAI):
        self.llm = llm_instance
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        
        self.schema = """
        Graph Schema (CLAUSE-CENTRIC):
        
        PRIMARY NODES:
        - Clause (GROUND TRUTH - regulatory requirements):
            Properties: {code: string, text: string, framework: string ('JCI' or 'SHCC')}
            Examples: 
              • JCI: "IPSG.3" (medication safety), "ACC.1" (patient access)
              • SHCC: "4.1.1" (strategic planning), "15.1" (infection control)
        
        SECONDARY NODES:
        - Concept (TERMINOLOGY - extracted medical terms for context enhancement):
            Properties: {name: string, label: string, description: string}
            Purpose: Help identify related clauses through terminology
            Note: These are NOT authoritative - use for filtering/context only
        
        RELATIONSHIPS:
        - (Clause)-[:SIMILAR_TO {score: float}]->(Clause) [PRIMARY for comparisons]
            Purpose: Semantic similarity between clauses (cross-framework)
            Score: 0.0 to 1.0 (use threshold 0.75+ for meaningful similarity)
            
        - (Clause)-[:MENTIONS]->(Concept) [SECONDARY - context only]
            Purpose: Link clauses to terminology (helps find related clauses)
        
        QUERY STRATEGY BY QUESTION TYPE:
        
        1. FRAMEWORK COMPARISON (e.g., "Compare JCI and SHCC for X"):
           PRIMARY: Use SIMILAR_TO relationships
           SECONDARY: Optionally filter by concepts
           Example:
           MATCH (jci:Clause {framework: 'JCI'})-[s:SIMILAR_TO]->(shcc:Clause {framework: 'SHCC'})
           WHERE s.score > 0.75
           OPTIONAL MATCH (jci)-[:MENTIONS]->(co:Concept)
           WHERE co.name CONTAINS 'hand-hygiene'
           RETURN jci, shcc, s.score
           
        2. FIND SIMILAR CLAUSES (e.g., "Find clauses similar to IPSG.3"):
           PRIMARY: SIMILAR_TO from specified clause
           Example:
           MATCH (source:Clause {code: 'IPSG.3'})-[s:SIMILAR_TO]->(similar:Clause)
           WHERE s.score > 0.75
           RETURN source, similar, s.score ORDER BY s.score DESC
           
        3. TOPIC SEARCH (e.g., "What are requirements for X"):
           PRIMARY: Search clause text directly
           SECONDARY: Use concepts to help filter
           Example:
           MATCH (cl:Clause)
           WHERE cl.text CONTAINS 'hand hygiene' OR cl.text CONTAINS 'hand washing'
           OPTIONAL MATCH (cl)-[:MENTIONS]->(co:Concept)
           WHERE co.name CONTAINS 'hand' OR co.name CONTAINS 'hygiene'
           RETURN cl, collect(co.name) as related_concepts
           
        4. FRAMEWORK-SPECIFIC QUERIES:
           Filter by framework property
           Example:
           MATCH (cl:Clause {framework: 'JCI'})
           WHERE cl.code STARTS WITH 'IPSG'
           
        IMPORTANT RULES:
        - ALWAYS prioritize Clause nodes and SIMILAR_TO relationships
        - Use Concepts OPTIONALLY to enhance filtering/context
        - For comparisons, MUST use SIMILAR_TO relationships
        - Score thresholds: >0.85 (very similar), >0.75 (similar), >0.65 (somewhat related)
        - Return actual clause text, not just concepts
        """
        
        print("✓ Connected to Neo4j")
        print("✓ Clause-centric GraphRAG initialized")
        print("✓ Audit Assistant ready")
    
    # ==========================================
    # Step 1: Enhanced NL to Cypher
    # ==========================================
    
    def nl_to_cypher(self, question: str) -> str:
        """Convert natural language to clause-centric Cypher query."""
        
        cypher_prompt = """You are an expert Neo4j Cypher generator for a hospital audit compliance system.

Database Schema:
{schema}

User Question: "{question}"

CRITICAL INSTRUCTIONS:
1. For FRAMEWORK COMPARISONS (comparing JCI vs SHCC):
   - MUST use SIMILAR_TO relationships between clauses
   - Start with: MATCH (jci:Clause {{framework: 'JCI'}})-[s:SIMILAR_TO]->(shcc:Clause {{framework: 'SHCC'}})
   - Use score threshold: WHERE s.score > 0.75
   - Concepts are OPTIONAL for additional filtering only
   
2. For SIMILARITY SEARCHES (find similar clauses):
   - Use: MATCH (source:Clause {{code: 'X'}})-[s:SIMILAR_TO]->(similar:Clause)
   - Order by: ORDER BY s.score DESC
   
3. For TOPIC SEARCHES (find requirements about X):
   - Primary: Search clause.text directly with CONTAINS or regex
   - Secondary: Optionally join concepts for additional context
   - Example: WHERE cl.text CONTAINS 'hand hygiene' OR cl.text =~ '(?i).*hand.*hygiene.*'
   
4. ALWAYS return:
   - Clause codes (cl.code)
   - Clause text (cl.text) 
   - Framework (cl.framework)
   - Similarity scores if applicable (s.score)
   - Concepts as supplementary info only: collect(DISTINCT co.name) as concepts
   
5. Use LIMIT 20 for large result sets

6. For text searches, use case-insensitive regex: =~ '(?i).*keyword.*'

Generate ONLY the Cypher query without explanation or markdown formatting.

Cypher Query:"""
        
        prompt = cypher_prompt.format(schema=self.schema, question=question)
        response = self.llm.invoke([HumanMessage(content=prompt)])
        
        cypher = response.content.strip()
        cypher = cypher.replace("```cypher", "").replace("```", "").strip()
        
        return cypher
    
    # ==========================================
    # Step 2: Execute & Retrieve
    # ==========================================
    
    def execute_cypher(self, cypher_query: str) -> List[Dict[str, Any]]:
        """Execute Cypher query and return results."""
        try:
            with self.driver.session() as session:
                result = session.run(cypher_query)
                records = [record.data() for record in result]
                return records
        except Exception as e:
            print(f"⚠️  Cypher execution error: {str(e)}")
            return []
    
    # ==========================================
    # Step 3: Build Context from Clauses
    # ==========================================
    
    def build_context_from_results(self, results: List[Dict], question: str) -> str:
        """
        Transform clause-centric results into structured context.
        Focus on clauses and their relationships.
        """
        if not results:
            return "No relevant clauses found in the knowledge graph."
        
        context_parts = []
        context_parts.append(f"Retrieved {len(results)} relevant audit clauses:\n")
        
        # Track what we've seen
        jci_clauses = []
        shcc_clauses = []
        similarities = []
        all_concepts = set()
        
        for i, record in enumerate(results, 1):
            record_text = []
            
            # Extract clause information (primary focus)
            for key, value in record.items():
                # Handle Clause nodes
                if isinstance(value, dict) and 'code' in value:
                    clause_info = {
                        'code': value.get('code', 'N/A'),
                        'text': value.get('text', ''),
                        'framework': value.get('framework', 'N/A')
                    }
                    
                    if clause_info['framework'] == 'JCI':
                        jci_clauses.append(clause_info)
                    elif clause_info['framework'] == 'SHCC':
                        shcc_clauses.append(clause_info)
                
                # Handle similarity scores
                elif 'score' in key.lower() and isinstance(value, (int, float)):
                    similarities.append(value)
                
                # Handle concepts (supplementary)
                elif 'concept' in key.lower() and value:
                    if isinstance(value, list):
                        all_concepts.update(value)
                    elif isinstance(value, str):
                        all_concepts.add(value)
        
        # Format JCI clauses
        if jci_clauses:
            context_parts.append("\n" + "="*70)
            context_parts.append("📘 JCI (Joint Commission International) Clauses:")
            context_parts.append("="*70)
            for idx, clause in enumerate(jci_clauses[:10], 1):  # Limit to 10
                context_parts.append(f"\n{idx}. Code: {clause['code']}")
                context_parts.append(f"   Text: {clause['text'][:400]}...")
        
        # Format SHCC clauses
        if shcc_clauses:
            context_parts.append("\n" + "="*70)
            context_parts.append("📗 SHCC (Sindh Healthcare Commission) Clauses:")
            context_parts.append("="*70)
            for idx, clause in enumerate(shcc_clauses[:10], 1):
                context_parts.append(f"\n{idx}. Code: {clause['code']}")
                context_parts.append(f"   Text: {clause['text'][:400]}...")
        
        # Add similarity information
        if similarities:
            context_parts.append("\n" + "="*70)
            context_parts.append("🔗 Similarity Analysis:")
            context_parts.append("="*70)
            avg_sim = sum(similarities) / len(similarities)
            max_sim = max(similarities)
            context_parts.append(f"   Average similarity score: {avg_sim:.3f}")
            context_parts.append(f"   Highest similarity score: {max_sim:.3f}")
            context_parts.append(f"   Total clause pairs compared: {len(similarities)}")
        
        # Add concepts as supplementary context (if present)
        if all_concepts:
            context_parts.append("\n" + "="*70)
            context_parts.append("🏷️  Related Terminology (for context):")
            context_parts.append("="*70)
            context_parts.append(f"   {', '.join(sorted(all_concepts)[:15])}")
        
        # Summary statistics
        context_parts.append("\n" + "="*70)
        context_parts.append("📊 Summary:")
        context_parts.append("="*70)
        context_parts.append(f"   JCI clauses: {len(jci_clauses)}")
        context_parts.append(f"   SHCC clauses: {len(shcc_clauses)}")
        if similarities:
            context_parts.append(f"   Clause pairs with similarity data: {len(similarities)}")
        
        return "\n".join(context_parts)
    
    # ==========================================
    # Step 4: Generate Response
    # ==========================================
    
    def generate_response(self, question: str, context: str, cypher_used: str) -> str:
        """Generate audit compliance response focused on clause requirements."""
        
        response_prompt = """You are an expert hospital audit compliance assistant specializing in JCI and SHCC standards.

Your role:
1. Answer questions about regulatory requirements using ACTUAL CLAUSE TEXT
2. Compare frameworks based on clause similarity and content
3. Explain requirements in clear, actionable terms
4. Reference specific clause codes for traceability

User Question:
{question}

Retrieved Clauses and Context:
{context}

Cypher Query Used (for your reference):
{cypher}

CRITICAL INSTRUCTIONS:
1. Base your answer PRIMARILY on the clause text provided in the context
2. When comparing frameworks:
   - Quote or paraphrase actual clause requirements
   - Highlight similarities AND differences
   - Reference specific clause codes (e.g., "JCI IPSG.5 requires...")
3. If concepts are mentioned in context, use them only to understand terminology
4. Structure your response:
   - Start with a direct answer
   - Provide framework-specific details with clause references
   - Highlight key similarities/differences if comparing
   - End with actionable insights
5. If context is insufficient, clearly state what's missing
6. Be precise: Use similarity scores to indicate alignment strength
   - >0.90: "Nearly identical requirements"
   - 0.80-0.90: "Highly similar with minor variations"  
   - 0.75-0.80: "Similar core requirements, some differences"
   - <0.75: "Related but distinct requirements"

Response (150-400 words):"""
        
        prompt = response_prompt.format(
            question=question,
            context=context,
            cypher=cypher_used
        )
        
        messages = [
            SystemMessage(content="You are a hospital audit compliance expert focused on precise regulatory interpretation."),
            HumanMessage(content=prompt)
        ]
        
        response = self.llm.invoke(messages)
        return response.content.strip()
    
    # ==========================================
    # Main Query Pipeline
    # ==========================================
    
    def query(self, question: str, show_intermediate: bool = True) -> str:
        """
        Complete clause-centric GraphRAG pipeline.
        """
        
        if show_intermediate:
            print(f"\n{'='*70}")
            print(f"❓ Question: {question}")
            print(f"{'='*70}")
        
        # Step 1: Generate Cypher
        if show_intermediate:
            print("\n[Step 1/4] Generating clause-centric Cypher query...")
        
        cypher = self.nl_to_cypher(question)
        
        if show_intermediate:
            print(f"\n📝 Generated Cypher:")
            print(f"   {cypher}")
        
        # Step 2: Execute
        if show_intermediate:
            print(f"\n[Step 2/4] Executing against Neo4j...")
        
        results = self.execute_cypher(cypher)
        
        if show_intermediate:
            print(f"   ✓ Retrieved {len(results)} records")
        
        # Step 3: Build context
        if show_intermediate:
            print(f"\n[Step 3/4] Building context from clauses...")
        
        context = self.build_context_from_results(results, question)
        
        if show_intermediate:
            print(f"\n📚 Context Preview:")
            preview = context[:300] + "..." if len(context) > 300 else context
            print(f"   {preview}")
        
        # Step 4: Generate response
        if show_intermediate:
            print(f"\n[Step 4/4] Generating response from clause requirements...")
        
        response = self.generate_response(question, context, cypher)
        
        if show_intermediate:
            print(f"\n{'='*70}")
            print(f"🤖 ASSISTANT RESPONSE:")
            print(f"{'='*70}")
        
        print(f"\n{response}\n")
        
        return response
    
    def close(self):
        """Close Neo4j connection."""
        self.driver.close()
        print("\n✓ Neo4j connection closed")


# ============================================
# Interactive Chatbot
# ============================================

def run_interactive_chatbot(assistant: GraphRAGAuditAssistant):
    """Run clause-centric audit compliance chatbot."""
    
    print("\n" + "="*70)
    print("  🏥 HOSPITAL AUDIT COMPLIANCE ASSISTANT")
    print("  Clause-Centric GraphRAG for JCI & SHCC Standards")
    print("="*70)
    print("\nWhat I can help you with:")
    print("  • Compare JCI and SHCC requirements (uses SIMILAR_TO)")
    print("  • Find similar clauses across frameworks")
    print("  • Search for specific audit requirements by topic")
    print("  • Explain regulatory clause requirements")
    print("  • Analyze framework alignment and gaps")
    print("\nCommands:")
    print("  'exit' or 'quit' - Close chatbot")
    print("  'help' - Show example questions")
    print("  'verbose on/off' - Toggle detailed output")
    print("="*70)
    
    show_intermediate = True
    
    while True:
        try:
            user_input = input("\n💬 You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("\n👋 Thank you for using the Audit Compliance Assistant!")
                break
            
            elif user_input.lower() == 'help':
                print("\n📖 Example Questions:")
                print("\n🔄 Framework Comparisons:")
                print("  • Compare JCI and SHCC standards for hand hygiene")
                print("  • What are the differences between JCI and SHCC medication safety requirements?")
                print("  • Compare infection control standards across frameworks")
                print("\n🔍 Similarity Searches:")
                print("  • Find SHCC clauses similar to JCI IPSG.3")
                print("  • Show me clauses related to SHCC 15.1")
                print("\n📋 Topic Searches:")
                print("  • What are the requirements for patient identification?")
                print("  • Find all JCI clauses about informed consent")
                print("  • Show SHCC requirements for fire safety")
                continue
            
            elif user_input.lower() == 'verbose on':
                show_intermediate = True
                print("✓ Verbose mode enabled")
                continue
            
            elif user_input.lower() == 'verbose off':
                show_intermediate = False
                print("✓ Verbose mode disabled")
                continue
            
            # Process question
            assistant.query(user_input, show_intermediate=show_intermediate)
            
        except KeyboardInterrupt:
            print("\n\n👋 Chatbot interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            print("Please try rephrasing or type 'help' for examples.")


# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    
    NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
    NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "your_password")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "your_api_key")
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")


    print("🚀 Initializing Clause-Centric Audit Assistant...")
    
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.3,
            google_api_key=GOOGLE_API_KEY
        )
        # llm = OpenAIInvokeAdapter(
        #     api_key=DEEPSEEK_API_KEY,
        #     base_url="https://api.deepseek.com",   # only if you really need custom base_url
        #     model="deepseek-chat"  # choose model you have access to
        #     )                   
    


        
        assistant = GraphRAGAuditAssistant(
            NEO4J_URI,
            NEO4J_USER,
            NEO4J_PASSWORD,
            llm
        )
        
        run_interactive_chatbot(assistant)
        
    except Exception as e:
        print(f"\n❌ Fatal error: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        if 'assistant' in locals():
            assistant.close()