import os
from typing import List, Optional

# LangChain & Neo4j - FIXED IMPORTS
from langchain_community.graphs import Neo4jGraph
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain.agents import AgentExecutor, create_react_agent

from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURATION ---
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "password")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

# Initialize Graph & LLM
graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USER, password=NEO4J_PASSWORD)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",  # Use 2.0 for better performance
    temperature=0, 
    google_api_key=GOOGLE_API_KEY
)

print("="*70)
print("📊 Graph Schema:")
print("="*70)
print(graph.schema)
print("="*70)

# ==========================================
# 🛠️ TOOL 1: Multi-Framework Comparison
# ==========================================
@tool
def compare_frameworks(topic: str) -> str:
    """
    Compare JCI and SHCC standards on a specific topic (e.g., 'hand hygiene', 'medication safety').
    Uses Concept nodes and SIMILAR_TO relationships for cross-framework mapping.
    
    Args:
        topic: The audit topic to compare (e.g., 'hand hygiene', 'patient identification')
    
    Returns:
        Comparison of JCI and SHCC clauses with similarity scores
    """
    
    cypher = """
    // Find Concept matching topic
    MATCH (co:Concept)
    WHERE toLower(co.name) CONTAINS toLower($topic)
       OR toLower(co.label) CONTAINS toLower($topic)
    
    // Find JCI clauses mentioning this concept
    MATCH (jci:Clause {framework: 'JCI'})-[:MENTIONS]->(co)
    
    // Find SHCC clauses mentioning this concept
    MATCH (shcc:Clause {framework: 'SHCC'})-[:MENTIONS]->(co)
    
    // Check for similarity relationships
    OPTIONAL MATCH (jci)-[s:SIMILAR_TO]->(shcc)
    WHERE s.score > 0.75
    
    RETURN 
        co.name as concept,
        jci.code as jci_code,
        substring(jci.text, 0, 200) as jci_text,
        shcc.code as shcc_code,
        substring(shcc.text, 0, 200) as shcc_text,
        s.score as similarity_score
    ORDER BY s.score DESC
    LIMIT 10
    """
    
    try:
        results = graph.query(cypher, {"topic": topic})
        
        if not results:
            return f"No comparison data found for topic: '{topic}'. Try terms like 'hand hygiene', 'medication', 'patient identification'."
        
        # Format results
        formatted = []
        for r in results:
            formatted.append(f"""
Concept: {r['concept']}
JCI: {r['jci_code']} - {r['jci_text']}...
SHCC: {r['shcc_code']} - {r['shcc_text']}...
Similarity: {r['similarity_score']:.3f if r['similarity_score'] else 'N/A'}
""")
        
        return "\n---\n".join(formatted)
        
    except Exception as e:
        return f"Error querying graph: {str(e)}"

# ==========================================
# 🛠️ TOOL 2: Audit Gap Analyzer
# ==========================================
@tool
def analyze_audit_gaps(keyword: str) -> str:
    """
    Find past audit findings related to a keyword to identify recurring compliance gaps.
    Useful for root cause analysis and tracking remediation history.
    
    Args:
        keyword: Topic or clause code (e.g., 'medication', 'fall risk', 'IPSG.3')
    
    Returns:
        Past audit findings with dates, departments, and remediation status
    """
    
    cypher = """
    MATCH (f:AuditFinding)
    WHERE toLower(f.text) CONTAINS toLower($keyword)
    
    // Get audit context
    MATCH (ci:ClauseInstance)-[:HAS_FINDING]->(f)
    MATCH (ci)-[:PART_OF]->(audit:Audit)
    MATCH (audit)-[:AUDITED_BY]->(auditor:Auditor)
    MATCH (audit)-[:AUDIT_OF]->(dept:Department)
    MATCH (ci)-[:INSTANCE_OF]->(clause:Clause)
    
    RETURN 
        audit.date as audit_date,
        dept.name as department,
        clause.code as standard_code,
        clause.framework as framework,
        f.text as finding_text,
        f.grade as severity,
        f.status as status,
        auditor.name as auditor
    ORDER BY audit.date DESC
    LIMIT 10
    """
    
    try:
        results = graph.query(cypher, {"keyword": keyword})
        
        if not results:
            return f"No audit findings found for keyword: '{keyword}'"
        
        formatted = []
        for r in results:
            formatted.append(f"""
Date: {r['audit_date']}
Department: {r['department']}
Standard: {r['framework']} {r['standard_code']}
Finding: {r['finding_text']}
Severity: {r['severity']}
Status: {r['status']}
Auditor: {r['auditor']}
""")
        
        return "\n---\n".join(formatted)
        
    except Exception as e:
        return f"Error querying audit history: {str(e)}"

# ==========================================
# 🛠️ TOOL 3: Audit Trail Retriever
# ==========================================
@tool
def retrieve_audit_trail(department: str) -> str:
    """
    Retrieve complete audit trail for a specific department.
    Useful for compliance evidence and external audit preparation.
    
    Args:
        department: Department name (e.g., 'Emergency Department', 'ICU')
    
    Returns:
        Chronological list of all audit findings for the department
    """
    
    cypher = """
    MATCH (dept:Department)<-[:AUDIT_OF]-(a:Audit)
    WHERE toLower(dept.name) CONTAINS toLower($dept)
    
    MATCH (a)<-[:PART_OF]-(ci:ClauseInstance)-[:HAS_FINDING]->(f:AuditFinding)
    MATCH (ci)-[:INSTANCE_OF]->(c:Clause)
    
    RETURN 
        a.id as audit_id,
        a.date as audit_date,
        c.framework as framework,
        c.code as standard_code,
        f.text as finding,
        f.status as status,
        f.grade as severity
    ORDER BY a.date DESC
    LIMIT 20
    """
    
    try:
        results = graph.query(cypher, {"dept": department})
        
        if not results:
            return f"No audit trail found for department: '{department}'"
        
        formatted = [f"Audit Trail for: {department}\n{'='*60}"]
        for r in results:
            formatted.append(f"""
Audit: {r['audit_id']} on {r['audit_date']}
Standard: {r['framework']} {r['standard_code']}
Finding: {r['finding']}
Severity: {r['severity']} | Status: {r['status']}
""")
        
        return "\n".join(formatted)
        
    except Exception as e:
        return f"Error retrieving audit trail: {str(e)}"

# ==========================================
# 🤖 AGENT CONSTRUCTION
# ==========================================

tools = [compare_frameworks, analyze_audit_gaps, retrieve_audit_trail]

# ReAct Prompt Template
prompt_template = """You are an expert hospital quality management and audit compliance assistant.

You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Guidelines:
- For framework comparisons, use compare_frameworks
- For finding recurring issues or root cause analysis, use analyze_audit_gaps
- For getting complete audit evidence, use retrieve_audit_trail
- Be specific with your Action Input - extract key terms from the question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

prompt = PromptTemplate.from_template(prompt_template)

# Create agent
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=True, 
    handle_parsing_errors=True,
    max_iterations=5
)

# ==========================================
# 📊 SIMPLE EVALUATION (WITHOUT RAGAS)
# ==========================================

def evaluate_agent_response(question: str, expected_info: str = None):
    """
    Run agent and display results with simple evaluation.
    """
    print("\n" + "="*70)
    print(f"🧪 Testing: {question}")
    print("="*70)
    
    try:
        response = agent_executor.invoke({"input": question})
        answer = response["output"]
        
        print("\n✅ Agent Response:")
        print("-"*70)
        print(answer)
        print("-"*70)
        
        if expected_info:
            print(f"\n📋 Expected Info: {expected_info}")
            # Simple check if key terms are present
            key_terms = expected_info.lower().split()
            found = sum(1 for term in key_terms if term in answer.lower())
            coverage = (found / len(key_terms)) * 100
            print(f"📊 Coverage: {coverage:.1f}% of expected terms found")
        
        return answer
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        return None

# ==========================================
# INTERACTIVE CHATBOT
# ==========================================

def run_interactive():
    print("\n" + "="*70)
    print("  🏥 HOSPITAL AUDIT AGENT")
    print("  ReAct Agent with Multi-Tool Access")
    print("="*70)
    print("\nCommands: 'exit', 'test', 'help'")
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
                print("\n🔄 Framework Comparisons:")
                print("  • How do JCI and SHCC compare on hand hygiene?")
                print("  • Compare medication safety standards")
                print("\n🔍 Audit Gap Analysis:")
                print("  • Have we had recurring findings about medication?")
                print("  • Show me past issues with patient identification")
                print("\n📋 Audit Trail Retrieval:")
                print("  • Get audit trail for Emergency Department")
                print("  • Show all findings for ICU")
                continue
            
            elif user_input.lower() == 'test':
                run_tests()
                continue
            
            # Process question
            evaluate_agent_response(user_input)
            
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")

# ==========================================
# TEST SUITE
# ==========================================

def run_tests():
    print("\n🧪 Running Test Suite...")
    
    tests = [
        {
            "question": "How do JCI and SHCC compare regarding hand hygiene?",
            "expected": "JCI IPSG.5 and SHCC infection control clauses emphasize hand hygiene with high similarity"
        },
        {
            "question": "Have we had recurring findings about medication?",
            "expected": "Past findings related to medication in Emergency Department"
        },
        {
            "question": "Prepare audit evidence for Emergency Department",
            "expected": "Audit findings including IPSG.3, IPSG.6.1, ACC.2"
        }
    ]
    
    for i, test in enumerate(tests, 1):
        print(f"\n{'='*70}")
        print(f"Test {i}/{len(tests)}")
        evaluate_agent_response(test["question"], test["expected"])
        
        if i < len(tests):
            input("\nPress Enter for next test...")

# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        run_tests()
    else:
        run_interactive()