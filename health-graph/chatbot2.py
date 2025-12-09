import os
import json
from neo4j import GraphDatabase
from typing import List, Dict, Any

# --- LangChain Imports ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

# ============================================
# GraphRAG Audit Assistant Class
# ============================================

class GraphRAGAuditAssistant:
    """
    Advanced GraphRAG pipeline for Provenance-Backed Audit Compliance.
    Handles Static (Clauses) and Dynamic (Findings) Knowledge retrieval.
    """
    
    def __init__(self, neo4j_uri, neo4j_user, neo4j_password, llm_instance):
        self.llm = llm_instance
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        
        # 🧠 UPDATED SCHEMA: Includes Findings, Instances, and Audit Context
        self.schema = """
        Graph Schema:
        
        [Static Regulatory Layer]
        - (c:Clause) {code, text, framework} 
          Represents the standard (e.g., JCI IPSG.3).
        - (co:Concept) {name, label, description}
          Represents medical/audit terms extracted from text.
        
        [Dynamic Audit Layer]
        - (f:AuditFinding) {id, text, grade, status}
          Represents the actual violation (e.g., "NC #01", grade: "major").
        - (ci:ClauseInstance) {id}
          Represents the specific check of a clause during an audit.
        - (a:Audit) {id, date}
          The audit event container.
        - (dept:Department) {name}
        - (auditor:Auditor) {name}
        
        [Relationships]
        - (ci)-[:HAS_FINDING]->(f)        : Links a check to its negative result.
        - (ci)-[:INSTANCE_OF]->(c)        : Links the check back to the Regulatory Clause.
        - (ci)-[:PART_OF]->(a)            : Links the check to the Audit event.
        - (a)-[:AUDIT_OF]->(dept)         : Links audit to department.
        - (a)-[:AUDITED_BY]->(auditor)    : Links audit to the auditor.
        - (c)-[:MENTIONS]->(co)           : Clause mentions concept.
        - (c)-[:SIMILAR_TO]->(c)          : Similarity between JCI and SHCC clauses.
        
        [Common Retrieval Paths]
        1. Find violations of a specific standard:
           MATCH (f:AuditFinding)<-[:HAS_FINDING]-(ci:ClauseInstance)-[:INSTANCE_OF]->(c:Clause) WHERE c.code = '...'
        
        2. Find all findings in a specific department:
           MATCH (dept:Department)<-[:AUDIT_OF]-(a:Audit)<-[:PART_OF]-(ci)-[:HAS_FINDING]->(f) WHERE dept.name CONTAINS '...'
        
        3. Find violations regarding a concept (e.g., "Medication"):
           MATCH (c:Clause)-[:MENTIONS]->(co:Concept) WHERE co.name CONTAINS 'medication'
           MATCH (c)<-[:INSTANCE_OF]-(ci)-[:HAS_FINDING]->(f)
           RETURN c.code, f.text, f.grade
        """
        
        print("✓ Connected to Neo4j")
        print("✓ GraphRAG Schema Loaded (Static + Dynamic Layers)")

    # ==========================================
    # Step 1: NL to Cypher Generation
    # ==========================================
    
    def nl_to_cypher(self, question: str) -> str:
        """Generates Cypher with emphasis on traversing the Instance/Finding bridge."""
        
        cypher_prompt = f"""You are an expert Neo4j Cypher generator for Hospital Audits.
        
        Schema:
        {self.schema}
        
        Goal: Translate the user question into a precise Cypher query.
        
        Rules:
        1. ALWAYS return variables that give context: Clause Code, Finding Text, Grade, Department.
        2. Use Case-Insensitive matching (toLower) for text search.
        3. If asking about "Violations" or "Non-Conformities", you MUST traverse (:AuditFinding).
        4. If asking about "Standards" or "Rules", verify if they want actual findings or just the static clause text.
        5. LIMIT results to 20.
        
        User Question: "{question}"
        
        Output: ONLY the Cypher query. No markdown.
        """
        
        response = self.llm.invoke([HumanMessage(content=cypher_prompt)])
        # Cleanup
        cypher = response.content.replace("```cypher", "").replace("```", "").strip()
        return cypher

    # ==========================================
    # Step 2: Execute Cypher
    # ==========================================
    
    def execute_cypher(self, cypher_query: str) -> List[Dict[str, Any]]:
        try:
            with self.driver.session() as session:
                result = session.run(cypher_query)
                return [record.data() for record in result]
        except Exception as e:
            return [{"error": str(e)}]

    # ==========================================
    # Step 3: Build Context (The "Provenance" Layer)
    # ==========================================
    
    def build_context_from_results(self, results: List[Dict]) -> str:
        """
        Organizes raw graph data into a narrative for the LLM.
        Distinguishes between RULES (Clauses) and FACTS (Findings).
        """
        if not results:
            return "No data found in the graph."
        if "error" in results[0]:
            return f"Query Error: {results[0]['error']}"

        context_buffer = ["### RETRIEVED KNOWLEDGE GRAPH DATA ###\n"]
        
        for i, row in enumerate(results):
            context_buffer.append(f"--- Record {i+1} ---")
            
            # 1. Handle Findings (The "What Happened")
            # Look for common variable names generated by LLM for findings
            finding_node = None
            for key, val in row.items():
                if isinstance(val, dict) and ('grade' in val or 'nc_id' in val): # Detect finding node
                     finding_node = val
                # Also check flattened returns
                if key == 'f.text' or key == 'finding_text':
                     context_buffer.append(f"🚨 VIOLATION: {val}")
                if key == 'f.grade' or key == 'grade':
                     context_buffer.append(f"   SEVERITY: {val}")
                if key == 'f.id' or key == 'nc_id':
                     context_buffer.append(f"   ID: {val}")

            if finding_node:
                context_buffer.append(f"🚨 VIOLATION [{finding_node.get('id','?')}]: {finding_node.get('text','')} (Grade: {finding_node.get('grade','')})")

            # 2. Handle Clauses (The "Rule")
            clause_code = row.get('c.code') or row.get('code')
            clause_text = row.get('c.text') or row.get('text')
            
            # Sometimes LLM returns full node objects
            for key, val in row.items():
                if isinstance(val, dict) and 'framework' in val:
                    clause_code = val.get('code')
                    clause_text = val.get('text')

            if clause_code:
                context_buffer.append(f"📜 STANDARD VIOLATED: {clause_code}")
                if clause_text:
                    context_buffer.append(f"   RULE TEXT: {clause_text[:150]}...")

            # 3. Handle Context (Dept/Auditor)
            dept = row.get('dept.name') or row.get('department')
            if dept:
                context_buffer.append(f"📍 LOCATION: {dept}")

        return "\n".join(context_buffer)

    # ==========================================
    # Step 4: Generate Provenance-Backed Response
    # ==========================================
    
    def generate_response(self, question: str, context: str) -> str:
        """
        Forces the LLM to cite specific NC IDs and Clauses.
        """
        final_prompt = f"""You are a Hospital Audit Compliance Officer.
        
        User Question: "{question}"
        
        Graph Data:
        {context}
        
        Directives:
        1. PROVENANCE IS MANDATORY: You must cite the 'ID' (e.g., NC #01) for every claim you make about a violation.
        2. If discussing standards, cite the Clause Code (e.g., IPSG.3).
        3. If the user asks about a specific department (e.g. ER), summarize the severity of findings there.
        4. If findings map to multiple clauses (e.g. JCI and SHCC), mention both to show the cross-framework impact.
        5. Be professional and objective.
        
        Answer:
        """
        
        msg = [HumanMessage(content=final_prompt)]
        response = self.llm.invoke(msg)
        return response.content

    # ==========================================
    # Main Query Method
    # ==========================================
    def query(self, question: str):
        print(f"\nUser: {question}")
        print("Thinking (Generating Cypher)...")
        
        cypher = self.nl_to_cypher(question)
        print(f"🔍 Cypher: {cypher}")
        
        results = self.execute_cypher(cypher)
        print(f"📊 Retrieved {len(results)} records.")
        
        context = self.build_context_from_results(results)
        
        print("💡 Synthesizing Answer...")
        answer = self.generate_response(question, context)
        
        print(f"\nAssistant:\n{answer}")
        return answer

# ============================================
# Execution
# ============================================
if __name__ == "__main__":
    NEO4J_URI = os.environ.get("NEO4J_URI")
    NEO4J_USER = os.environ.get("NEO4J_USER")
    NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

    # Init
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GOOGLE_API_KEY)
    bot = GraphRAGAuditAssistant(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, llm)
    
    print("\n--- TEST SCENARIOS ---\n")
    
    # Test 1: Provenance Check (Specific Finding)
    bot.query("What violations were found regarding High Alert Medications?")
    
    # Test 2: Cross-Framework Impact (The NC #16 scenario)
    bot.query("Were there any issues with patient assessment that violated both JCI and SHCC?")
    
    # Test 3: Departmental Summary
    bot.query("Give me a summary of major risks identified in the Adult Emergency Department.")