import re
import os
from neo4j import GraphDatabase
import datetime
import audit_findings
from dotenv import load_dotenv
load_dotenv()

# --- Configuration ---
NEO4J_URI = os.environ.get("NEO4J_URI")
NEO4J_USER = os.environ.get("NEO4J_USER")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")

class AuditGraphBuilder:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def clean_clause_codes(self, raw_code_str):
        """
        Parses strings like "8.3.9 SHCC, COP.3.3 JCI" into list ["8.3.9", "COP.3.3"]
        so they match the existing Clause nodes in DB.
        """
        # Split by comma if multiple clauses exist
        raw_codes = raw_code_str.split(',')
        cleaned_codes = []
        
        for rc in raw_codes:
            # Remove brackets, framework names, and whitespace
            # Regex looks for the code pattern: numbers/dots/letters
            # This regex allows "AOP.1.1" or "6.3.5" but ignores "(JCI)"
            match = re.search(r'([A-Z]+\.[\d\.]+|[\d\.]+\d)', rc.strip())
            if match:
                cleaned_codes.append(match.group(0))
            else:
                # Fallback: simple strip if regex fails
                cleaned_codes.append(rc.strip().split()[0])
                
        return cleaned_codes

    def create_audit_context(self, auditor_name, department_name, audit_date):
        """Creates the container nodes for the Audit event."""
        with self.driver.session() as session:
            session.run("""
                MERGE (a:Auditor {name: $auditor})
                MERGE (d:Department {name: $dept})
                MERGE (audit:Audit {id: $audit_id})
                SET audit.date = $date
                
                MERGE (audit)-[:AUDITED_BY]->(a)
                MERGE (audit)-[:AUDIT_OF]->(d)
            """, auditor=auditor_name, dept=department_name, 
                 audit_id=f"AUDIT_{department_name}_{audit_date}", date=audit_date)
            print(f"Context created: {auditor_name} audited {department_name}")

    def ingest_findings(self, findings, department_name, audit_date):
        audit_id = f"AUDIT_{department_name}_{audit_date}"
        
        with self.driver.session() as session:
            for item in findings:
                # 1. Clean the codes
                target_codes = self.clean_clause_codes(item['clause_code'])
                
                # 2. Create the Finding Node (One per finding, regardless of how many clauses)
                # We add a 'status' property defaulting to 'Open'
                session.run("""
                    MERGE (f:AuditFinding {id: $nc_id})
                    SET f.text = $text,
                        f.grade = $grade,
                        f.status = 'Open' 
                """, nc_id=item['nc_id'], text=item['finding_text'], grade=item['grade'])
                
                # 3. Create Clause Instances and Link Everything
                for code in target_codes:
                    # Logic:
                    # - Find the Static Clause (c) matching the code
                    # - Create a dynamic ClauseInstance (ci) linked to the Audit
                    # - Link Instance -> Clause
                    # - Link Instance -> Finding
                    
                    result = session.run("""
                        MATCH (audit:Audit {id: $audit_id})
                        MATCH (f:AuditFinding {id: $nc_id})
                        
                        // Try to find the clause. Optional match prevents crashing if clause missing.
                        MATCH (c:Clause) WHERE c.code = $code
                        
                        // Create the Instance
                        MERGE (ci:ClauseInstance {id: $instance_id})
                        
                        // Link Instance to Static Clause (The Rule)
                        MERGE (ci)-[:INSTANCE_OF]->(c)
                        
                        // Link Instance to Audit Event (The Context)
                        MERGE (ci)-[:PART_OF]->(audit)
                        
                        // Link Instance to the Violation (The Finding)
                        MERGE (ci)-[:HAS_FINDING]->(f)
                        
                        RETURN c.code as found_code
                    """, 
                    audit_id=audit_id, 
                    nc_id=item['nc_id'], 
                    code=code,
                    instance_id=f"{audit_id}_{code}" # Unique ID for this specific check
                    )
                    
                    if result.peek() is None:
                        print(f"⚠️ Warning: Finding {item['nc_id']} references Clause '{code}' which was not found in the DB.")
                    else:
                        print(f"✅ Linked {item['nc_id']} to Clause {code}")

if __name__ == "__main__":
    # Initialize
    kg = AuditGraphBuilder(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    
    # Run Ingestion
    try:
        # 1. Setup Context
        kg.create_audit_context("John Doe", "Adult Emergency Department", "2023-10-25")
        
        # 2. Ingest Findings
        kg.ingest_findings(audit_findings.AUDIT_FINDINGS, "Adult Emergency Department", "2023-10-25")
        
        print("\n=== Audit Ingestion Complete ===")
        print("Run this query to see your multi-clause overlap (e.g. NC #16):")
        print("""
        MATCH (f:AuditFinding {id: 'NC #16'})
        MATCH (ci:ClauseInstance)-[:HAS_FINDING]->(f)
        MATCH (ci)-[:INSTANCE_OF]->(c:Clause)
        RETURN f.text, c.code, c.framework
        """)
        
    finally:
        kg.close()