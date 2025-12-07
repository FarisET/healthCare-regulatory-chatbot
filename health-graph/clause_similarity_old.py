import spacy
import re
from neo4j import GraphDatabase

# Load spaCy model
nlp = spacy.load("en_core_web_md")

# ============================================
# PART 1: EXTRACTION
# ============================================

def extract_clauses(filepath, framework_name):
    """Extract clauses based on specific regex patterns per framework."""
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    
    clauses = []
    
    if framework_name == "JCI":
        # JCI: Code followed by text
        matches = re.findall(r'([A-Z]{2,4}\.\d+(?:\.\d+)*)\s+([^\n]+)', text)
        for code, txt in matches:
            clauses.append({
                'code': code.strip(),
                'text': txt.strip(),
                'framework': 'JCI',
                'id': f"JCI_{code.strip()}" # Unique ID for Neo4j
            })
            
    elif framework_name == "SHCC":
        # SHCC: Number pattern handling multiline
        pattern = r'(\d+\.\d+\.\d+)\s*\n+\s*([^\d][^\n]*(?:\n(?!\d+\.\d+\.\d+)[^\n]*)*)'
        matches = re.findall(pattern, text, re.MULTILINE)
        for code, txt in matches:
            clean_text = ' '.join(txt.split())
            if clean_text:
                clauses.append({
                    'code': code.strip(),
                    'text': clean_text,
                    'framework': 'SHCC',
                    'id': f"SHCC_{code.strip()}" # Unique ID for Neo4j
                })
    
    return clauses

# ============================================
# PART 2: CONTROLLED SIMILARITY (Inter-Framework Only)
# ============================================

def find_best_matches(source_clauses, target_clauses, top_k=1, threshold=0.88):
    """
    Finds similarity ONLY between Source and Target lists.
    Returns only the TOP K matches per source clause to prevent edge explosion.
    """
    relationships = []
    
    print(f"Processing similarity: {len(source_clauses)} Source vs {len(target_clauses)} Target items...")
    
    # Pre-process docs to speed up loop
    source_docs = [(c, nlp(c['text'])) for c in source_clauses]
    target_docs = [(c, nlp(c['text'])) for c in target_clauses]
    
    for src_clause, src_doc in source_docs:
        matches = []
        
        for tgt_clause, tgt_doc in target_docs:
            score = src_doc.similarity(tgt_doc)
            if score >= threshold:
                matches.append({
                    'from': src_clause['id'],
                    'to': tgt_clause['id'],
                    'score': float(score), # Convert numpy float to python float
                    'source_text': src_clause['text'][:50], # For debugging
                    'target_text': tgt_clause['text'][:50]
                })
        
        # Sort matches by score descending
        matches.sort(key=lambda x: x['score'], reverse=True)
        
        # Take only Top K
        best_matches = matches[:top_k]
        relationships.extend(best_matches)

    return relationships

# ============================================
# PART 3: GRAPH BUILDER
# ============================================

class PipelineGraph:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def reset_graph(self):
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            # Create constraints for speed and data integrity
            try:
                session.run("CREATE CONSTRAINT FOR (c:Clause) REQUIRE c.id IS UNIQUE")
            except:
                pass # Constraint might already exist
            print("Graph cleared and constraints set.")

    def ingest_data(self, clauses, relationships):
        with self.driver.session() as session:
            # 1. Create Nodes
            print("Creating Nodes...")
            for c in clauses:
                session.run("""
                    MERGE (n:Clause {id: $id})
                    SET n.code = $code,
                        n.text = $text,
                        n.framework = $framework
                """, id=c['id'], code=c['code'], text=c['text'], framework=c['framework'])
            
            # 2. Create Edges
            print("Creating SIMILAR_TO Edges...")
            for r in relationships:
                session.run("""
                    MATCH (a:Clause {id: $from})
                    MATCH (b:Clause {id: $to})
                    MERGE (a)-[r:SIMILAR_TO]->(b)
                    SET r.score = $score
                """, r)

# ============================================
# EXECUTION
# ============================================

if __name__ == "__main__":
    # 1. Load Data
    jci = extract_clauses("../data/jci_standards.txt", "JCI")
    shcc = extract_clauses("../data/shcc_standards.txt", "SHCC")
    
    print(f"Loaded: JCI={len(jci)}, SHCC={len(shcc)}")
    
    # 2. Compute Similarity (JCI -> SHCC)
    # strict inter-framework, Top-1 match only
    similar_edges = find_best_matches(jci, shcc, top_k=1, threshold=0.88)
    
    print(f"Generated {len(similar_edges)} SIMILAR_TO edges (Top-1 strategy).")
    
    # 3. Push to Neo4j
    # Update these credentials
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "Ejaz@24470"
    
    kg = PipelineGraph(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    try:
        kg.reset_graph()
        kg.ingest_data(jci + shcc, similar_edges)
        print("Done. Check Neo4j.")
    finally:
        kg.close()