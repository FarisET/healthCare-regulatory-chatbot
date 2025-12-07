import re
import os
from dotenv import load_dotenv
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer, util
import torch

# ============================================
# PART 1: EXTRACTION (Standard)
# ============================================
load_dotenv()

def extract_clauses(filepath, framework_name):
    # ... (Same extraction code as previous step - no changes needed here) ...
    # Copy pasting the fixed regex version for completeness
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    clauses = []
    if framework_name == "JCI":
        jci_pattern = r'([A-Z]{2,4}\.\d+(?:\.\d+)*)\s+(.+?)(?=\n[A-Z]{2,4}\.\d|\Z)'
        matches = re.findall(jci_pattern, text, re.DOTALL)
        
        for code, txt in matches:
            # 1. Clean up multi-line spaces/newlines
            clean_text = ' '.join(txt.split()).strip()
            
            # 2. Aggressive artifact removal (The new step)
            # Remove common document artifacts like page numbers, document names, etc.
            # This targets patterns like "19 --- Page 20 --- Joint Commission..."
            clean_text = re.sub(r'\s*\d+\s*---\s*Page\s*\d+\s*---\s*.*$', '', clean_text, flags=re.IGNORECASE).strip()
            # Remove trailing single words that look like sections/footers ("Administration", "Admin")
            clean_text = re.sub(r'\s*[A-Z][a-z]+$', '', clean_text).strip()
            
            # 3. Final cleanup for trailing symbols/junk
            clean_text = clean_text.rstrip(' P') 
            
            clauses.append({
                'code': code.strip(),
                'text': clean_text,
                'framework': 'JCI',
                'id': f"JCI_{code.strip()}"
            })
    # Updated SHCC block (inside extract_clauses)
    elif framework_name == "SHCC":
        
        # NEW PATTERN: Catches both 3-level (1.1.1) and 2-level (15.1) codes
        # (\d+\.\d+(?:\.\d+)*)  <-- This optional group allows 15.1 or 1.1.1
        # Match code, then use lookahead to capture text until the next code block starts.
        flexible_code_pattern = r'(\d+\.\d+(?:\.\d+)*)\s*\n+\s*(.+?)(?=\n\s*\d+\.\d+|\Z)'
        
        # We must use re.DOTALL here too, as the content can span multiple lines
        matches = re.findall(flexible_code_pattern, text, re.DOTALL)
        
        for code, txt in matches:
            # Clean up extra whitespace and newlines
            clean_text = ' '.join(txt.split())
            
            if clean_text:
                clauses.append({
                    'code': code.strip(),
                    'text': clean_text,
                    'framework': 'SHCC',
                    'id': f"SHCC_{code.strip()}"
                }) 
    return clauses

# ============================================
# PART 2: STRICT 1:1 MATCHING (The Fix)
# ============================================

def find_strict_1to1_matches(source_clauses, target_clauses, threshold=0.70):
    """
    Enforces that any clause can have AT MOST one SIMILAR_TO relationship.
    Prioritizes the highest scores globally.
    """
    print("Loading SBERT model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    src_texts = [c['text'] for c in source_clauses]
    tgt_texts = [c['text'] for c in target_clauses]
    
    print("Computing embeddings...")
    src_embeddings = model.encode(src_texts, convert_to_tensor=True)
    tgt_embeddings = model.encode(tgt_texts, convert_to_tensor=True)
    
    print("Computing similarity matrix...")
    # Compute cosine similarity between all pairs
    cosine_scores = util.cos_sim(src_embeddings, tgt_embeddings)
    
    # Convert matrix to a list of (score, src_idx, tgt_idx)
    # This allows us to sort ALL possibilities globally
    all_pairs = []
    rows, cols = cosine_scores.shape
    for i in range(rows):
        for j in range(cols):
            score = float(cosine_scores[i][j])
            if score >= threshold:
                all_pairs.append((score, i, j))
    
    # Sort by score descending (Highest confidence first)
    all_pairs.sort(key=lambda x: x[0], reverse=True)
    
    print(f"Found {len(all_pairs)} potential matches above threshold {threshold}.")
    print("Filtering for strict 1-to-1 mapping...")
    
    relationships = []
    matched_src_indices = set()
    matched_tgt_indices = set()
    
    for score, src_idx, tgt_idx in all_pairs:
        # If either clause is already matched, SKIP IT
        if src_idx in matched_src_indices or tgt_idx in matched_tgt_indices:
            continue
            
        # Otherwise, create the match
        source_c = source_clauses[src_idx]
        target_c = target_clauses[tgt_idx]
        
        relationships.append({
            'from': source_c['id'],
            'to': target_c['id'],
            'score': score,
            'source_code': source_c['code'],
            'target_code': target_c['code']
        })
        
        # Mark these indices as taken
        matched_src_indices.add(src_idx)
        matched_tgt_indices.add(tgt_idx)
        
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
                # 1. Clear existing data
                session.run("MATCH (n) DETACH DELETE n")
                
                # 2. Update Constraint Creation
                # The 'IF NOT EXISTS' clause prevents the error you saw.
                session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Clause) REQUIRE c.id IS UNIQUE")
                print("Graph cleared and uniqueness constraint ensured.")
                
    def ingest_data(self, clauses, relationships):
        with self.driver.session() as session:
            # Nodes
            print("Ingesting Nodes...")
            for c in clauses:
                session.run("""
                    MERGE (n:Clause {id: $id})
                    SET n.code = $code, n.text = $text, n.framework = $framework
                """, c)
            
            # Edges
            print(f"Ingesting {len(relationships)} strict 1:1 Edges...")
            for r in relationships:
                session.run("""
                    MATCH (a:Clause {id: $from})
                    MATCH (b:Clause {id: $to})
                    MERGE (a)-[r:SIMILAR_TO]->(b)
                    SET r.score = $score
                """, r)

# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    jci = extract_clauses("../data/jci_standards.txt", "JCI")
    shcc = extract_clauses("../data/shcc_standards.txt", "SHCC")
    
    # Threshold 0.70 is quite strict.
    # If you get too few matches, try 0.65. If you still get junk, try 0.75.
    strict_edges = find_strict_1to1_matches(jci, shcc, threshold=0.70)
    
    neo4j_uri = os.environ.get("NEO4J_URI")
    neo4j_user = os.environ.get("NEO4J_USER")
    neo4j_password = os.environ.get("NEO4J_PASSWORD")

    
    kg = PipelineGraph(neo4j_uri, neo4j_user, neo4j_password)
    try:
        kg.reset_graph()
        kg.ingest_data(jci + shcc, strict_edges)
        print("Done.")
    finally:
        kg.close()