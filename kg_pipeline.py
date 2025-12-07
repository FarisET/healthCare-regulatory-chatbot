import spacy
import re
from collections import defaultdict
from neo4j import GraphDatabase

# Load spaCy model with word vectors
nlp = spacy.load("en_core_web_md")

# ============================================
# TASK 1: Extract CLAUSES from both frameworks
# ============================================

def extract_clauses_from_file(filepath, framework_name):
    """Extract clauses from a framework file."""
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    
    clauses = []
    
    if framework_name == "JCI":
        # JCI pattern: CODE.NUMBER followed by text on same line
        # Example: IPSG.3 The hospital develops...
        clause_pattern = r'([A-Z]{2,4}\.\d+(?:\.\d+)*)\s+([^\n]+)'
        matches = re.findall(clause_pattern, text)
        
        for code, clause_text in matches:
            clauses.append({
                'code': code,
                'text': clause_text.strip(),
                'framework': framework_name,
                'full_text': f"{code} {clause_text.strip()}"
            })
    
    elif framework_name == "SHCC":
        # SHCC pattern: NUMBER followed by text (possibly multi-line)
        # Example: 4.1.1 \n\n There is a universal mission statement...
        # Match number, then capture text until next number or end
        clause_pattern = r'(\d+\.\d+\.\d+)\s*\n+\s*([^\d][^\n]*(?:\n(?!\d+\.\d+\.\d+)[^\n]*)*)'
        matches = re.findall(clause_pattern, text, re.MULTILINE)
        
        for code, clause_text in matches:
            # Clean up extra whitespace and newlines
            clean_text = ' '.join(clause_text.split())
            if clean_text:  # Only add non-empty clauses
                clauses.append({
                    'code': code,
                    'text': clean_text.strip(),
                    'framework': framework_name,
                    'full_text': f"{code} {clean_text.strip()}"
                })
    
    return clauses

# Extract clauses from both frameworks
jci_clauses = extract_clauses_from_file("data/jci_standards.txt", "JCI")
shcc_clauses = extract_clauses_from_file("data/shcc_standards.txt", "SHCC")

all_clauses = jci_clauses + shcc_clauses
print(f"Extracted {len(jci_clauses)} JCI clauses and {len(shcc_clauses)} SHCC clauses")

# ============================================
# TASK 2: Link Similar CLAUSES across frameworks
# ============================================

def find_similar_clauses(clauses, similarity_threshold=0.85):
    """Find similar clauses between different frameworks."""
    similar_pairs = []
    
    # Group clauses by framework
    framework_groups = defaultdict(list)
    for clause in clauses:
        framework_groups[clause['framework']].append(clause)
    
    # Compare clauses across different frameworks
    frameworks = list(framework_groups.keys())
    for i in range(len(frameworks)):
        for j in range(i + 1, len(frameworks)):
            fw1, fw2 = frameworks[i], frameworks[j]
            
            for clause1 in framework_groups[fw1]:
                doc1 = nlp(clause1['text'])
                
                for clause2 in framework_groups[fw2]:
                    doc2 = nlp(clause2['text'])
                    
                    # Calculate similarity
                    similarity = doc1.similarity(doc2)
                    
                    if similarity >= similarity_threshold:
                        similar_pairs.append({
                            'clause1': clause1,
                            'clause2': clause2,
                            'similarity': similarity
                        })
    
    return similar_pairs

similar_clauses = find_similar_clauses(all_clauses, similarity_threshold=0.85)
print(f"Found {len(similar_clauses)} similar clause pairs across frameworks")

# ============================================
# TASK 3: Extract UNIQUE CONCEPTS from CLAUSES
# ============================================

def extract_concepts_from_clause(clause_text):
    """Extract medical/domain concepts from clause text."""
    doc = nlp(clause_text)
    concepts = set()
    
    # Extract noun chunks as potential concepts
    for chunk in doc.noun_chunks:
        # Filter out very short or common chunks
        if len(chunk.text.split()) >= 2 or chunk.root.pos_ in ['NOUN', 'PROPN']:
            concepts.add(chunk.text.lower().strip())
    
    # Extract named entities (organizations, medical terms, etc.)
    for ent in doc.ents:
        if ent.label_ in ['ORG', 'PRODUCT', 'LAW', 'WORK_OF_ART']:
            concepts.add(ent.text.lower().strip())
    
    # Extract key medical/domain terms (adjective + noun combinations)
    for token in doc:
        if token.pos_ == 'NOUN' and token.dep_ in ['dobj', 'pobj', 'nsubj']:
            # Get modifiers
            concept_parts = [child.text for child in token.lefts if child.pos_ in ['ADJ', 'NOUN']]
            concept_parts.append(token.text)
            concept_parts.extend([child.text for child in token.rights if child.pos_ in ['NOUN']])
            
            if len(concept_parts) >= 2:
                concepts.add(' '.join(concept_parts).lower().strip())
    
    return list(concepts)

# Extract concepts from all clauses
clause_concepts = {}
all_concepts = set()

for clause in all_clauses:
    concepts = extract_concepts_from_clause(clause['text'])
    clause_concepts[clause['code']] = concepts
    all_concepts.update(concepts)

print(f"Extracted {len(all_concepts)} unique concepts from all clauses")

# Create concept-clause mapping
concept_to_clauses = defaultdict(list)
for clause in all_clauses:
    for concept in clause_concepts[clause['code']]:
        concept_to_clauses[concept].append(clause['code'])

# ============================================
# TASK 4: Build Knowledge Graph in Neo4j
# ============================================

class KnowledgeGraphBuilder:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def close(self):
        self.driver.close()
    
    def clear_database(self):
        """Clear all nodes and relationships."""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            print("Database cleared")
    
    def create_clause_nodes(self, clauses):
        """Create CLAUSE nodes in Neo4j."""
        with self.driver.session() as session:
            for clause in clauses:
                session.run("""
                    CREATE (c:CLAUSE {
                        code: $code,
                        text: $text,
                        framework: $framework,
                        full_text: $full_text
                    })
                """, code=clause['code'], 
                     text=clause['text'],
                     framework=clause['framework'],
                     full_text=clause['full_text'])
        
        print(f"Created {len(clauses)} CLAUSE nodes")
    
    def create_similar_relationships(self, similar_pairs):
        """Create SIMILAR_TO relationships between clauses."""
        with self.driver.session() as session:
            for pair in similar_pairs:
                session.run("""
                    MATCH (c1:CLAUSE {code: $code1})
                    MATCH (c2:CLAUSE {code: $code2})
                    CREATE (c1)-[:SIMILAR_TO {score: $similarity}]->(c2)
                """, code1=pair['clause1']['code'],
                     code2=pair['clause2']['code'],
                     similarity=pair['similarity'])
        
        print(f"Created {len(similar_pairs)} SIMILAR_TO relationships")
    
    def create_concept_nodes_and_relationships(self, clause_concepts, all_concepts):
        """Create CONCEPT nodes and MENTIONS relationships."""
        with self.driver.session() as session:
            # Create concept nodes
            for concept in all_concepts:
                session.run("""
                    MERGE (c:CONCEPT {name: $name})
                """, name=concept)
            
            print(f"Created {len(all_concepts)} CONCEPT nodes")
            
            # Create MENTIONS relationships
            mention_count = 0
            for clause_code, concepts in clause_concepts.items():
                for concept in concepts:
                    session.run("""
                        MATCH (cl:CLAUSE {code: $clause_code})
                        MATCH (co:CONCEPT {name: $concept})
                        MERGE (cl)-[:MENTIONS]->(co)
                    """, clause_code=clause_code, concept=concept)
                    mention_count += 1
            
            print(f"Created {mention_count} MENTIONS relationships")
    
    def get_statistics(self):
        """Get graph statistics."""
        with self.driver.session() as session:
            # Count nodes
            result = session.run("""
                MATCH (c:CLAUSE) RETURN count(c) as clause_count
            """)
            clause_count = result.single()['clause_count']
            
            result = session.run("""
                MATCH (c:CONCEPT) RETURN count(c) as concept_count
            """)
            concept_count = result.single()['concept_count']
            
            # Count relationships
            result = session.run("""
                MATCH ()-[r:SIMILAR_TO]->() RETURN count(r) as similar_count
            """)
            similar_count = result.single()['similar_count']
            
            result = session.run("""
                MATCH ()-[r:MENTIONS]->() RETURN count(r) as mention_count
            """)
            mention_count = result.single()['mention_count']
            
            print("\n=== Knowledge Graph Statistics ===")
            print(f"CLAUSE nodes: {clause_count}")
            print(f"CONCEPT nodes: {concept_count}")
            print(f"SIMILAR_TO relationships: {similar_count}")
            print(f"MENTIONS relationships: {mention_count}")

# ============================================
# Main execution
# ============================================

if __name__ == "__main__":
    # Neo4j connection details
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "Faris@24470"
    
    # Build the knowledge graph
    kg = KnowledgeGraphBuilder(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    
    try:
        # Optional: Clear existing data
        # kg.clear_database()
        
        # Create nodes and relationships
        kg.create_clause_nodes(all_clauses)
        kg.create_similar_relationships(similar_clauses)
        kg.create_concept_nodes_and_relationships(clause_concepts, all_concepts)
        
        # Show statistics
        kg.get_statistics()
        
    finally:
        kg.close()
    
    print("\n=== Knowledge Graph Construction Complete ===")
    print("You can now query the graph in Neo4j Browser")
    print("\nExample Cypher queries:")
    print("1. Find all concepts mentioned by a clause:")
    print("   MATCH (cl:CLAUSE {code: 'IPSG.3'})-[:MENTIONS]->(co:CONCEPT) RETURN co.name")
    print("\n2. Find similar clauses across frameworks:")
    print("   MATCH (c1:CLAUSE)-[r:SIMILAR_TO]->(c2:CLAUSE) RETURN c1.code, c2.code, r.score")
    print("\n3. Find clauses connected through shared concepts:")
    print("   MATCH (c1:CLAUSE)-[:MENTIONS]->(co:CONCEPT)<-[:MENTIONS]-(c2:CLAUSE)")
    print("   WHERE c1.framework <> c2.framework RETURN c1.code, co.name, c2.code")