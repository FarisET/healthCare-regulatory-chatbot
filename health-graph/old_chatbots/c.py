import re
import spacy
import os
from dotenv import load_dotenv
from neo4j import GraphDatabase
from collections import defaultdict
from math import log
import audit_terms
load_dotenv()

# ============================================
# CONFIGURATION
# ============================================

# STRICT CONFIG
MAX_TOKENS = 6
MAX_CHARS = 60
MIN_CHARS = 3

EXCLUDE_WORDS = {
    # Function words
    "the","a","an","in","is","of","and","or","for","to",
    "not","was","are","by","on","it","its","as",
    # Generic hospital terms
    "hospital","process","system","procedure","policies",
    "all","patient","patients","ch","chart","unit","units",
    # Generic nouns to filter
    "staff","equipment","service","treatment","information",
    "place","area","time","use","data","action","duty","case",
    "evidence","individual","decision","response","condition",
    "responsibility","accordance","experience","access",
    "management","education","training","charge","compliance",
    "skill","resource","process"
}

# NEW: Strict filter for problematic single nouns
EXCLUDE_SINGLE_NOUNS = {
    "minute", "arrival", "sex", "extent", "site", "order", 
    "aids", "number", "attendance", "visit", "visits",
    "date", "name", "age", "address", "telephone", "occupation",
    "school", "history", "finding", "detail", "record"
}

STOP_POS = {"VERB", "AUX", "ADV", "DET", "ADP"}

SPLIT_DELIMS = re.compile(r"[,;/]| and ", re.I)

# IDF thresholds
DF_RATIO_THRESHOLD = 0.10
MIN_IDF = 0.5
MIN_TOKEN_COUNT_FOR_KEEP = 2

# ============================================
# HIGH-VALUE AUDIT TERMS
# ============================================


# ============================================
# SPACY PIPELINE WITH ENTITYRULER
# ============================================

def get_nlp_pipeline():
    """Sets up the spaCy pipeline including the EntityRuler."""
    nlp = spacy.load("en_core_web_md")
    
    if 'entity_ruler' not in nlp.pipe_names:
        ruler = nlp.add_pipe("entity_ruler", before="ner")
        ruler.add_patterns(audit_terms.HIGH_VALUE_TERMS)
        print(f"✓ EntityRuler added with {len(audit_terms.HIGH_VALUE_TERMS)} high-value patterns")
    
    return nlp

nlp = get_nlp_pipeline()

# ============================================
# DEFINITION PATTERN EXTRACTION
# ============================================

def extract_concept_and_description(clause_text):
    """
    Detects definition patterns like "Clinical review includes: X, Y, Z"
    Returns the primary concept and prevents list items from becoming FPs.
    """
    
    # Pattern: Concept phrase + cue word + colon/semicolon + description
    pattern = r"^(.*?)(?:\s+includes?|\s+is defined as|\s+are defined as|\s+must include|\s+refers? to|\s+covers?|\s+consists? of|\s+are|\s+is|\s+such as|for example|e\.?g\.?)\s*[:;]\s*(.+)$"
    
    match = re.search(pattern, clause_text, re.DOTALL | re.IGNORECASE)
    
    if match:
        primary_phrase = match.group(1).strip()
        description_list = match.group(2).strip()
        
        # Validate: phrase should be 2+ words and end with noun/adjective
        doc = nlp(primary_phrase)
        
        if len(doc) >= 2 and doc[-1].pos_ in ["NOUN", "PROPN", "ADJ"]:
            # Clean description
            description_list = re.sub(r'(\s*[-•*]|\d+\.)\s*', ' ', description_list)
            description_list = re.sub(r'\s{2,}', ' ', description_list).strip()
            
            # Extract canonical form of primary concept
            lemmas = []
            for tok in doc:
                if tok.is_stop or tok.is_punct:
                    continue
                if tok.text.isupper():
                    lemmas.append(tok.text)
                else:
                    lemmas.append(tok.lemma_.lower())
            
            if lemmas:
                canonical = "-".join(lemmas)
                return {
                    "primary_concept": canonical,
                    "label": primary_phrase,
                    "description": description_list
                }
    
    return None

# ============================================
# CONCEPT EXTRACTION (ENHANCED)
# ============================================

def extract_domain_concepts(text):
    """
    High-precision concept extraction with:
    1. Definition pattern detection (highest priority)
    2. EntityRuler for known audit terms
    3. Dependency parsing for new terms
    4. Strict single-word filtering
    5. Substring deduplication
    """
    
    # -------------------------------------------------------
    # 0) DEFINITION CHECK (Priority over all other extraction)
    # -------------------------------------------------------
    definition_match = extract_concept_and_description(text)
    
    if definition_match:
        # Only return the primary concept with its description
        return [definition_match]
    
    # --- If no definition pattern, proceed with standard extraction ---
    
    doc = nlp(text)
    raw_candidates = set()
    high_priority_candidates = set()

    # -------------------------------------------------------
    # 1) ENTITY RULER (HIGH PRECISION BOOSTER)
    # -------------------------------------------------------
    for ent in doc.ents:
        if ent.label_ == "AUDIT_TERM":
            phrase = ent.text.strip().lower()
            high_priority_candidates.add(phrase)
            raw_candidates.add(phrase)
        elif ent.label_ in ["PRODUCT", "ORG", "GPE", "FACILITY"]:
            phrase = ent.text.strip().lower()
            raw_candidates.add(phrase)

    # -------------------------------------------------------
    # 2) NOUN CHUNKS (primary source for new terms)
    # -------------------------------------------------------
    for chunk in doc.noun_chunks:
        phrase = chunk.text.strip().lower()

        if any(tok.pos_ in STOP_POS for tok in chunk):
            continue

        if chunk.root.text.lower() in EXCLUDE_WORDS:
            continue

        raw_candidates.add(phrase)

    # -------------------------------------------------------
    # 3) COMPOUND NOUNS
    # -------------------------------------------------------
    for token in doc:
        if token.pos_ == "NOUN" and token.text.lower() not in EXCLUDE_WORDS:
            modifiers = [
                child.text for child in token.lefts
                if child.dep_ in {"amod", "compound"} and child.text.lower() not in EXCLUDE_WORDS
            ]
            if modifiers:
                phrase = " ".join([*modifiers, token.text]).lower()
                raw_candidates.add(phrase)

    # -------------------------------------------------------
    # 4) LIST SPLITTING
    # -------------------------------------------------------
    split_candidates = set()
    for c in raw_candidates:
        parts = SPLIT_DELIMS.split(c)
        for p in parts:
            p = p.strip()
            if p:
                if c in high_priority_candidates:
                    high_priority_candidates.add(p)
                split_candidates.add(p)
    raw_candidates = split_candidates

    # -------------------------------------------------------
    # 5) CLEANUP AND VALIDATION
    # -------------------------------------------------------
    cleaned = set()

    for c in raw_candidates:
        c = c.strip().lower()

        # Remove punctuation
        c = re.sub(r"^[\W_]+|[\W_]+$", "", c)
        c = re.sub(r"\s+", " ", c)

        # Remove leading determiners
        while True:
            parts = c.split()
            if parts and parts[0] in {"the", "a", "an"}:
                parts = parts[1:]
                c = " ".join(parts)
            else:
                break

        # Remove trailing noise
        parts = c.split()
        while parts and parts[-1] in EXCLUDE_WORDS:
            parts.pop()
        c = " ".join(parts)

        if not c or len(c) < MIN_CHARS:
            continue

        # **NEW: STRICT SINGLE-WORD FILTER**
        tokens = c.split()
        if len(tokens) == 1 and c in EXCLUDE_SINGLE_NOUNS:
            continue

        # HIGH PRIORITY: Skip validation
        if c in high_priority_candidates:
            cleaned.add(c)
            continue

        # Validation for normal priority
        doc2 = nlp(c)
        if any(tok.pos_ in STOP_POS for tok in doc2):
            continue

        # Token length enforcement
        if len(tokens) > MAX_TOKENS:
            tokens = tokens[-MAX_TOKENS:]
            c = " ".join(tokens)

        if len(c) > MAX_CHARS:
            continue

        cleaned.add(c)

    # -------------------------------------------------------
    # 6) CANONICALIZATION
    # -------------------------------------------------------
    final_concepts = {}
    concept_priorities = {}
    
    for label in cleaned:
        doc3 = nlp(label)
        is_high_priority = label in high_priority_candidates

        # Lemmatize
        lemmas = []
        for tok in doc3:
            if tok.is_stop or tok.is_punct:
                continue
            if tok.text.isupper():
                lemmas.append(tok.text)
            else:
                lemmas.append(tok.lemma_.lower())

        # Deduplicate while preserving order
        seen = set()
        dedup = []
        for l in lemmas:
            if l not in seen:
                seen.add(l)
                dedup.append(l)

        if not dedup:
            continue

        canonical = "-".join(dedup)

        # Prioritize high-priority terms
        if canonical not in final_concepts:
            final_concepts[canonical] = label
            concept_priorities[canonical] = is_high_priority
        else:
            existing_priority = concept_priorities.get(canonical, False)
            if is_high_priority and not existing_priority:
                final_concepts[canonical] = label
                concept_priorities[canonical] = True
            elif is_high_priority == existing_priority and len(label) < len(final_concepts[canonical]):
                final_concepts[canonical] = label

    # -------------------------------------------------------
    # 7) REMOVE SUBSTRING DUPLICATES
    # -------------------------------------------------------
    sorted_concepts = sorted(
        final_concepts.items(), 
        key=lambda x: len(x[0].split('-')), 
        reverse=True
    )
    
    filtered_concepts = {}
    seen_tokens = set()
    
    for canonical, label in sorted_concepts:
        tokens = set(canonical.split('-'))
        is_priority = concept_priorities.get(canonical, False)
        
        # Always keep high-priority
        if is_priority:
            filtered_concepts[canonical] = label
            seen_tokens.update(tokens)
            continue
        
        # Skip if ALL tokens seen and < 3 tokens (likely substring)
        if tokens.issubset(seen_tokens) and len(tokens) < 3:
            continue
        
        filtered_concepts[canonical] = label
        seen_tokens.update(tokens)

    # -------------------------------------------------------
    # 8) HANDLE HYPHENATION VARIATIONS
    # -------------------------------------------------------
    hyphen_normalized = {}
    for canonical, label in filtered_concepts.items():
        normalized_key = canonical.replace('-', '')
        
        if normalized_key not in hyphen_normalized:
            hyphen_normalized[normalized_key] = (canonical, label)
        else:
            existing_canonical, existing_label = hyphen_normalized[normalized_key]
            existing_priority = concept_priorities.get(existing_canonical, False)
            current_priority = concept_priorities.get(canonical, False)
            
            if current_priority and not existing_priority:
                hyphen_normalized[normalized_key] = (canonical, label)
            elif current_priority == existing_priority and len(label) < len(existing_label):
                hyphen_normalized[normalized_key] = (canonical, label)
    
    final_filtered = {canon: lbl for canon, lbl in hyphen_normalized.values()}
    
    return [
        {
            "name": canon, 
            "label": lbl,
            "description": None
        } 
        for canon, lbl in final_filtered.items()
    ]

# ============================================
# GRAPH BUILDER (Neo4j)
# ============================================

class KnowledgeGraphUpdater:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()
    
    def clear_concept_data(self):
        with self.driver.session() as session:
            session.run("MATCH (c:Concept) DETACH DELETE c")
            print("Cleared existing Concept nodes and MENTIONS relationships.")

    def fetch_all_clauses_from_graph(self):
        with self.driver.session() as session:
            print("Fetching Clause data from Neo4j...")
            result = session.run("MATCH (c:Clause) RETURN c.id, c.text")
            return [dict(record) for record in result]

    def create_concept_nodes_and_relationships(self, clause_data):
        all_concepts_map = {}
        clause_to_concept_names = defaultdict(list)

        # Extract concepts
        print("Extracting concepts from clauses...")
        for clause in clause_data:
            clause_id = clause.get('c.id') or clause.get('id')
            clause_text = clause.get('c.text') or clause.get('text')
            
            if clause_id is None:
                raise ValueError("Clause record missing 'id' field")
            
            concepts = extract_domain_concepts(clause_text or "")
            names_for_clause = []
            
            for c in concepts:
                name = c.get('name')
                label = c.get('label') or name
                description = c.get('description')
                
                if not name:
                    continue
                
                if name in all_concepts_map:
                    if len(label) < len(all_concepts_map[name]['label']):
                        all_concepts_map[name] = {'label': label, 'description': description}
                else:
                    all_concepts_map[name] = {'label': label, 'description': description}
                
                names_for_clause.append(name)
            
            clause_to_concept_names[clause_id] = names_for_clause

        print(f"Total unique canonical concepts extracted: {len(all_concepts_map)}")
        
        # IDF Filtering
        print("Applying IDF filtering...")
        N_clauses = len(clause_to_concept_names)
        df = {}
        for clause_id, cnames in clause_to_concept_names.items():
            for cname in set(cnames):
                df[cname] = df.get(cname, 0) + 1

        filtered_concepts = {}
        filtered_out = 0
        
        for cname, info in all_concepts_map.items():
            tokens = cname.replace('-', ' ').split()
            doc_freq = df.get(cname, 0)
            idf = log((N_clauses + 1) / (doc_freq + 1))
            
            if len(tokens) == 1 and (doc_freq / N_clauses) > DF_RATIO_THRESHOLD and idf < MIN_IDF:
                filtered_out += 1
                continue
            
            filtered_concepts[cname] = info

        print(f"Filtered out {filtered_out} generic concepts")
        print(f"Keeping {len(filtered_concepts)} concepts")

        # Persist to Neo4j
        with self.driver.session() as session:
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (co:Concept) REQUIRE co.name IS UNIQUE")

            print("Creating Concept nodes...")
            concept_items = [
                {
                    "name": k, 
                    "label": v['label'],
                    "description": v['description']
                } 
                for k, v in filtered_concepts.items()
            ]
            session.execute_write(self._create_concepts_tx, concept_items)

            print("Creating MENTIONS relationships...")
            mention_count = 0
            for clause_id, concept_names in clause_to_concept_names.items():
                for cname in concept_names:
                    if cname in filtered_concepts:
                        session.run("""
                            MATCH (cl:Clause {id: $clause_id})
                            MATCH (co:Concept {name: $concept_name})
                            MERGE (cl)-[:MENTIONS]->(co)
                        """, clause_id=clause_id, concept_name=cname)
                        mention_count += 1

            print(f"Created {mention_count} MENTIONS relationships.")

    @staticmethod
    def _create_concepts_tx(tx, concepts):
        query = """
        UNWIND $concepts AS concept
        MERGE (c:Concept {name: concept.name})
        SET c.label = concept.label,
            c.description = concept.description
        """
        tx.run(query, concepts=concepts)

# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    neo4j_uri = os.environ.get("NEO4J_URI")
    neo4j_user = os.environ.get("NEO4J_USER")
    neo4j_password = os.environ.get("NEO4J_PASSWORD")
    
    kg_updater = KnowledgeGraphUpdater(neo4j_uri, neo4j_user, neo4j_password)
    
    try:
        kg_updater.clear_concept_data()
        clauses_from_graph = kg_updater.fetch_all_clauses_from_graph()
        
        if not clauses_from_graph:
            print("Error: No CLAUSE nodes found in the database.")
        else:
            kg_updater.create_concept_nodes_and_relationships(clauses_from_graph)
            
            print("\n=== Concept Extraction Complete ===")
            print("✓ Definition pattern detection applied")
            print("✓ EntityRuler applied with high-value patterns")
            print("✓ Strict single-word filtering")
            print("✓ Substring deduplication")
            print("✓ IDF filtering applied")
            
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        kg_updater.close()