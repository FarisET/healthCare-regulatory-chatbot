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

MAX_TOKENS = 6
MAX_CHARS = 60
MIN_CHARS = 3

EXCLUDE_WORDS = {
    "the","a","an","in","is","of","and","or","for","to",
    "not","was","are","by","on","it","its","as",
    "hospital","process","system","procedure","policies",
    "all","patient","patients","ch","chart","unit","units",
    "staff","equipment","service","treatment","information",
    "place","area","time","use","data","action","duty","case",
    "evidence","individual","decision","response","condition",
    "responsibility","accordance","experience","access",
    "management","education","training","charge","compliance",
    "skill","resource","process","health care"
}

# EXPANDED: More problematic single nouns
EXCLUDE_SINGLE_NOUNS = {
    "minute", "arrival", "sex", "extent", "site", "order", "aids",
    "number", "attendance", "visit", "visits", "date", "name", 
    "age", "address", "telephone", "occupation", "school", "history",
    "finding", "detail", "record", "requirement", "activity", "member",
    "level", "type", "result", "location", "status", "issue",
    "example", "provision", "basis", "purpose", "nature", "range"
}

STOP_POS = {"VERB", "AUX", "ADV", "DET", "ADP"}
SPLIT_DELIMS = re.compile(r"[,;/]| and ", re.I)

# IDF thresholds
DF_RATIO_THRESHOLD = 0.10
MIN_IDF = 0.5

# ============================================
# SPACY PIPELINE
# ============================================

def get_nlp_pipeline():
    nlp = spacy.load("en_core_web_md")
    
    if 'entity_ruler' not in nlp.pipe_names:
        ruler = nlp.add_pipe("entity_ruler", before="ner")
        ruler.add_patterns(audit_terms.HIGH_VALUE_TERMS)
        print(f"✓ EntityRuler added with {len(audit_terms.HIGH_VALUE_TERMS)} patterns")
    
    return nlp

nlp = get_nlp_pipeline()

# ============================================
# DEFINITION PATTERN EXTRACTION
# ============================================

def extract_concept_and_description(clause_text):
    """
    Detects definition patterns: "Clinical review includes: X, Y, Z"
    Returns primary concept with description to prevent list items becoming FPs.
    """
    # Pattern with multiple cue phrases
    pattern = r"^(.*?)\s*(?:includes?|is defined as|are defined as|must include|refers? to|covers?|consists? of|such as|for example|e\.?g\.?)\s*[:;]\s*(.+)$"
    
    match = re.search(pattern, clause_text, re.DOTALL | re.IGNORECASE)
    
    if match:
        primary_phrase = match.group(1).strip()
        description_list = match.group(2).strip()
        
        # Validate: must be 2+ words and end with noun/adjective
        doc = nlp(primary_phrase)
        
        if len(doc) >= 2 and doc[-1].pos_ in ["NOUN", "PROPN", "ADJ"]:
            # Clean description
            description_list = re.sub(r'(\s*[-•*]|\d+\.)\s*', ' ', description_list)
            description_list = re.sub(r'\s{2,}', ' ', description_list).strip()
            
            # Canonicalize primary concept
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
                    "name": canonical,
                    "label": primary_phrase,
                    "description": description_list
                }
    
    return None

# ============================================
# CONCEPT EXTRACTION
# ============================================

def extract_domain_concepts(text):
    """
    High-precision concept extraction with:
    1. Definition pattern detection (highest priority)
    2. EntityRuler for known audit terms
    3. Single-concept enforcement per clause
    """
    
    # -------------------------------------------------------
    # 0) DEFINITION PATTERN CHECK (HIGHEST PRIORITY)
    # -------------------------------------------------------
    definition_match = extract_concept_and_description(text)
    
    if definition_match:
        # Return ONLY the primary concept with description
        return [definition_match]
    
    # --- Standard extraction if no definition pattern ---
    
    doc = nlp(text)
    raw_candidates = set()
    high_priority_candidates = set()

    # -------------------------------------------------------
    # 1) ENTITY RULER (HIGH PRECISION)
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
    # 2) NOUN CHUNKS
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

        # HIGH PRIORITY: Skip validation
        if c in high_priority_candidates:
            cleaned.add(c)
            continue

        # STRICT SINGLE-WORD FILTER
        if len(c.split()) == 1 and c in EXCLUDE_SINGLE_NOUNS:
            continue

        # Validation
        doc2 = nlp(c)
        if any(tok.pos_ in STOP_POS for tok in doc2):
            continue

        # Token length enforcement
        tokens = c.split()
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

        # Deduplicate preserving order
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
        
        # Skip if ALL tokens seen and < 3 tokens
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
    
    # -------------------------------------------------------
    # 9) SINGLE-CONCEPT ENFORCEMENT (NEW!)
    # -------------------------------------------------------
    # If multiple concepts remain, keep only the most specific one
    if len(final_filtered) > 3:
        # Sort by specificity: longer token count + high priority wins
        sorted_by_specificity = sorted(
            final_filtered.items(),
            key=lambda x: (
                concept_priorities.get(x[0], False),  # Priority first
                len(x[0].split('-')),                 # Then token count
                len(x[0])                              # Then character length
            ),
            reverse=True
        )
        
        # Keep top 2-3 most specific concepts
        final_filtered = dict(sorted_by_specificity[:2])
    
    return [
        {
            "name": canon, 
            "label": lbl, 
            "description": None
        } 
        for canon, lbl in final_filtered.items()
    ]


# ============================================
# GRAPH BUILDER
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
        """Creates Concept nodes with descriptions and MENTIONS relationships."""
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
                
                # Store concept with description
                if name in all_concepts_map:
                    # Keep shorter label, but preserve description
                    if len(label) < len(all_concepts_map[name]['label']):
                        all_concepts_map[name]['label'] = label
                    # Prefer non-null descriptions
                    if description and not all_concepts_map[name]['description']:
                        all_concepts_map[name]['description'] = description
                else:
                    all_concepts_map[name] = {
                        'label': label,
                        'description': description
                    }
                
                names_for_clause.append(name)
            
            clause_to_concept_names[clause_id] = names_for_clause

        print(f"Total unique concepts extracted: {len(all_concepts_map)}")
        
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

            print("Creating Concept nodes with descriptions...")
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
            print("Error: No CLAUSE nodes found.")
        else:
            kg_updater.create_concept_nodes_and_relationships(clauses_from_graph)
            
            print("\n=== Concept Extraction Complete ===")
            print("✓ Definition pattern detection applied")
            print("✓ Single-concept enforcement (max 2-3 per clause)")
            print("✓ Description property stored")
            print(f"✓ EntityRuler with {len(audit_terms.HIGH_VALUE_TERMS)} patterns")
            print("✓ Expanded single-word filter")
            print("\nVerify with:")
            print("MATCH (co:Concept) WHERE co.description IS NOT NULL")
            print("RETURN co.name, co.description LIMIT 10")
            
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        kg_updater.close()