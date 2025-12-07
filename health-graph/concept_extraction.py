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

# STRICT CONFIG (tune if needed)
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
    "skill","resource","process", "health care"
}

STOP_POS = {"VERB", "AUX", "ADV", "DET", "ADP"}

SPLIT_DELIMS = re.compile(r"[,;/]| and ", re.I)

# IDF thresholds
DF_RATIO_THRESHOLD = 0.10
MIN_IDF = 0.5
MIN_TOKEN_COUNT_FOR_KEEP = 2

# ============================================
# HIGH-VALUE AUDIT TERMS (Expand this list!)
# ============================================
# HIGH_VALUE_TERMS = [
#     # Medical Safety
#     {"label": "AUDIT_TERM", "pattern": "high alert medication"},
#     {"label": "AUDIT_TERM", "pattern": "high-alert medication"},
#     {"label": "AUDIT_TERM", "pattern": "patient identification"},
#     {"label": "AUDIT_TERM", "pattern": "patient safety"},
#     {"label": "AUDIT_TERM", "pattern": "infection control"},
#     {"label": "AUDIT_TERM", "pattern": "hand hygiene"},
    
#     # Documentation
#     {"label": "AUDIT_TERM", "pattern": "medical records"},
#     {"label": "AUDIT_TERM", "pattern": "medical record"},
#     {"label": "AUDIT_TERM", "pattern": "informed consent"},
#     {"label": "AUDIT_TERM", "pattern": "advance directive"},
    
#     # Clinical Services
#     {"label": "AUDIT_TERM", "pattern": "blood products"},
#     {"label": "AUDIT_TERM", "pattern": "blood transfusion"},
#     {"label": "AUDIT_TERM", "pattern": "diagnostic imaging"},
#     {"label": "AUDIT_TERM", "pattern": "laboratory services"},
    
#     # Staff & Training
#     {"label": "AUDIT_TERM", "pattern": "staff competency"},
#     {"label": "AUDIT_TERM", "pattern": "staff competence"},
#     {"label": "AUDIT_TERM", "pattern": "credentialing"},
#     {"label": "AUDIT_TERM", "pattern": "privileging"},
    
#     # Emergency & Safety
#     {"label": "AUDIT_TERM", "pattern": "fire safety"},
#     {"label": "AUDIT_TERM", "pattern": "fire safety plan"},
#     {"label": "AUDIT_TERM", "pattern": "emergency preparedness"},
#     {"label": "AUDIT_TERM", "pattern": "disaster preparedness"},
#     {"label": "AUDIT_TERM", "pattern": "code blue"},
    
#     # Quality & Risk
#     {"label": "AUDIT_TERM", "pattern": "quality improvement"},
#     {"label": "AUDIT_TERM", "pattern": "quality assurance"},
#     {"label": "AUDIT_TERM", "pattern": "risk management"},
#     {"label": "AUDIT_TERM", "pattern": "adverse event"},
#     {"label": "AUDIT_TERM", "pattern": "sentinel event"},
#     {"label": "AUDIT_TERM", "pattern": "root cause analysis"},
    
#     # Patient Rights
#     {"label": "AUDIT_TERM", "pattern": "patient rights"},
#     {"label": "AUDIT_TERM", "pattern": "privacy"},
#     {"label": "AUDIT_TERM", "pattern": "confidentiality"},
#     {"label": "AUDIT_TERM", "pattern": "do not resuscitate"},
#     {"label": "AUDIT_TERM", "pattern": "DNR"},
    
#     # Equipment & Environment
#     {"label": "AUDIT_TERM", "pattern": "medical equipment"},
#     {"label": "AUDIT_TERM", "pattern": "resuscitation equipment"},
#     {"label": "AUDIT_TERM", "pattern": "sterile processing"},
#     {"label": "AUDIT_TERM", "pattern": "waste management"},
    
#     # Add more terms from your 35 TPs here
# ]

# ============================================
# SPACY PIPELINE WITH ENTITYRULER
# ============================================

def get_nlp_pipeline():
    """Sets up the spaCy pipeline including the EntityRuler."""
    nlp = spacy.load("en_core_web_md")
    
    # Add EntityRuler before NER for high-priority matching
    if 'entity_ruler' not in nlp.pipe_names:
        ruler = nlp.add_pipe("entity_ruler", before="ner")
        ruler.add_patterns(audit_terms.HIGH_VALUE_TERMS)
        print(f"✓ EntityRuler added with {len(audit_terms.HIGH_VALUE_TERMS)} high-value patterns")
    
    return nlp

nlp = get_nlp_pipeline()

# ============================================
# CONCEPT EXTRACTION (ENHANCED)
# ============================================

def extract_domain_concepts(text):
    """
    High-precision concept extraction using:
    1. EntityRuler for known high-value audit terms
    2. Dependency parsing for discovering new terms
    Returns list of dicts: [{name: canonical, label: human_readable}]
    """
    doc = nlp(text)
    raw_candidates = set()
    high_priority_candidates = set()

    # -------------------------------------------------------
    # 1) ENTITY RULER (HIGH PRECISION BOOSTER) - NEW!
    # -------------------------------------------------------
    for ent in doc.ents:
        if ent.label_ == "AUDIT_TERM":
            # High-priority terms from our dictionary
            phrase = ent.text.strip().lower()
            high_priority_candidates.add(phrase)
            raw_candidates.add(phrase)
        elif ent.label_ in ["PRODUCT", "ORG", "GPE", "FACILITY"]:
            # Also capture standard NER entities that might be relevant
            phrase = ent.text.strip().lower()
            raw_candidates.add(phrase)

    # -------------------------------------------------------
    # 2) NOUN CHUNKS (primary source for new terms)
    # -------------------------------------------------------
    for chunk in doc.noun_chunks:
        phrase = chunk.text.strip().lower()

        # Skip trivial chunks or long multi-clauses
        if any(tok.pos_ in STOP_POS for tok in chunk):
            continue

        # Skip if root is generic
        if chunk.root.text.lower() in EXCLUDE_WORDS:
            continue

        raw_candidates.add(phrase)

    # -------------------------------------------------------
    # 3) COMPOUND NOUNS (e.g., high-alert medication)
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
    # 4) LIST SPLITTING (e.g., oral airways / iv cannula / oxygen)
    # -------------------------------------------------------
    split_candidates = set()
    for c in raw_candidates:
        parts = SPLIT_DELIMS.split(c)
        for p in parts:
            p = p.strip()
            if p:
                # Preserve high-priority status through splits
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

        # Remove punctuation around edges
        c = re.sub(r"^[\W_]+|[\W_]+$", "", c)

        # Remove duplicate whitespace
        c = re.sub(r"\s+", " ", c)

        # Remove leading determiners
        while True:
            parts = c.split()
            if parts and parts[0] in {"the", "a", "an"}:
                parts = parts[1:]
                c = " ".join(parts)
            else:
                break

        # Remove trailing noise tokens
        parts = c.split()
        while parts and parts[-1] in EXCLUDE_WORDS:
            parts.pop()
        c = " ".join(parts)

        # Remove empty or very short fragments
        if not c or len(c) < MIN_CHARS:
            continue

        # HIGH PRIORITY: Skip validation for EntityRuler matches
        if c in high_priority_candidates:
            cleaned.add(c)
            continue

        # Reject if contains verb/determiner/etc.
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
    # 6) CANONICALIZATION (PRESERVE WORD ORDER - FIXED!)
    # -------------------------------------------------------
    final_concepts = {}
    concept_priorities = {}  # Track which concepts are high-priority
    
    for label in cleaned:
        doc3 = nlp(label)

        # Check if this was a high-priority EntityRuler match
        is_high_priority = label in high_priority_candidates

        # Lemmatize while preserving order
        lemmas = []
        for tok in doc3:
            if tok.is_stop or tok.is_punct:
                continue
            if tok.text.isupper():
                lemmas.append(tok.text)  # Preserve acronyms
            else:
                lemmas.append(tok.lemma_.lower())

        # Remove duplicates while preserving order
        seen = set()
        dedup = []
        for l in lemmas:
            if l not in seen:
                seen.add(l)
                dedup.append(l)

        if not dedup:
            continue

        # PRESERVE WORD ORDER - Don't sort!
        canonical = "-".join(dedup)

        # Keep shortest label for display, but prioritize high-priority terms
        if canonical not in final_concepts:
            final_concepts[canonical] = label
            concept_priorities[canonical] = is_high_priority
        else:
            # Replace if this is high-priority and existing isn't
            # OR if both same priority and this label is shorter
            existing_priority = concept_priorities.get(canonical, False)
            if is_high_priority and not existing_priority:
                final_concepts[canonical] = label
                concept_priorities[canonical] = True
            elif is_high_priority == existing_priority and len(label) < len(final_concepts[canonical]):
                final_concepts[canonical] = label

    # -------------------------------------------------------
    # 7) REMOVE SUBSTRING DUPLICATES (NEW!)
    # -------------------------------------------------------
    # Sort by token count (descending) to process longer terms first
    sorted_concepts = sorted(
        final_concepts.items(), 
        key=lambda x: len(x[0].split('-')), 
        reverse=True
    )
    
    filtered_concepts = {}
    seen_tokens = set()
    
    for canonical, label in sorted_concepts:
        tokens = set(canonical.split('-'))
        
        # Check if this is a high-priority term
        is_priority = concept_priorities.get(canonical, False)
        
        # For high-priority terms, always keep them and mark tokens as seen
        if is_priority:
            filtered_concepts[canonical] = label
            seen_tokens.update(tokens)
            continue
        
        # For non-priority terms, check if it's a subset of an existing concept
        # Skip if ALL tokens are already seen (it's likely a substring)
        if tokens.issubset(seen_tokens) and len(tokens) < 3:
            # It's a 1-2 token phrase that's fully contained in a larger term
            continue
        
        # Keep this concept
        filtered_concepts[canonical] = label
        seen_tokens.update(tokens)

    # -------------------------------------------------------
    # 8) HANDLE HYPHENATION VARIATIONS (OPTIONAL)
    # -------------------------------------------------------
    # Merge variations like "health-care" and "healthcare"
    hyphen_normalized = {}
    for canonical, label in filtered_concepts.items():
        # Create a normalized key without hyphens for comparison
        normalized_key = canonical.replace('-', '')
        
        if normalized_key not in hyphen_normalized:
            hyphen_normalized[normalized_key] = (canonical, label)
        else:
            # Keep the version that was high-priority
            existing_canonical, existing_label = hyphen_normalized[normalized_key]
            existing_priority = concept_priorities.get(existing_canonical, False)
            current_priority = concept_priorities.get(canonical, False)
            
            if current_priority and not existing_priority:
                hyphen_normalized[normalized_key] = (canonical, label)
            elif current_priority == existing_priority:
                # Keep shorter version
                if len(label) < len(existing_label):
                    hyphen_normalized[normalized_key] = (canonical, label)
    
    # Final filtered list
    final_filtered = {canon: lbl for canon, lbl in hyphen_normalized.values()}
    
    return [{"name": canon, "label": lbl} for canon, lbl in final_filtered.items()]


# ============================================
# GRAPH BUILDER (Neo4j) - UNCHANGED
# ============================================

class KnowledgeGraphUpdater:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()
    
    def clear_concept_data(self):
        """Clears existing Concept nodes and MENTIONS relationships."""
        with self.driver.session() as session:
            session.run("MATCH (c:Concept) DETACH DELETE c")
            print("Cleared existing Concept nodes and MENTIONS relationships.")

    def fetch_all_clauses_from_graph(self):
        """Fetches all Clause IDs and Text from Neo4j."""
        with self.driver.session() as session:
            print("Fetching Clause data from Neo4j...")
            result = session.run("MATCH (c:Clause) RETURN c.id, c.text")
            return [dict(record) for record in result]

    def create_concept_nodes_and_relationships(self, clause_data):
        """Creates Concept nodes and MENTIONS relationships with IDF filtering."""
        all_concepts_map = {}
        clause_to_concept_names = defaultdict(list)

        # 1) Extraction
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
                if not name:
                    continue
                
                if name in all_concepts_map:
                    if len(label) < len(all_concepts_map[name]):
                        all_concepts_map[name] = label
                else:
                    all_concepts_map[name] = label
                
                names_for_clause.append(name)
            
            clause_to_concept_names[clause_id] = names_for_clause

        print(f"Total unique canonical concepts extracted: {len(all_concepts_map)}")
        
        # 2) IDF Filtering
        print("Applying IDF filtering...")
        N_clauses = len(clause_to_concept_names)
        df = {}
        for clause_id, cnames in clause_to_concept_names.items():
            for cname in set(cnames):
                df[cname] = df.get(cname, 0) + 1

        filtered_concepts = {}
        filtered_out = 0
        
        for cname, label in all_concepts_map.items():
            tokens = cname.replace('-', ' ').split()
            doc_freq = df.get(cname, 0)
            idf = log((N_clauses + 1) / (doc_freq + 1))
            
            # Filter rule: single-token + high frequency + low IDF = generic
            if len(tokens) == 1 and (doc_freq / N_clauses) > DF_RATIO_THRESHOLD and idf < MIN_IDF:
                filtered_out += 1
                continue
            
            filtered_concepts[cname] = {'label': label}

        print(f"Filtered out {filtered_out} generic concepts")
        print(f"Keeping {len(filtered_concepts)} concepts")

        # 3) Persist to Neo4j
        with self.driver.session() as session:
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (co:Concept) REQUIRE co.name IS UNIQUE")

            print("Creating Concept nodes...")
            concept_items = [{"name": k, "label": v['label']} for k, v in filtered_concepts.items()]
            session.execute_write(self._create_concepts_tx, concept_items)

            print("Creating MENTIONS relationships...")
            mention_count = 0
            for clause_id, concept_names in clause_to_concept_names.items():
                for cname in concept_names:
                    if cname in filtered_concepts:  # Only create if not filtered
                        session.run("""
                            MATCH (cl:Clause {id: $clause_id})
                            MATCH (co:Concept {name: $concept_name})
                            MERGE (cl)-[:MENTIONS]->(co)
                        """, clause_id=clause_id, concept_name=cname)
                        mention_count += 1

            print(f"Created {mention_count} MENTIONS relationships.")

    @staticmethod
    def _create_concepts_tx(tx, concepts):
        """Transaction function to batch create concept nodes."""
        query = """
        UNWIND $concepts AS concept
        MERGE (c:Concept {name: concept.name})
        SET c.label = concept.label
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
            print(f"✓ EntityRuler applied with {len(audit_terms.HIGH_VALUE_TERMS)} high-value patterns")
            print("✓ IDF filtering applied")
            print("✓ Word order preserved in canonicalization")
            print("\nRun this Cypher to inspect:")
            print("MATCH (cl:Clause)-[:MENTIONS]->(co:Concept)")
            print("RETURN cl.code, co.name, co.label LIMIT 25")
            
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        kg_updater.close()