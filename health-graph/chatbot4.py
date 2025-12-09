#!/usr/bin/env python3
"""
Refactored Graph RAG pipeline (read-only, safe, better prompts).
Requirements:
 - python-dotenv
 - langchain_community (Neo4jGraph)
 - langchain_experimental.llms.ollama_functions (or adjust to your LLM wrapper)
 - langchain_ollama (if using embeddings later)
Set env vars: NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
"""

import os
import json
from string import Template
from dotenv import load_dotenv

# Langchain/neoj4 wrappers you already used
from langchain_community.graphs import Neo4jGraph
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# ----------------- CONFIG -----------------
NEO4J_URI = os.environ.get("NEO4J_URI")
NEO4J_USER = os.environ.get("NEO4J_USER")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

# Fulltext index name you must create in Neo4j if not present:
# CALL db.index.fulltext.createNodeIndex("ConceptIndex", ["Concept"], ["name","label","aliases"])
FULLTEXT_INDEX_NAME = "ConceptIndex"

# Ollama LLMs: one for structured extraction (json) and one for synthesis
entity_llm = OllamaFunctions(model="llama3.1:8b", temperature=0, format="json")
qa_llm = OllamaFunctions(model="llama3.1:8b", temperature=0)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GOOGLE_API_KEY)

# Neo4j graph connection (read-only user recommended)
graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USER, password=NEO4J_PASSWORD, sanitize=True)


# ----------------- UTIL: Robust LLM JSON parse -----------------
def call_and_parse_json(llm, prompt: str, fallback=None):
    """
    Call an LLM and attempt to extract JSON from its response robustly.
    Strips code fences and leading/trailing text.
    """
    raw = ""
    try:
        resp = llm.invoke(prompt)
        raw = getattr(resp, "content", "") or str(resp)
        text = raw

        # If code fence with json block, prefer that
        if "```json" in text:
            text = text.split("```json", 1)[1].rsplit("```", 1)[0]
        elif "```" in text:
            # take content inside first code fence
            text = text.split("```", 1)[1].rsplit("```", 1)[0]

        text = text.strip()

        # sometimes the model adds extraneous prefix/suffix — attempt to find first '{' ... last '}'
        if not text.startswith("{"):
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                text = text[start:end+1]

        parsed = json.loads(text)
        return parsed
    except Exception as e:
        print("⚠️ LLM JSON parse failed:", e)
        if raw:
            preview = raw[:1000] + ("..." if len(raw) > 1000 else "")
            print("LLM raw output preview:", preview)
        return fallback if fallback is not None else {}


# ----------------- PROMPT (Template to avoid brace-escaping) -----------------
ENTITY_PROMPT_TMPL = Template("""
Extract key information from this hospital audit question.

Question: "$question"

Return ONLY JSON in this exact format (no extra commentary):

{
  "topics": ["hand hygiene", "medication safety"],
  "frameworks": ["JCI", "SHCC"],
  "query_type": "comparison"  // one of "comparison" | "search" | "specific"
}

JSON:
""")

def extract_entities(question: str):
    """
    Returns dict: {'topics': [...], 'frameworks': [...], 'query_type': 'comparison'|'search'|'specific'}
    Uses the entity_llm to extract structured JSON.
    """
    prompt = ENTITY_PROMPT_TMPL.substitute(question=question)
    parsed = call_and_parse_json(llm, prompt, fallback={})
    # normalization + defensive defaults
    topics = parsed.get("topics") or []
    frameworks = parsed.get("frameworks") or []
    qtype = parsed.get("query_type") or ("comparison" if "compare" in question.lower() else "search")
    return {"topics": topics, "frameworks": frameworks, "query_type": qtype}


# ----------------- Neo4j Fulltext concept lookup -----------------
def find_concepts_for_topic(topic: str, limit=8):
    """
    Uses Neo4j fulltext index for Concepts to get the best matching concept node IDs.
    Falls back to a simple CONTAINS match if fulltext call fails.
    """
    cypher_fulltext = """
    CALL db.index.fulltext.queryNodes($index, $query) YIELD node, score
    WHERE $label IN labels(node)
    RETURN id(node) AS node_id, node.name AS name, node.label AS label, score
    ORDER BY score DESC
    LIMIT $limit
    """
    params = {"index": FULLTEXT_INDEX_NAME, "query": topic, "label": "Concept", "limit": limit}
    try:
        rows = graph.query(cypher_fulltext, params)
        return [r.get("node_id") for r in rows]
    except Exception as e:
        print("⚠️ Fulltext lookup failed:", e)
        # fallback (slower) - case-insensitive contains
        cy = """
        MATCH (c:Concept)
        WHERE toLower(coalesce(c.name,'')) CONTAINS toLower($q) OR toLower(coalesce(c.label,'')) CONTAINS toLower($q)
        RETURN id(c) AS node_id LIMIT $limit
        """
        rows = graph.query(cy, {"q": topic, "limit": limit})
        return [r.get("node_id") for r in rows]


# ----------------- Fetch clauses by concept ids (parameterized) -----------------
def fetch_clauses_for_concepts(concept_ids: list, frameworks: list = None, limit=200):
    """
    Safe, parameterized Cypher to fetch clauses that mention given concept ids and match frameworks.
    """
    if not concept_ids:
        return []
    # ensure frameworks defaults
    frameworks = frameworks or []
    cypher = """
    MATCH (cl:Clause)-[:MENTIONS]->(co)
    WHERE id(co) IN $concept_ids
      AND ($frameworks_size = 0 OR cl.framework IN $frameworks)
    RETURN DISTINCT cl.code AS code, cl.text AS text, cl.framework AS framework, id(co) AS concept_id
    LIMIT $limit
    """
    params = {
        "concept_ids": concept_ids,
        "frameworks": frameworks,
        "frameworks_size": len(frameworks),
        "limit": limit,
    }
    try:
        rows = graph.query(cypher, params)
        return rows
    except Exception as e:
        print("⚠️ Clause fetch failed:", e)
        return []


# ----------------- Synthesize final answer using clauses + qa_llm -----------------
SYNTH_PROMPT_TMPL = Template("""
You are an expert hospital compliance assistant.

User question:
$question

Below are retrieved clause snippets (showing framework and code). Use them to produce:
- A concise comparison between the frameworks (if comparison requested)
- Or a short list of relevant clauses and a short explanation (if search/specific)
Be factual, cite clause codes and frameworks in the answer, and do NOT invent clause codes.

Retrieved clauses:
$clauses_text

Answer (concise):
""")

def synthesize_answer(question: str, clauses: list, qtype: str):
    # prepare clause snippets for the LLM
    snippet_lines = []
    for i, r in enumerate(clauses, start=1):
        fw = r.get("framework") or "UNKNOWN"
        code = r.get("code") or "N/A"
        text = (r.get("text") or "").replace("\n", " ").strip()[:900]
        snippet_lines.append(f"{i}. [{fw}] {code} — {text}")
    clauses_text = "\n".join(snippet_lines[:30]) if snippet_lines else "No clauses retrieved."

    prompt = SYNTH_PROMPT_TMPL.substitute(question=question, clauses_text=clauses_text)
    resp = qa_llm.invoke(prompt)
    return getattr(resp, "content", str(resp))


# ----------------- High-level pipeline -----------------
def process_query(question: str, top_k_concepts=5):
    print("\n[1] Extracting entities from question...")
    entities = extract_entities(question)
    print("  → topics:", entities.get("topics"))
    print("  → frameworks:", entities.get("frameworks"))
    print("  → query_type:", entities.get("query_type"))

    # Collect concept ids by topic
    all_concept_ids = []
    for t in entities.get("topics", []):
        ids = find_concepts_for_topic(t, limit=top_k_concepts)
        if ids:
            print(f"  → topic='{t}' found concept ids: {ids}")
            all_concept_ids.extend(ids)
    # dedupe and preserve order
    seen = set()
    concept_ids = []
    for cid in all_concept_ids:
        if cid not in seen:
            seen.add(cid)
            concept_ids.append(cid)

    if not concept_ids:
        print("❌ No concepts found for the provided topics. Try broader terms.")
        return

    print(f"[2] Fetching clauses for {len(concept_ids)} concept ids...")
    clauses = fetch_clauses_for_concepts(concept_ids, frameworks=entities.get("frameworks") or [], limit=200)
    print(f"  → retrieved {len(clauses)} clause records.")

    if not clauses:
        print("❌ No clauses found matching those concepts & frameworks.")
        return

    print("[3] Synthesizing answer with LLM...")
    answer = synthesize_answer(question, clauses, entities.get("query_type"))
    print("\n--- ANSWER ---\n")
    print(answer)
    print("\n---------------\n")


# ----------------- Simple interactive loop -----------------
if __name__ == "__main__":
    print("🏥 Hospital Compliance Assistant (read-only mode). Type 'exit' to quit.")
    while True:
        try:
            q = input("\nQuestion (or 'exit'): ").strip()
            if not q:
                continue
            if q.lower() in ("exit", "quit", "q"):
                break
            process_query(q)
        except KeyboardInterrupt:
            print("\nInterrupted. Exiting.")
            break
        except Exception as e:
            print("Unhandled error:", e)
            import traceback
            traceback.print_exc()
