#!/usr/bin/env python3
"""
LangGraph RAG Agent using StateGraph and tools bound to an LLM.

Save as langgraph_lan_graph_agent.py and run after setting:
  NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, GOOGLE_API_KEY

Requirements:
 - langgraph
 - langchain_core
 - langchain_google_genai (or swap to your LLM provider)
 - langchain_neo4j or langchain_community
 - python-dotenv
"""

import os
import json
from typing import TypedDict, Annotated, Sequence
from operator import add as add_messages
from dotenv import load_dotenv

# LangGraph + LangChain core message & tool types
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

try:
    # prefer langchain_neo4j if available
    from langchain_neo4j import Neo4jGraph
except Exception:
    from langchain_community.graphs import Neo4jGraph

from langchain_google_genai import ChatGoogleGenerativeAI
import traceback, time, re, json

load_dotenv()

# ---------------- CONFIG ----------------
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "password")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
FULLTEXT_INDEX_NAME = os.environ.get("FULLTEXT_INDEX_NAME", "ConceptIndex")

# --------- LLMs & Graph connection ----------
# Orchestrator LLM (bound to tools) — deterministic -> temperature=0
orchestrator_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, google_api_key=GOOGLE_API_KEY)
# LLMs used by some tools for structured extraction & synthesis (you can reuse orchestrator_llm if desired)
entity_llm = orchestrator_llm
qa_llm = orchestrator_llm

# Neo4j (read-only user recommended)
neo4j_graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USER, password=NEO4J_PASSWORD, sanitize=True)


# ---------------- Helpers / prompts ----------------
def _run_cypher_debug(cypher: str, params: dict = None):
    """Run cypher on neo4j_graph with debug printing and return rows (list-of-records)."""
    params = params or {}
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[DEBUG] CYPHER @ {ts}")
    print("------- CYPHER -------")
    print(cypher.strip())
    print("------- PARAMS -------")
    print(json.dumps(params, ensure_ascii=False, indent=2))
    print("----------------------")
    try:
        rows = neo4j_graph.query(cypher, params)
        num = len(rows) if hasattr(rows, "__len__") else "unknown"
        print(f"[DEBUG] Query returned {num} rows.")
        return rows
    except Exception as e:
        print("[DEBUG] Cypher execution error:", e)
        traceback.print_exc()
        # propagate so calling tool can handle
        raise

def sanitize_tool_output(res_text: str) -> str:
    """
    Remove LLM/tool metadata and provide a clean human-friendly string.
    - If JSON with 'markdown' return that markdown.
    - If JSON list of messages (with 'text'), join them.
    - Otherwise strip long 'signature' extras and return text.
    """
    if not res_text:
        return ""
    # try JSON parse
    try:
        obj = json.loads(res_text)
        # dict with markdown
        if isinstance(obj, dict):
            if "markdown" in obj:
                return obj["markdown"]
            # if top-level 'analysis' contains markdown
            if "analysis" in obj and isinstance(obj["analysis"], dict) and "markdown" in obj["analysis"]:
                return obj["analysis"]["markdown"]
            # if keys look like human text, try to extract 'text'
            if "text" in obj and isinstance(obj["text"], str):
                return obj["text"]
            # fallback: pretty JSON string
            return json.dumps(obj, ensure_ascii=False, indent=2)
        # list -> extract text fields
        if isinstance(obj, list):
            texts = []
            for item in obj:
                if isinstance(item, dict) and "text" in item:
                    texts.append(item["text"])
                else:
                    texts.append(json.dumps(item, ensure_ascii=False))
            return "\n\n".join(texts)
    except Exception:
        pass

    # fallback: remove signature/extras blocks (common in LLM metadata)
    cleaned = re.sub(r"['\"]?extras['\"]?\s*:\s*\{[^}]+\}", "", res_text, flags=re.DOTALL)
    # reduce very long single-token signatures
    cleaned = re.sub(r"[A-Za-z0-9+/=]{80,}", "[long_signature_removed]", cleaned)
    return cleaned.strip()

def safe_json_dumps(obj):
    return json.dumps(obj, default=str, ensure_ascii=False)

def rows_to_list(rows):
    try:
        return [dict(r) for r in rows]
    except Exception:
        out = []
        for r in rows:
            try:
                out.append({k: r.get(k) for k in r.keys()})
            except Exception:
                out.append(str(r))
        return out

def call_and_parse_json(llm, prompt: str, fallback=None):
    raw = ""
    try:
        resp = llm.invoke(prompt)
        raw = getattr(resp, "content", "") or str(resp)
        text = raw
        if "```json" in text:
            text = text.split("```json", 1)[1].rsplit("```", 1)[0]
        elif "```" in text:
            text = text.split("```", 1)[1].rsplit("```", 1)[0]
        text = text.strip()
        if not text.startswith("{"):
            start = text.find("{"); end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                text = text[start:end+1]
        return json.loads(text)
    except Exception as e:
        print("⚠️ LLM JSON parse failed:", e)
        if raw:
            preview = raw[:1000] + ("..." if len(raw) > 1000 else "")
            print("LLM raw output preview:", preview)
        return fallback if fallback is not None else {}

ENTITY_PROMPT_TMPL = """
Extract key information from this hospital audit question.

Question: "{question}"

Return ONLY JSON in this exact format:

{
  "topics": ["hand hygiene", "medication safety"],
  "frameworks": ["JCI", "SHCC"],
  "query_type": "comparison"
}
"""

SYNTH_PROMPT_TMPL = """
You are an expert Quality Management and Regulatory Compliance Auditor. Use the retrieved clauses to provide a concise, factual answer.

Question:
{question}

Clauses:
{clauses_text}

Answer:
"""

# ---------------- Tools ----------------
# Each tool returns a JSON string (so the LLM can receive tool outputs easily)

@tool
def extract_entities_tool(question: str) -> str:
    """Return JSON: {'topics':[], 'frameworks':[], 'query_type':...}"""
    prompt = ENTITY_PROMPT_TMPL.format(question=question)
    parsed = call_and_parse_json(entity_llm, prompt, fallback={})
    topics = parsed.get("topics") or []
    frameworks = parsed.get("frameworks") or []
    qtype = parsed.get("query_type") or ("comparison" if "compare" in question.lower() else "search")
    out = {"topics": topics, "frameworks": frameworks, "query_type": qtype}
    return safe_json_dumps(out)

@tool
def find_concepts_tool(topic: str, limit: int = 8) -> str:
    """Fulltext + fallback to find concepts. Returns JSON list of {element_id, name, label, score?}"""
    cypher_fulltext = """
    CALL db.index.fulltext.queryNodes($index, $query) YIELD node, score
    RETURN elementId(node) AS element_id, node.name AS name, node.label AS label, score
    ORDER BY score DESC
    LIMIT $limit
    """
    try:
        rows = _run_cypher_debug(cypher_fulltext, {"index": FULLTEXT_INDEX_NAME, "query": topic, "limit": limit})
        return safe_json_dumps(rows_to_list(rows))
    except Exception as e:
        # fallback
        cy = """
        MATCH (c:Concept)
        WHERE toLower(coalesce(c.name,'')) CONTAINS toLower($q)
           OR toLower(coalesce(c.label,'')) CONTAINS toLower($q)
        RETURN elementId(c) AS element_id, c.name AS name, c.label AS label
        LIMIT $limit
        """
        rows = _run_cypher_debug(cy, {"q": topic, "limit": limit})
        return safe_json_dumps(rows_to_list(rows))


@tool
def fetch_clauses_tool(concept_element_ids_json: str, frameworks_json: str = "[]", limit: int = 200) -> str:
    """Fetch clauses by element ids. Returns JSON list of clauses."""
    try:
        in_obj = json.loads(concept_element_ids_json)
        if in_obj and isinstance(in_obj[0], dict) and "element_id" in in_obj[0]:
            ids = [r["element_id"] for r in in_obj]
        else:
            ids = in_obj
    except Exception:
        ids = []

    try:
        frameworks = json.loads(frameworks_json)
    except Exception:
        frameworks = []

    if not ids:
        return safe_json_dumps([])

    cypher = """
    UNWIND $concept_ids AS cid
    MATCH (co) WHERE elementId(co) = cid
    WITH co
    MATCH (cl:Clause)-[:MENTIONS]->(co)
    WHERE ($frameworks_size = 0 OR cl.framework IN $frameworks)
    RETURN DISTINCT cl.code AS code, cl.text AS text, cl.framework AS framework, elementId(co) AS concept_id
    LIMIT $limit
    """
    rows = _run_cypher_debug(cypher, {"concept_ids": ids, "frameworks": frameworks, "frameworks_size": len(frameworks), "limit": limit})
    return safe_json_dumps(rows_to_list(rows))

@tool
def compare_frameworks_tool(topic: str) -> str:
    """Compare clauses mapping across frameworks using SIMILAR_TO edges (returns JSON list)."""
    cypher = """
    MATCH (c:Concept)
    WHERE toLower(c.name) CONTAINS toLower($topic)
       OR toLower(c.label) CONTAINS toLower($topic)
       OR any(a IN coalesce(c.aliases,[]) WHERE toLower(a) CONTAINS toLower($topic))
    WITH c
    MATCH (j:Clause)-[:MENTIONS]->(c)
    OPTIONAL MATCH (j)-[s:SIMILAR_TO]-(k:Clause)
    WHERE s.score IS NOT NULL
    RETURN c.name AS concept,
           j.framework AS framework_a, j.code AS code_a, j.text AS text_a,
           k.framework AS framework_b, k.code AS code_b, k.text AS text_b,
           s.score AS similarity
    ORDER BY s.score DESC
    LIMIT 50
    """
    try:
        rows = neo4j_graph.query(cypher, {"topic": topic})
        return safe_json_dumps(rows_to_list(rows))
    except Exception as e:
        return safe_json_dumps({"error": str(e)})

@tool
def analyze_audit_gaps_tool(finding_keyword: str) -> str:
    """Find past audit findings containing a keyword (returns JSON list)."""
    cypher = """
    MATCH (f:AuditFinding)
    WHERE toLower(f.text) CONTAINS toLower($keyword)
    MATCH (ci:ClauseInstance)-[:HAS_FINDING]->(f)
    MATCH (ci)-[:PART_OF]->(audit:Audit)
    MATCH (ci)-[:INSTANCE_OF]->(clause:Clause)
    RETURN audit.date AS date, clause.code AS standard_violated, f.text AS finding, f.grade AS severity, f.status AS status
    ORDER BY audit.date DESC
    LIMIT 20
    """
    try:
        rows = neo4j_graph.query(cypher, {"keyword": finding_keyword})
        return safe_json_dumps(rows_to_list(rows))
    except Exception as e:
        return safe_json_dumps({"error": str(e)})



@tool
def analyze_department_checklist_tool(department_name: str) -> str:
    """
    Department checklist analysis (auditor-friendly):

    1) Find clauses APPLIES_TO the department
    2) For each such clause, find SIMILAR_TO clauses that are NOT linked to the department (strong suggestions)
    3) If no strong suggestions, find clauses that share Concepts (weak suggestions)

    Returns JSON: {"analysis": {...}, "markdown": "..."}.

    Design goals:
    - No numeric "gap score" in user-facing output
    - No jargon like SIMILAR_TO, APPLIES_TO, similarity_score
    - For each suggested clause, show the linked clause it is similar to
    - Provide a short, natural-language rationale for why the suggestion is relevant
    - Keep deterministic structure so the LLM can reliably build on it
    """
    try:
        # 1) Clauses currently linked to the department
        cy_linked = """
        MATCH (d:Department)
        WHERE toLower(d.name) CONTAINS toLower($dept)
        WITH d
        MATCH (c:Clause)-[:APPLIES_TO]->(d)
        RETURN elementId(c) AS clause_id,
               c.framework AS framework,
               c.code AS code,
               c.text AS text
        """
        linked_rows = _run_cypher_debug(cy_linked, {"dept": department_name})
        linked = rows_to_list(linked_rows)
        linked_count = len(linked)
        print(f"[DEBUG] linked_clauses count: {linked_count}")

        # If nothing is mapped yet, return a simple message
        if not linked:
            md = (
                f"# Checklist Gap Analysis of {department_name}\n\n"
                f"## Summary\n"
                f"No clauses are currently linked to this department in the model.\n\n"
                f"- The Emergency / clinical checklist for this department is not yet mapped to any standard clauses.\n\n"
                f"## Recommendations\n"
                f"- First, map your department checklist items to the relevant Clause nodes.\n"
                f"- Once the mapping is in place, re-run this analysis to see targeted recommendations.\n"
            )
            analysis = {
                "department": department_name,
                "linked_clauses_count": 0,
                "strong_suggestions_count": 0,
                "weak_suggestions_count": 0,
                "strong_suggestions_topk": [],
                "note": "No clauses currently linked to this department."
            }
            return safe_json_dumps({"analysis": analysis, "markdown": md})

        # 2) Strong suggestions: candidate clauses similar to linked ones, but not yet linked to the department
        cy_strong = """
        MATCH (d:Department)
        WHERE toLower(d.name) CONTAINS toLower($dept)
        WITH d
        MATCH (linked:Clause)-[:APPLIES_TO]->(d)
        OPTIONAL MATCH (linked)-[s:SIMILAR_TO]-(cand:Clause)
        WHERE cand IS NOT NULL
          AND cand.code IS NOT NULL
          AND NOT (cand)-[:APPLIES_TO]->(d)
        RETURN elementId(linked) AS linked_clause_id,
               linked.code AS linked_code,
               linked.framework AS linked_framework,
               linked.text AS linked_text,
               elementId(cand) AS cand_clause_id,
               cand.framework AS cand_framework,
               cand.code AS cand_code,
               cand.text AS cand_text,
               s.score AS similarity
        """
        strong_rows = _run_cypher_debug(cy_strong, {"dept": department_name})
        strong_list = rows_to_list(strong_rows)
        print(f"[DEBUG] strong_rows returned: {len(strong_list)}")

        # Filter out any candidates without code
        strong_filtered = [r for r in strong_list if r.get("cand_code")]
        print(f"[DEBUG] strong_filtered (cand_code not null): {len(strong_filtered)}")
        if strong_filtered:
            print("[DEBUG] sample strong suggestion:", strong_filtered[0])

        # 3) Weak suggestions via shared Concepts (only if needed)
        weak = []
        if not strong_filtered:
            cy_weak = """
            MATCH (d:Department)
            WHERE toLower(d.name) CONTAINS toLower($dept)
            WITH d
            MATCH (linked:Clause)-[:APPLIES_TO]->(d)
            MATCH (linked)-[:MENTIONS]->(co:Concept)<-[:MENTIONS]-(other:Clause)
            WHERE NOT (other)-[:APPLIES_TO]->(d)
              AND elementId(other) <> elementId(linked)
            RETURN elementId(linked) AS linked_clause_id,
                   linked.code AS linked_code,
                   linked.framework AS linked_framework,
                   elementId(other) AS other_clause_id,
                   other.framework AS other_framework,
                   other.code AS other_code,
                   collect(DISTINCT co.name)[0..5] AS shared_concepts,
                   other.text AS other_text
            LIMIT 200
            """
            weak_rows = _run_cypher_debug(cy_weak, {"dept": department_name})
            weak = rows_to_list(weak_rows)
            print(f"[DEBUG] weak_rows returned: {len(weak)}")
            if weak:
                print("[DEBUG] sample weak suggestion:", weak[0])

        # Helper: numeric similarity value (for internal ranking only)
        def _sim_val(r):
            try:
                return float(r.get("similarity") or 0.0)
            except Exception:
                return 0.0

        # Rank strong suggestions by similarity for internal use, but we won't show the raw score to the user
        strong_sorted = sorted(strong_filtered, key=_sim_val, reverse=True)
        top_k = 1
        strong_topk = strong_sorted[:top_k]
        strong_count = len(strong_filtered)
        weak_count = len(weak)
        print(f"[DEBUG] strong_topk count: {len(strong_topk)}")

        # ---------------- Deterministic analysis dict (kept for the orchestrator) ----------------
        analysis = {
            "department": department_name,
            "linked_clauses_count": linked_count,
            "strong_suggestions_count": strong_count,
            "weak_suggestions_count": weak_count,
            "strong_suggestions_topk": strong_topk,
            "note": (
                "Additional clauses identified that are closely related to existing checklist items."
                if strong_filtered else
                ("Concept-based suggestions identified (shared themes with existing clauses)." if weak else
                 "No additional suggested clauses found.")
            )
        }

        # ---------------- Build auditor-friendly Markdown ----------------
        md_lines = []
        md_lines.append(f"# Checklist Gap Analysis of {department_name}")
        md_lines.append("## Summary")
        md_lines.append(
            f"Your current checklist for this department includes **{linked_count}** mapped clauses."
        )
        if strong_filtered:
            md_lines.append(
                f"We identified **{strong_count} additional clauses** that are closely related "
                f"to what you already cover but are **not yet included** in this department's checklist."
            )
            md_lines.append("### My Thought Process")
            md_lines.append(
                "These suggestions come from clauses that address similar requirements "
                "to the ones already linked for this department (for example, the same process "
                "such as patient transfers, documentation, or communication, but with extra detail "
                "or from another framework)."
            )

        elif weak:
            md_lines.append(
                f"No directly related clauses were found, but we identified **{weak_count} clauses** "
                "that share important themes or concepts with your existing checklist items."
            )
        else:
            md_lines.append(
                "We did not find any additional clauses that clearly extend or complement your current checklist."
            )

        # -------------- Strong suggestions: show top-K with linked clause + reason --------------
        if strong_topk:
            md_lines.append("## Top Recommended Additions")
            md_lines.append(
                "Below are the most relevant additional clauses. For each one, we show "
                "which existing requirement it aligns with and why it may be useful."
            )
            for idx, it in enumerate(strong_topk, start=1):
                cand_framework = it.get("cand_framework")
                cand_code = it.get("cand_code")
                linked_framework = it.get("linked_framework")
                linked_code = it.get("linked_code")
                # We don't expose similarity, but we can bias the language for high vs. medium match if desired.
                sim_val = _sim_val(it)

                # Very light heuristic wording (optional; doesn't expose numbers)
                if sim_val >= 0.80:
                    relation_phrase = "very closely aligned with"
                elif sim_val >= 0.65:
                    relation_phrase = "closely related to"
                else:
                    relation_phrase = "related to"

                md_lines.append(f"### {idx}. [{cand_framework}] {cand_code}")
                md_lines.append(
                    f"**Related existing clause:** [{linked_framework}] {linked_code}"
                )
                # md_lines.append(
                #     f"- This suggested clause is **{relation_phrase}** an existing requirement in your checklist.\n"
                #     f"- It likely covers a similar process or risk area but may add more explicit detail or a slightly different perspective.\n"
                #     f"- Consider adding it if it strengthens how this process is controlled in the department (for example, clearer handover, more precise documentation, or tighter communication expectations)."
                # )
                # --- NEW: Summaries + overlap reasoning ---
                linked_text = (it.get("linked_text") or "").strip()
                cand_text = (it.get("cand_text") or "").strip()

                # Short summaries (LLM-friendly deterministic truncation)
                linked_summary = linked_text[:350] + ("..." if len(linked_text) > 350 else "")
                cand_summary = cand_text[:350] + ("..." if len(cand_text) > 350 else "")

                md_lines.append("#### Summary of Clauses")
                md_lines.append(f"**Summary of related existing clause [{linked_framework}] {linked_code}:**")
                md_lines.append(f"{linked_summary}")
                md_lines.append(f"**Summary of suggested clause [{cand_framework}] {cand_code}:**")
                md_lines.append(f"{cand_summary}")
                md_lines.append("### Why you should include this clause")
                md_lines.append(
                    "> LLM: Using the summaries above, explain how the suggested clause complements or extends "
                    "the existing one. Focus on where the suggested clause adds detail, strengthens controls, "
                    "or covers aspects not fully addressed by the related existing clause."
                )
            if strong_count > len(strong_topk):
                md_lines.append(
                    f"_There are **{strong_count}** suggested clauses in total; only the top **{len(strong_topk)}** are shown here._"
                )
                md_lines.append(
                    "If you would like to review more of these suggestions, you can ask to **show more**."
                )
                md_lines.append("")

        # -------------- Weak suggestions: only if no strong suggestions --------------
        if not strong_topk and weak:
            md_lines.append("## Concept-based suggestions (themes in common)")
            md_lines.append(
                "We did not find directly related clauses from other standards, but we did identify "
                "clauses that share important themes or concepts with your existing checklist."
            )
            md_lines.append("Here are a few examples:")
            md_lines.append("")

            for r in weak[:6]:
                other_framework = r.get("other_framework")
                other_code = r.get("other_code")
                shared = ", ".join(r.get("shared_concepts") or [])
                linked_code = r.get("linked_code")
                linked_framework = r.get("linked_framework")

                md_lines.append(f"- **[{other_framework}] {other_code}**")
                md_lines.append(
                    f"  - Related existing clause: [{linked_framework}] {linked_code}"
                )
                if shared:
                    md_lines.append(
                        f"  - Shared themes: {shared}"
                    )
                md_lines.append(
                    "  - This clause touches on similar risk or process areas and may help you refine or extend your checklist."
                )
                md_lines.append("")

        # -------------- Recommendations --------------
        md_lines.append("## Recommendations")
        if strong_topk or weak:
            md_lines.append(
                "- **Review the suggested clauses** above and decide whether they should be added "
                "to the department checklist."
            )
            md_lines.append(
                "- **Review the suggested clauses*. If the department already has robust local procedures that cover the same intent, "
                "you may document the rationale for not adding a clause rather than duplicating controls."
            )
        else:
            md_lines.append(
                "- No obvious checklist gaps were detected based on the modeled data. You can still "
                "perform a qualitative review of high-risk processes in the department to confirm coverage."
            )

        md = "\n".join(md_lines)
        return safe_json_dumps({"analysis": analysis, "markdown": md})

    except Exception as e:
        print("[ERROR] analyze_department_checklist_tool:", e)
        traceback.print_exc()
        return safe_json_dumps({"error": "Tool execution failed. See logs."})

def _format_department_analysis_md(analysis: dict) -> str:
    """
    Deterministically format the analysis dict into a readable Markdown string.
    """
    dept = analysis.get("department", "Unknown Department")
    linked = analysis.get("linked_clauses", [])
    strong = analysis.get("strong_suggestions", [])
    weak = analysis.get("weak_suggestions", [])
    note = analysis.get("note", "")

    lines = []
    lines.append(f"# Department Checklist Analysis — **{dept}**\n")
    lines.append(f"**Summary:** {note}\n")

    # Linked clauses
    lines.append("## Clauses already linked to the department (APPLIES_TO)\n")
    if not linked:
        lines.append("_No linked clauses found._\n")
    else:
        for i, c in enumerate(linked, start=1):
            code = c.get("code", "N/A")
            fw = c.get("framework", "N/A")
            text = (c.get("text") or "").strip()
            lines.append(f"{i}. **[{fw}] {code}** — {text[:380]}{'...' if len(text) > 380 else ''}\n")

    # Strong suggestions
    lines.append("## Strong suggestions (SIMILAR_TO linked clause, not applied to department)\n")
    if strong:
        # group by linked_clause
        by_linked = {}
        for s in strong:
            lid = s.get("linked_clause_id")
            by_linked.setdefault(lid, []).append(s)
        for lid, items in by_linked.items():
            linked_code = items[0].get("linked_code", "N/A")
            lines.append(f"### Based on linked clause **{linked_code}**:\n")
            for it in items:
                sim = it.get("similarity")
                cand_code = it.get("cand_code", "N/A")
                cand_fw = it.get("cand_framework", "N/A")
                cand_text = (it.get("cand_text") or "").strip()
                lines.append(f"- **[{cand_fw}] {cand_code}** (similarity={sim}) — {cand_text[:300]}{'...' if len(cand_text) > 300 else ''}\n")
    else:
        lines.append("_No strong SIMILAR_TO suggestions found._\n")

    # Weak suggestions
    lines.append("## Weak suggestions (share Concepts with linked clauses)\n")
    if weak:
        # group by linked_clause_id
        by_linked_w = {}
        for w in weak:
            lid = w.get("linked_clause_id")
            by_linked_w.setdefault(lid, []).append(w)
        for lid, items in by_linked_w.items():
            linked_code = items[0].get("linked_code", "N/A")
            lines.append(f"### Based on linked clause **{linked_code}**:\n")
            for it in items:
                other_code = it.get("other_code", "N/A")
                other_fw = it.get("other_framework", "N/A")
                shared = it.get("shared_concepts", []) or []
                lines.append(f"- **[{other_fw}] {other_code}** — shares concepts: {', '.join(shared)[:200]}\n")
    else:
        lines.append("_No weak suggestions found._\n")

    # Closing recommendations
    lines.append("\n## Recommendations\n")
    lines.append("1. Review the **Strong suggestions** and consider whether they align with department scope; if so, add via APPLIES_TO.\n")
    lines.append("2. For **Weak suggestions**, review shared concepts and decide whether broader checklist coverage is needed or whether the other clause is out of scope.\n")
    lines.append("3. Prioritize adding clauses with high similarity scores or multiple concept overlaps.\n")

    return "\n".join(lines)


@tool
def synthesize_answer_tool(question: str = "", clauses_json: str = "[]", qtype: str = "search", analysis_json: str = None) -> str:
    """
    Robust synth: can accept analysis JSON *as the first positional argument* (question),
    OR via the analysis_json parameter. Returns clean markdown.
    """
    try:
        # --- robust detection: if 'question' looks like JSON analysis, treat it as analysis_json ---
        maybe_json = None
        if question and isinstance(question, str):
            s = question.strip()
            if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
                # received JSON in the first positional arg -> treat it as analysis_json
                maybe_json = s

        # If analysis_json not provided but json found in question, use that
        if analysis_json is None and maybe_json:
            analysis_json = maybe_json

        # Now the rest of your previous logic expects analysis_json to be JSON string
        if analysis_json:
            obj = json.loads(analysis_json)
            if isinstance(obj, dict):
                if "markdown" in obj and isinstance(obj["markdown"], str):
                    return obj["markdown"]
                if "analysis" in obj and isinstance(obj["analysis"], dict) and "note" in obj["analysis"]:
                    md = obj.get("markdown") or ""
                    if md:
                        return md
                    analysis = obj["analysis"]
                    parts = []
                    parts.append(f"**Summary:** {analysis.get('note','')}")
                    if analysis.get("strong_suggestions_count", 0):
                        parts.append(f"- Found {analysis['strong_suggestions_count']} strong suggestions (SIMILAR_TO candidates).")
                    elif analysis.get("weak_suggestions_count", 0):
                        parts.append(f"- Found {analysis['weak_suggestions_count']} weak suggestions (shared concepts).")
                    else:
                        parts.append("- No data-driven suggestions found; see general guidance below.")
                    parts.append("\n**Generic Guidance (if no clauses available):**")
                    parts.append("- Review ED’s patient safety, infection control, equipment readiness, triage, handover & training items.")
                    return "\n".join(parts)

        clauses = []
        try:
            clauses = json.loads(clauses_json)
        except Exception:
            clauses = []
        if not clauses:
            # generic guidance (short, deterministic)
            generic = [
                "No clauses were found in the knowledge base for this department/topic.",
                "",
                "Generic checklist improvement guidance:",
                "- Ensure critical high-risk items are present (medication safety, infection control, sepsis, airway).",
                "- Keep items concise, actionable, and ordered by workflow.",
                "- Pilot test with staff and update regularly.",
                "- Use incident history to prioritize changes."
            ]
            return "\n".join(generic)

        # Build deterministic prompt for LLM (but keep it short, temperature=0 in your LLM)
        snippets = []
        for i, r in enumerate(clauses, start=1):
            snippets.append(f"{i}. [{r.get('framework','?')}] {r.get('code','?')} — { (r.get('text') or '')[:250] }")
        prompt = SYNTH_PROMPT_TMPL.format(question=question, clauses_text="\n".join(snippets[:20]))
        resp = qa_llm.invoke(prompt)
        return getattr(resp, "content", str(resp))

    except Exception as e:
        print("[ERROR] synthesize_answer_tool:", e)
        traceback.print_exc()
        return "I’m sorry — unable to synthesize an answer due to internal error."


# ---------------- LangGraph orchestration ----------------
# Tools list & dict (LLM will call by name)
tools = [extract_entities_tool, find_concepts_tool, fetch_clauses_tool,
         compare_frameworks_tool, analyze_audit_gaps_tool, analyze_department_checklist_tool,
         synthesize_answer_tool]

# Bind tools to the orchestrator LLM so it can produce tool calls in its responses
orchestrator_llm = orchestrator_llm.bind_tools(tools)

# Build tools dict for executor
tools_dict = {t.name: t for t in tools}

# AgentState type: messages sequence aggregated via add_messages
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def should_continue(state: AgentState):
    """Continue if the LLM's last message contains tool_calls."""
    last = state['messages'][-1]
    return hasattr(last, "tool_calls") and len(getattr(last, "tool_calls", [])) > 0

# System prompt (your auditor ReACT system prompt)
system_prompt = """
You are an expert Quality Management and Regulatory Compliance Auditor. Your task is to analyze hospital audit data to help auditors improve their checklists and remediate gaps.

IMPORTANT: For questions about improving a department checklist (e.g., "How can I improve my Emergency Department checklist"), prefer the deterministic tool analyze_department_checklist_tool. Do NOT call analyze_audit_gaps_tool or audit-finding tools unless the user explicitly asks to examine past audit findings or incident history.

You have access to tools to:
- extract entities (topics/frameworks/query type),
- find concepts by topic,
- fetch clauses by concept & frameworks,
- compare frameworks,
- analyze audit gaps,
- analyze department checklists,
- synthesize answers.

Follow a ReACT-style loop: think -> call tool(s) -> observe -> think -> final answer.
Be factual, cite clause codes/framework names when present, and do not invent clause codes.
"""

# LLM call node
def call_llm_node(state: AgentState) -> AgentState:
    """Call the LLM with system prompt + current messages. Return state with the LLM message."""
    messages = list(state['messages'])
    # prepend system message
    messages = [SystemMessage(content=system_prompt)] + messages
    message = orchestrator_llm.invoke(messages)
    # return new state with the single LLM message
    return {'messages': [message]}

# Tool executor node
def executor_node(state: AgentState) -> AgentState:
    llm_message = state['messages'][-1]
    tool_calls = getattr(llm_message, "tool_calls", [])
    results = []
    for t in tool_calls:
        name = t['name']
        args = t.get('args', {}) or {}
        print(f"Calling tool: {name} with args: {args}")
        if name not in tools_dict:
            results.append(ToolMessage(tool_call_id=t.get('id'), name=name, content=f"Tool {name} not found."))
            continue
        try:
            tool_obj = tools_dict[name]

            # If the args dict contains one of these obvious single-arg names,
            # pass that value directly (string) — common for 'question','topic','department_name'
            if 'question' in args and len(args) == 1:
                tool_input = args['question']
            elif 'topic' in args and len(args) == 1:
                tool_input = args['topic']
            elif 'department_name' in args and len(args) == 1:
                tool_input = args['department_name']
            elif 'query' in args and len(args) == 1:
                tool_input = args['query']
            elif 'finding_keyword' in args and len(args) == 1:
                tool_input = args['finding_keyword']
            else:
                # Multi-arg case (e.g., concept_element_ids_json + frameworks_json + limit)
                # or generic key/value set: pass a single JSON string as tool_input.
                tool_input = json.dumps(args, ensure_ascii=False)

            # Always call invoke with a single tool_input argument (this works for LangChain BaseTool and our SimpleTool)
            raw_res = tool_obj.invoke(tool_input)
            raw_res = str(raw_res)
        except Exception as e:
            raw_res = f"Tool call error: {e}"
            print("Tool exception:", e)
            traceback.print_exc()

        # sanitize known tool outputs for user readability
        content = sanitize_tool_output(raw_res)
        results.append(ToolMessage(tool_call_id=t.get('id'), name=name, content=content))
    return {'messages': results}

# Build state graph
state_graph = StateGraph(AgentState)
state_graph.add_node("llm", call_llm_node)
state_graph.add_node("executor", executor_node)
state_graph.add_conditional_edges("llm", should_continue, {True: "executor", False: END})
state_graph.add_edge("executor", "llm")
state_graph.set_entry_point("llm")
rag_agent = state_graph.compile()

# ---------------- CLI runner ----------------
def running_agent():
    print("\n=== LangGraph Hospital Audit Agent (RAG) ===")
    while True:
        user_input = input("\nQuestion (or 'exit'): ").strip()
        if user_input.lower() in ["exit", "quit", "q"]:
            break

        # Quick routing: if user asks about a checklist, route directly to the checklist analyzer
        low = user_input.lower()
        if "checklist" in low or "department checklist" in low or "improve my" in low and "checklist" in low:
            # Try to extract department name like "emergency department"
            import re
            m = re.search(r"([A-Za-z\s]+department)", user_input, flags=re.IGNORECASE)
            dept = m.group(1).strip() if m else user_input  # fallback to full input if no dept found
            print(f"Routing directly to analyze_department_checklist_tool for department: {dept}")
            # Call the tool and print markdown (the tool returns JSON with 'markdown')
            raw = tools_dict["analyze_department_checklist_tool"].invoke(dept)
            try:
                out_obj = json.loads(str(raw))
                md = out_obj.get("markdown") or (out_obj.get("analysis") and out_obj.get("analysis").get("markdown")) or sanitize_tool_output(str(raw))
            except Exception:
                md = sanitize_tool_output(str(raw))
            print("\n--- AGENT RESPONSE ---\n")
            print(md)
            print("\n----------------------\n")
            continue

        # Otherwise run full LangGraph agent flow
        messages = [HumanMessage(content=user_input)]
        result = rag_agent.invoke({"messages": messages})
        out_msgs = result.get('messages', [])
        if not out_msgs:
            print("No response from agent.")
            continue
        final = out_msgs[-1]
        print("\n--- AGENT RESPONSE ---\n")
        print(final.content if hasattr(final, "content") else str(final))
        print("\n----------------------\n")

if __name__ == "__main__":
    running_agent()
