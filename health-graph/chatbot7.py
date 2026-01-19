#!/usr/bin/env python3
"""
LangGraph RAG Agent with NL->Cypher micro-agent for Department Checklist Gap Analysis.
Drop-in replacement: set env vars NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, GOOGLE_API_KEY
"""
import os, json, time, traceback, re
from typing import TypedDict, Annotated, Sequence
from operator import add as add_messages
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

# Neo4j adapter (prefers langchain_neo4j if available)
try:
    from langchain_neo4j import Neo4jGraph
except Exception:
    from langchain_community.graphs import Neo4jGraph

from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# ---------------- CONFIG ----------------
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "password")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
FULLTEXT_INDEX_NAME = "ConceptIndex"

# --------- LLMs & Graph connection ----------
orchestrator_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, google_api_key=GOOGLE_API_KEY)
entity_llm = orchestrator_llm
qa_llm = orchestrator_llm

# Neo4j (read-only user recommended)
neo4j_graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USER, password=NEO4J_PASSWORD, sanitize=True)

SYNTH_PROMPT_TMPL = """
You are an expert Quality Management and Regulatory Compliance Auditor. Use the retrieved clauses to provide a concise, factual answer.

Question:
{question}

Clauses:
{clauses_text}

Answer:
"""

# ---------------- Helpers / prompts ----------------
def _run_cypher_debug(cypher: str, params: dict = None):
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
        raise

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
        if not text.startswith("{") and "{" in text:
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

def _is_read_only_cypher(cypher: str) -> (bool, str):
    if not isinstance(cypher, str) or not cypher.strip():
        return False, "Empty Cypher string."
    s = cypher.lower()
    forbidden = [
        r"\bcreate\b", r"\bmerge\b", r"\bdelete\b", r"\bset\b",
        r"\bremove\b", r"\bdrop\b", r"\bcall\s+dbms\b", r"\bcall\s+apoc\b",
        r"\bapoc\.", r"\bcall\s+apoc\.", r"\bload\s+csv\b", r"\bwrite\b",
        r"\bforeach\b"
    ]
    for patt in forbidden:
        if re.search(patt, s):
            return False, f"Disallowed operation detected: {patt}"
    allowed_start_patterns = [
        r"^\s*match\b", r"^\s*optional\s+match\b", r"^\s*unwind\b",
        r"^\s*with\b", r"^\s*return\b", r"^\s*call\s+db\.index\.fulltext\.queryNodes\b",
        r"^\s*call\b"
    ]
    if not any(re.search(p, s) for p in allowed_start_patterns):
        if "return " not in s:
            return False, "Query does not look like a read-only MATCH/RETURN style query."
    return True, ""

def sanitize_tool_output(res_text: str) -> str:
    if not res_text:
        return ""
    try:
        obj = json.loads(res_text)
        if isinstance(obj, dict):
            if "markdown" in obj: return obj["markdown"]
            if "analysis" in obj and isinstance(obj["analysis"], dict) and "markdown" in obj["analysis"]:
                return obj["analysis"]["markdown"]
            if "text" in obj and isinstance(obj["text"], str): return obj["text"]
            return json.dumps(obj, ensure_ascii=False, indent=2)
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
    cleaned = re.sub(r"['\"]?extras['\"]?\s*:\s*\{[^}]+\}", "", res_text, flags=re.DOTALL)
    cleaned = re.sub(r"[A-Za-z0-9+/=]{80,}", "[long_signature_removed]", cleaned)
    return cleaned.strip()

# ---------------- Prompt snippets / templates ----------------
# For the single-template test we include one canonical template description.
SCHEMA_SNIPPET = "Clause(framework,code,text), Department(name), relationships: (Clause)-[:APPLIES_TO]->(Department), (Clause)-[:MENTIONS]->(Concept), (Clause)-[:SIMILAR_TO]-(Clause) with s.score"
NL2CYPHER_PROMPT_TMPL = """
You are a Cypher generation assistant. Input: a template_id and a small context JSON.
Rules:
- Output ONLY JSON with keys: cypher (string), params (object), template_id (string).
- Do NOT produce CREATE/MERGE/DELETE/SET/REMOVE/DROP/LOAD CSV or CALL dbms.* or APOC write calls.
- Use parameterized Cypher ($param) and elementId(node) when appropriate.
- Keep queries read-only and focused.

Template_id: "{template_id}"
Context (JSON): {context_json}
Schema snippet: {schema}

When template_id == "strong_similar_clauses_for_department" produce a Cypher that:
1) MATCHes clauses APPLIES_TO the department,
2) OPTIONAL MATCHes SIMILAR_TO candidates,
3) Filters candidate rows with cand.code IS NOT NULL and NOT (cand)-[:APPLIES_TO]->(d),
4) Returns elementId(linked) AS linked_clause_id, linked.code AS linked_code, linked.framework AS linked_framework,
   elementId(cand) AS cand_clause_id, cand.framework AS cand_framework, cand.code AS cand_code,
   s.score AS similarity, cand.text AS cand_text
5) Accepts params $dept and $limit.

Return JSON now.
"""

# ---------------- Tools ----------------

@tool(description="Generate a read-only, parameterized Cypher query from a template_id and context JSON. Returns JSON with keys: template_id, cypher, params, valid.")
def nl_to_cypher_tool(template_id: str, context_json: str) -> str:
    """
    Micro-agent: generate parameterized read-only Cypher for a known template.
    Returns JSON string: {"template_id":..,"cypher":"...","params":{...},"valid":true/false,"reason":...}
    """
    prompt = NL2CYPHER_PROMPT_TMPL.format(template_id=template_id, context_json=context_json, schema=SCHEMA_SNIPPET)
    parsed = call_and_parse_json(entity_llm, prompt, fallback=None)
    if not parsed or not isinstance(parsed, dict):
        return safe_json_dumps({"template_id": template_id, "cypher": "", "params": {}, "valid": False, "reason": "LLM failed to return JSON."})
    cypher = parsed.get("cypher", "") or ""
    params = parsed.get("params", {}) or {}
    # quick safety check
    safe, reason = _is_read_only_cypher(cypher)
    out = {"template_id": template_id, "cypher": cypher, "params": params, "valid": safe}
    if not safe:
        out["reason"] = reason
    # debug print
    print("\n[NL->CYPHER GENERATED]")
    print("Cypher preview:")
    print(cypher or "[EMPTY]")
    print("Params:", json.dumps(params, ensure_ascii=False))
    print("Valid:", safe, out.get("reason", ""))
    return safe_json_dumps(out)

def execute_cypher_tool(cypher: str, params_json: str = "{}") -> str:
    """
    Validate (read-only) and execute a Cypher query; return rows as JSON.
    """
    try:
        try:
            params = json.loads(params_json) if isinstance(params_json, str) else params_json
        except Exception:
            params = {}
        safe, reason = _is_read_only_cypher(cypher)
        if not safe:
            return safe_json_dumps({"ok": False, "error": f"Query rejected by validator: {reason}"})
        rows = _run_cypher_debug(cypher, params)
        result = rows_to_list(rows)
        return safe_json_dumps({"ok": True, "rows": result, "row_count": len(result)})
    except Exception as e:
        print("[ERROR] execute_cypher_tool:", e)
        traceback.print_exc()
        return safe_json_dumps({"ok": False, "error": str(e)})

@tool(description="Extract topics, frameworks and query type from an auditor question. Returns JSON string with keys: topics, frameworks, query_type.")
def extract_entities_tool(question: str) -> str:
    # Use f-string and escape literal braces with double braces to avoid format/key errors
    prompt = f"""
Extract key information from this hospital audit question.

Question: "{question}"

Return ONLY JSON in this exact format (no extra commentary):

{{
  "topics": ["hand hygiene", "medication safety"],
  "frameworks": ["JCI", "SHCC"],
  "query_type": "comparison"   // one of "comparison" | "search" | "specific"
}}

JSON:
"""
    parsed = call_and_parse_json(entity_llm, prompt, fallback={})
    topics = parsed.get("topics") or []
    frameworks = parsed.get("frameworks") or []
    qtype = parsed.get("query_type") or ("comparison" if "compare" in (question or "").lower() else "search")
    return safe_json_dumps({"topics": topics, "frameworks": frameworks, "query_type": qtype})

@tool(description="Find Concept nodes matching a topic. Uses fulltext index when available; returns JSON list of element_id, name, label, score.")
def find_concepts_tool(topic: str, limit: int = 8) -> str:
    cy = """
    CALL db.index.fulltext.queryNodes($index, $query) YIELD node, score
    RETURN elementId(node) AS element_id, node.name AS name, node.label AS label, score
    ORDER BY score DESC LIMIT $limit
    """
    try:
        rows = _run_cypher_debug(cy, {"index": FULLTEXT_INDEX_NAME, "query": topic, "limit": limit})
        return safe_json_dumps(rows_to_list(rows))
    except Exception:
        # fallback
        cy2 = """
        MATCH (c:Concept)
        WHERE toLower(coalesce(c.name,'')) CONTAINS toLower($q) OR toLower(coalesce(c.label,'')) CONTAINS toLower($q)
        RETURN elementId(c) AS element_id, c.name AS name, c.label AS label LIMIT $limit
        """
        rows = _run_cypher_debug(cy2, {"q": topic, "limit": limit})
        return safe_json_dumps(rows_to_list(rows))

@tool(description="Fetch Clause nodes that MENTION the provided concept element ids. Parameterized by optional framework list; returns JSON list of clauses.")
def fetch_clauses_tool(concept_element_ids_json: str, frameworks_json: str = "[]", limit: int = 200) -> str:
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
    cy = """
    UNWIND $concept_ids AS cid
    MATCH (co) WHERE elementId(co) = cid
    WITH co
    MATCH (cl:Clause)-[:MENTIONS]->(co)
    WHERE ($frameworks_size = 0 OR cl.framework IN $frameworks)
    RETURN DISTINCT cl.code AS code, cl.text AS text, cl.framework AS framework, elementId(co) AS concept_id
    LIMIT $limit
    """
    rows = _run_cypher_debug(cy, {"concept_ids": ids, "frameworks": frameworks, "frameworks_size": len(frameworks), "limit": limit})
    return safe_json_dumps(rows_to_list(rows))

@tool(description="Compare clauses across frameworks for a given concept topic using SIMILAR_TO edges; returns JSON list of paired clauses and similarity scores.")
def compare_frameworks_tool(topic: str) -> str:
    cy = """
    MATCH (c:Concept)
    WHERE toLower(c.name) CONTAINS toLower($topic) OR toLower(c.label) CONTAINS toLower($topic)
    WITH c
    MATCH (j:Clause)-[:MENTIONS]->(c)
    OPTIONAL MATCH (j)-[s:SIMILAR_TO]-(k:Clause)
    WHERE s.score IS NOT NULL
    RETURN c.name AS concept, j.framework AS framework_a, j.code AS code_a, j.text AS text_a,
           k.framework AS framework_b, k.code AS code_b, k.text AS text_b, s.score AS similarity
    ORDER BY s.score DESC LIMIT 50
    """
    try:
        rows = _run_cypher_debug(cy, {"topic": topic})
        return safe_json_dumps(rows_to_list(rows))
    except Exception as e:
        return safe_json_dumps({"error": str(e)})

@tool(description="Find past audit findings containing a keyword and return recent related ClauseInstance / Audit entries; returns JSON list.")
def analyze_audit_gaps_tool(finding_keyword: str) -> str:
    cy = """
    MATCH (f:AuditFinding)
    WHERE toLower(f.text) CONTAINS toLower($keyword)
    MATCH (ci:ClauseInstance)-[:HAS_FINDING]->(f)
    MATCH (ci)-[:PART_OF]->(audit:Audit)
    MATCH (ci)-[:INSTANCE_OF]->(clause:Clause)
    RETURN audit.date AS date, clause.code AS standard_violated, f.text AS finding, f.grade AS severity, f.status AS status
    ORDER BY audit.date DESC LIMIT 20
    """
    try:
        rows = _run_cypher_debug(cy, {"keyword": finding_keyword})
        return safe_json_dumps(rows_to_list(rows))
    except Exception as e:
        return safe_json_dumps({"error": str(e)})

# ---------------- Department checklist analyzer (now delegates Cypher generation) ----------------

def analyze_department_checklist_tool(department_name: str) -> str:
    """
    Delegates Cypher generation to nl_to_cypher_tool using template:
      - strong_similar_clauses_for_department
    Executes generated query via execute_cypher_tool and processes results.
    Returns JSON: {"analysis":..., "markdown": "..."}
    """
    try:
        # Step 0: quick normalization
        dept = department_name.strip()
        context = {"dept": dept, "limit": 200}

        # 1) Ask NL->Cypher micro-agent for the canonical template
        gen_json_str = nl_to_cypher_tool.invoke("strong_similar_clauses_for_department", safe_json_dumps(context))
        # parse
        try:
            gen = json.loads(str(gen_json_str))
        except Exception:
            # fallback: report error and exit gracefully
            print("[ERROR] nl_to_cypher_tool returned invalid JSON:", gen_json_str)
            return safe_json_dumps({"error": "nl_to_cypher_tool failed to return valid JSON."})

        if not gen.get("valid"):
            # fallback: show the rejection reason and run legacy inline cypher as last resort
            reason = gen.get("reason", "unknown")
            print("[WARN] NL->Cypher rejected:", reason)
            # As fallback we could use a safe inline query (same as before); we'll attempt fallback inline
            cy_fallback = """
            MATCH (d:Department)
            WHERE toLower(d.name) CONTAINS toLower($dept)
            WITH d
            MATCH (linked:Clause)-[:APPLIES_TO]->(d)
            OPTIONAL MATCH (linked)-[s:SIMILAR_TO]-(cand:Clause)
            WHERE cand IS NOT NULL AND cand.code IS NOT NULL AND NOT (cand)-[:APPLIES_TO]->(d)
            RETURN elementId(linked) AS linked_clause_id, linked.code AS linked_code, linked.framework AS linked_framework,
                   elementId(cand) AS cand_clause_id, cand.framework AS cand_framework, cand.code AS cand_code,
                   s.score AS similarity, cand.text AS cand_text
            """
            gen = {"cypher": cy_fallback, "params": {"dept": dept, "limit": 200}, "valid": True}

        # 2) Execute generated cypher via execute_cypher_tool
        exec_res_str = execute_cypher_tool.invoke(gen.get("cypher", ""), safe_json_dumps(gen.get("params", {})))
        try:
            exec_res = json.loads(str(exec_res_str))
        except Exception:
            print("[ERROR] execute_cypher_tool returned invalid JSON:", exec_res_str)
            return safe_json_dumps({"error": "execute_cypher_tool failed to return valid JSON."})

        if not exec_res.get("ok"):
            print("[ERROR] execute_cypher_tool error:", exec_res.get("error"))
            return safe_json_dumps({"error": f"Execution failed: {exec_res.get('error')}"})

        rows = exec_res.get("rows", [])
        print(f"[DEBUG] raw rows returned: {len(rows)}")

        # convert and filter (cand_code not null)
        # rows already plain dicts
        strong_list = [r for r in rows if r.get("cand_code")]
        print(f"[DEBUG] strong_filtered count: {len(strong_list)}")
        if strong_list:
            print("[DEBUG] sample strong suggestion:", strong_list[0])

        # compute top-3 by similarity
        def _sim_val(r):
            try:
                return float(r.get("similarity") or 0.0)
            except Exception:
                return 0.0
        strong_sorted = sorted(strong_list, key=_sim_val, reverse=True)
        top_k = 3
        strong_topk = strong_sorted[:top_k]

        # counts
        linked_count = len({ r.get("linked_clause_id") for r in rows })
        strong_count = len(strong_list)
        weak_count = 0  # in this flow we didn't compute weak suggestions (could be added as separate template)

        # gap score (same transparent logic)
        penalty_strong = min(50, int(strong_count * 6))
        penalty_weak = 0
        penalty_total = penalty_strong + penalty_weak
        score = max(0, 100 - penalty_total)

        analysis = {
            "department": dept,
            "linked_clauses_count": linked_count,
            "strong_suggestions_count": strong_count,
            "weak_suggestions_count": weak_count,
            "strong_suggestions_topk": strong_topk,
            "note": ("Strong suggestions found." if strong_count else "No strong suggestions found."),
            "gap_score": score,
            "penalties": {"penalty_strong": penalty_strong, "penalty_weak": penalty_weak, "penalty_total": penalty_total}
        }

        # build markdown (Gap Analysis title)
        md_lines = [f"# Gap Analysis of {dept}", ""]
        md_lines.append(f"**GAP SCORE:** **{score}/100**  ")
        md_lines.append("_Higher is better — score penalizes missing mapped clauses (strong)._\n")
        md_lines.append(f"**Summary:** {analysis['note']}")
        md_lines.append(f"- Clauses currently linked to department (unique linked clause count): **{linked_count}**")
        md_lines.append(f"- Strong suggestions (mapped but missing): **{strong_count}** — showing top {len(strong_topk)} by similarity")
        md_lines.append("")
        if strong_topk:
            md_lines.append("## Top gaps (strong matches not applied to department)")
            for it in strong_topk:
                sim = _sim_val(it)
                md_lines.append(f"- **[{it.get('cand_framework')}] {it.get('cand_code')}**  (similarity={sim:.3f}) — { (it.get('cand_text') or '')[:240] }")
        else:
            md_lines.append("## Top gaps (strong matches): _none found_")

        if strong_count > len(strong_topk):
            md_lines.append("")
            md_lines.append(f"_There are {strong_count} total strong suggestions; only the top {len(strong_topk)} are shown._")
            md_lines.append("Would you like to **see more** suggestions (show top 10)? Reply `yes` to show more.")

        md_lines.append("\n## Recommendations (gaps only)\n- Review the **Top gaps** above and decide whether to add those clauses to the department checklist (APPLIES_TO).\n- Prioritize items by similarity score and operational relevance.\n")
        md = "\n".join(md_lines)
        return safe_json_dumps({"analysis": analysis, "markdown": md})

    except Exception as e:
        print("[ERROR] analyze_department_checklist_tool:", e)
        traceback.print_exc()
        return safe_json_dumps({"error": "Tool execution failed. See logs."})

# ---------------- Synthesize tool (keeps behavior) ----------------
@tool(description="Synthesize a final human-friendly answer or markdown from analysis JSON or clause lists; accepts analysis JSON as positional arg and returns clean markdown.")
def synthesize_answer_tool(question: str = "", clauses_json: str = "[]", qtype: str = "search", analysis_json: str = None) -> str:
    try:
        maybe_json = None
        if question and isinstance(question, str):
            s = question.strip()
            if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
                maybe_json = s
        if analysis_json is None and maybe_json:
            analysis_json = maybe_json
        if analysis_json:
            obj = json.loads(analysis_json)
            if isinstance(obj, dict):
                if "markdown" in obj and isinstance(obj["markdown"], str):
                    return obj["markdown"]
                if "analysis" in obj and isinstance(obj["analysis"], dict) and "note" in obj["analysis"]:
                    md = obj.get("markdown") or ""
                    if md: return md
                    analysis = obj["analysis"]
                    parts = [f"**Summary:** {analysis.get('note','')}"]
                    if analysis.get("strong_suggestions_count", 0):
                        parts.append(f"- Found {analysis['strong_suggestions_count']} strong suggestions.")
                    elif analysis.get("weak_suggestions_count", 0):
                        parts.append(f"- Found {analysis['weak_suggestions_count']} weak suggestions.")
                    else:
                        parts.append("- No data-driven suggestions found.")
                    parts.append("\n**Generic Guidance (if no clauses available):**")
                    parts.append("- Review ED’s patient safety, infection control, equipment readiness, triage, handover & training items.")
                    return "\n".join(parts)
        clauses = []
        try:
            clauses = json.loads(clauses_json)
        except Exception:
            clauses = []
        if not clauses:
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

# ---------------- LangGraph orchestration (unchanged flow) ----------------
tools = [
    extract_entities_tool, find_concepts_tool, fetch_clauses_tool,
    compare_frameworks_tool, analyze_audit_gaps_tool, analyze_department_checklist_tool,
    synthesize_answer_tool, nl_to_cypher_tool, execute_cypher_tool
]

# ---------------- Tool normalization (robust adapter) ----------------
class SimpleTool:
    """Minimal wrapper to give a plain function a .name and .invoke() like LangChain tools."""
    def __init__(self, func, name=None, description=None):
        self._func = func
        self.name = name or getattr(func, "__name__", "unnamed_tool")
        self.description = description or getattr(func, "__doc__", "") or ""
    def invoke(self, *args, **kwargs):
        # maintain original behavior (many tools expect positional args)
        return self._func(*args, **kwargs)
    def __repr__(self):
        return f"<SimpleTool name={self.name}>"

# Build a normalized list of tool-like objects:
normalized_tools = []
for t in tools:
    # If already a LangChain Tool-like object, keep as-is
    if hasattr(t, "name") and callable(getattr(t, "invoke", None)):
        normalized_tools.append(t)
    # If it's a plain function, wrap it
    elif callable(t):
        normalized_tools.append(SimpleTool(t, name=getattr(t, "__name__", None)))
    else:
        raise TypeError(f"Unsupported tool type in tools list: {type(t)}")

# Build tools_dict mapping names -> tool-like objects for executor_node
tools_dict = {t.name: t for t in normalized_tools}

# For LLM binding: only pass real LangChain Tool objects to bind_tools (if you want the LLM to emit tool_calls)
langchain_tools_for_binding = [t for t in normalized_tools if hasattr(t, "_func") is False]
# If there are no LangChain Tool objects (likely), skip binding to avoid errors.
if langchain_tools_for_binding:
    try:
        orchestrator_llm = orchestrator_llm.bind_tools(langchain_tools_for_binding)
    except Exception as e:
        print("[WARN] bind_tools failed with LangChain tool objects:", e)
        # continue without binding; LLM will not auto-emit tool_calls, but executor_node still works.

# (re)bind tools to LLM so it can emit tool calls if needed
# orchestrator_llm = orchestrator_llm.bind_tools(tools)
# tools_dict = {t.name: t for t in tools}

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def should_continue(state: AgentState):
    last = state['messages'][-1]
    return hasattr(last, "tool_calls") and len(getattr(last, "tool_calls", [])) > 0

system_prompt = """
You are an expert Quality Management and Regulatory Compliance Auditor. Your task is to analyze hospital audit data to help auditors improve their checklists and remediate gaps.

You have access to tools to:
- extract entities, find concepts, fetch clauses, compare frameworks, analyze audit gaps, analyze department checklists (delegates Cypher generation when needed), and synthesize answers.

Follow a ReACT-style loop: think -> call tool(s) -> observe -> think -> final answer.
Be factual, cite clause codes/framework names when present, and do not invent clause codes.
"""

def call_llm_node(state: AgentState) -> AgentState:
    messages = list(state['messages'])  # these may be ToolMessages returned by executor_node
    # Prepend system prompt
    messages = [SystemMessage(content=system_prompt)] + messages

    # Ensure there is at least one non-system message with non-empty content
    has_non_system = False
    for m in messages:
        # we accept HumanMessage, AIMessage, ToolMessage as non-system, but require non-empty content
        content = getattr(m, "content", None)
        if content and isinstance(content, str) and content.strip():
            # consider valid non-system content present (note: SystemMessage is earlier)
            has_non_system = True
            break

    if not has_non_system:
        # find last ToolMessage content if available
        last_tool_text = None
        for m in reversed(messages):
            if m.__class__.__name__ == "ToolMessage" and getattr(m, "content", None):
                last_tool_text = m.content
                break
        if not last_tool_text:
            last_tool_text = "No observation available from tools. Please proceed with the next step."
        # Add as a HumanMessage so Gemini has non-empty content
        messages.append(HumanMessage(content=str(last_tool_text)[:4000]))

    # Call LLM
    message = orchestrator_llm.invoke(messages)
    return {'messages': [message]}

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
            # Preferred single-arg names
            if 'department_name' in args:
                raw_res = tool_obj.invoke(args['department_name'])
            elif 'topic' in args:
                raw_res = tool_obj.invoke(args['topic'])
            elif 'question' in args:
                raw_res = tool_obj.invoke(args['question'])
            elif 'query' in args:
                raw_res = tool_obj.invoke(args['query'])
            elif 'finding_keyword' in args:
                raw_res = tool_obj.invoke(args['finding_keyword'])
            elif 'concept_element_ids_json' in args:
                # common signature (concepts_json, frameworks_json?, limit?)
                ce = args.get('concept_element_ids_json')
                fw = args.get('frameworks_json', "[]")
                lim = args.get('limit', 200)
                raw_res = tool_obj.invoke(ce, fw, lim)
            elif 'params_json' in args and 'cypher' in args:
                # for NL->Cypher run pattern (if you choose to pass both)
                raw_res = tool_obj.invoke(args['cypher'], args.get('params_json', "{}"))
            else:
                # fallback heuristics:
                if len(args) == 1:
                    # pass single value
                    raw_res = tool_obj.invoke(next(iter(args.values())))
                elif len(args) == 2:
                    raw_res = tool_obj.invoke(*list(args.values()))
                elif len(args) == 3:
                    raw_res = tool_obj.invoke(*list(args.values()))
                else:
                    # as last resort, send whole args dict (function should accept a json string)
                    try:
                        raw_res = tool_obj.invoke(json.dumps(args))
                    except Exception:
                        raw_res = tool_obj.invoke(str(args))
            raw_res = str(raw_res)
        except Exception as e:
            raw_res = f"Tool call error: {e}"
            print("Tool exception:", e)
            traceback.print_exc()

        # sanitize known tool outputs for user readability
        content = sanitize_tool_output(raw_res)
        results.append(ToolMessage(tool_call_id=t.get('id'), name=name, content=content))
    return {'messages': results}

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
