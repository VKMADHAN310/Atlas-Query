from __future__ import annotations

import os
import re
import json
import time
import hashlib
import datetime as dt
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
import requests
import psycopg2
import sqlparse
from pydantic import BaseModel, Field, ValidationError

# Optional HF provider import (lazy used)
try:
    from llama import call_hf_chat
except Exception:
    call_hf_chat = None  # type: ignore


# Configuration (environment variables may override these)
load_dotenv()
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi3:mini")

PGDATABASE = os.getenv("PGDATABASE", "USCountyDB")
PGHOST = os.getenv("PGHOST", "localhost")
PGPORT = int(os.getenv("PGPORT", "5432"))
PGUSER = os.getenv("PGUSER", "postgres")
PGPASSWORD = os.getenv("PGPASSWORD")  # use .pgpass, env, or peer auth

LOG_DIR = os.getenv("NL2SQL_LOG_DIR", "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, "m2_runs.jsonl")

ALLOWED_TABLE = "counties"

# Schema description shown to the model inside the system prompt
SCHEMA_TEXT = (
    "counties(geoid text unique, name text, namelsad text, state text, "
    "stateaabrv text, aland bigint, awater bigint, "
    "geom geometry(MULTIPOLYGON,4269))"
)

# Early filter for out-of-scope prompts (non US regions, provinces, etc.)
OOS_PATTERNS = [
    r"\bcanada\b",
    r"\bprovince\b",
    r"\bprovinces\b",
    r"\bengland\b|\buk\b|\bscotland\b|\bwales\b|\bnorthern ireland\b",
    r"\bmexico\b",
    r"\bindia\b",
    r"\bprovince of\b",
]


# Data models
@dataclass
class LogEntry:
    """A single JSONL log record for each user query and its processing path."""
    timestamp: str
    nl_query: str
    lm_model: str
    system_prompt_hash: str
    lm_duration_ms: int
    lm_raw_response: str
    sql_candidate: str
    sql_valid: bool
    sql_final: Optional[str]
    sql_rewrite_reason: Optional[str]
    db_duration_ms: Optional[int]
    row_count: Optional[int]
    errors: List[str]


class Answer(BaseModel):
    """Compact response object returned to the frontend."""
    ok: bool
    error: Optional[str] = None
    rows_preview: List[Dict[str, Any]] = Field(default_factory=list)
    rows_total: int = 0
    sql: Optional[str] = None
    lm_ms: int = 0
    db_ms: int = 0
    scope_rejected: bool = False


def answer_for_rows(rows: List[Dict[str, Any]], sql: str, db_ms: int) -> Answer:
    """Helper to build a standard Answer for direct DB queries (no LM)."""
    has_geojson = bool(rows) and isinstance(rows[0], dict) and ("geojson" in rows[0])
    preview = rows if (has_geojson and len(rows) <= 500) else rows[:20]
    return Answer(
        ok=True,
        rows_preview=preview,
        rows_total=len(rows),
        sql=sql,
        lm_ms=0,
        db_ms=db_ms,
    )


# Helpers
def now_iso() -> str:
    """UTC timestamp in ISO-8601 with millisecond precision (for logs)."""
    # Use timezone-aware UTC to avoid deprecation of datetime.utcnow()
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def sha1(s: str) -> str:
    """Stable SHA-1 hash (used to identify the exact system prompt used)."""
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def is_out_of_scope(nlq: str) -> bool:
    """Heuristically reject prompts that clearly ask about non-US geographies."""
    q = nlq.lower()
    return any(re.search(p, q) for p in OOS_PATTERNS)


def call_ollama(system: str, prompt: str, model: Optional[str] = None) -> Tuple[str, int]:
    """Call a local Ollama server and return (model_text, duration_ms)."""
    payload = {
        "model": model or OLLAMA_MODEL,
        "system": system,
        "prompt": prompt,
        "options": {"temperature": 0, "num_predict": 256},
        "stream": False,
    }
    t0 = time.perf_counter_ns()
    r = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=180)
    t1 = time.perf_counter_ns()
    r.raise_for_status()
    data = r.json()
    txt = data.get("response", "")
    return txt, int((t1 - t0) / 1e6)


def call_lm(system: str, prompt: str, *, model: Optional[str] = None, provider: Optional[str] = None) -> Tuple[str, int, str]:
    """
    Provider-agnostic LM call. Returns (text, ms, effective_model).
    provider: 'ollama' (default) or 'hf'. If model startswith 'hf:', route to HF.
    """
    eff_model = model or OLLAMA_MODEL
    # Routing by explicit provider or model prefix
    use_hf = (provider == "hf") or (isinstance(model, str) and model.lower().startswith("hf:"))
    if use_hf:
        if not call_hf_chat:
            raise RuntimeError("HF provider not available (missing llama.call_hf_chat import)")
        # Choose HF model: explicit model (strip hf:), else env HF_MODEL, else sensible default
        import os as _os
        if model:
            hf_model = eff_model[3:] if eff_model.lower().startswith("hf:") else eff_model
        else:
            hf_model = _os.getenv("HF_MODEL", "meta-llama/Llama-3.3-70B-Instruct:cerebras")
        txt, ms = call_hf_chat(system, prompt, model=hf_model)
        return txt, ms, hf_model
    # Default to Ollama
    txt, ms = call_ollama(system, prompt, model=eff_model)
    return txt, ms, eff_model


def strip_code_fences(s: str) -> str:
    """Remove ```sql fences LMs sometimes include around SQL."""
    s = s.strip()
    s = re.sub(r"^```(?:sql)?", "", s, flags=re.IGNORECASE).strip()
    s = re.sub(r"```$", "", s).strip()
    return s


def strip_sql_comments(s: str) -> str:
    """Remove SQL line comments (--) and block comments (/* ... */)."""
    s = re.sub(r"/\*[^*]*\*+(?:[^/*][^*]*\*+)*/", " ", s, flags=re.DOTALL)
    s = re.sub(r"--[^\n]*", " ", s)
    return s


def should_include_geom(nlq: str) -> bool:
    """
    Detect visualization intent; tells us to return geometry for mapping.
    Include geometry for any query that asks for county information (list, show, find, etc.)
    unless it's explicitly a count-only query.
    """
    q = nlq.lower()
    
    # Explicit visualization keywords
    visualization_keys = ["visualize", "visualisation", "visualization", "map", "geometry", "geojson", "polygon", "shape"]
    if any(k in q for k in visualization_keys):
        return True
    
    # Queries that ask to list/show/find/display counties should include geometry
    list_keywords = ["list", "show", "find", "display", "get", "give", "return", "what are", "which are", "tell me"]
    if any(k in q for k in list_keywords):
        # But exclude pure count queries
        if re.search(r"\b(count|how many|number of)\b", q) and not re.search(r"\b(list|show|find|display|get|give|return|what are|which are|tell me)\b", q):
            return False
        return True
    
    # Queries asking "what" or "which" about counties should include geometry
    if re.search(r"\b(what|which|where)\b.*count", q) and not re.search(r"\b(count|how many|number of)\b", q):
        return True
    
    return False


def is_spatial_query(nlq: str) -> bool:
    """Detect spatial relationship queries that should be visualized on the map."""
    q = nlq.lower()
    spatial_keywords = [
        "neighbor", "neighbors", "neighbour", "neighbours",
        "touching", "touch", "adjacent", "bordering", "border",
        "touches", "intersects", "intersect", "contains", "contain",
        "surrounding", "surround", "nearby", "near"
    ]
    return any(k in q for k in spatial_keywords)


def add_geom_to_spatial_count_query(sql: str, nl_query: str) -> Tuple[str, bool]:
    """
    For spatial COUNT queries (e.g., "How many neighbors does X have?"),
    modify the query to return the actual counties involved with geometry
    so they can be visualized on the map.
    """
    # Check if this is a COUNT query with spatial join
    if not re.search(r"\bCOUNT\s*\(", sql, flags=re.IGNORECASE):
        return sql, False
    
    if not re.search(r"\b(ST_Touches|ST_Intersects|ST_Contains|JOIN.*ON.*geom)", sql, flags=re.IGNORECASE):
        return sql, False
    
    # Pattern: SELECT COUNT(...) FROM counties a JOIN counties b ON ST_Touches(a.geom, b.geom) WHERE ...
    # Extract the join condition and WHERE clause
    # Match JOIN ... ON ... but stop before WHERE if present
    join_match = re.search(r"JOIN\s+counties\s+(\w+)\s+ON\s+((?:(?!\s+WHERE).)+?)(?:\s+WHERE|\s*$)", sql, flags=re.IGNORECASE | re.DOTALL)
    if not join_match:
        return sql, False
    
    alias_b = join_match.group(1)
    join_condition = join_match.group(2).strip()
    
    # Extract FROM clause alias
    from_match = re.search(r"FROM\s+counties\s+(\w+)", sql, flags=re.IGNORECASE)
    if not from_match:
        return sql, False
    
    alias_a = from_match.group(1)
    
    # Extract WHERE clause
    where_match = re.search(r"WHERE\s+(.+?)(?:\s+GROUP\s+BY|\s+ORDER\s+BY|\s*$)", sql, flags=re.IGNORECASE | re.DOTALL)
    where_clause = where_match.group(1).strip() if where_match else ""
    
    # Build new query: return both the source county and its neighbors with geometry
    if where_clause:
        new_sql = f"""SELECT {alias_a}.namelsad, {alias_a}.stateaabrv, {alias_a}.geoid, {alias_a}.geom, 'source' AS county_type
FROM counties {alias_a}
WHERE {where_clause}
UNION ALL
SELECT {alias_b}.namelsad, {alias_b}.stateaabrv, {alias_b}.geoid, {alias_b}.geom, 'neighbor' AS county_type
FROM counties {alias_a} JOIN counties {alias_b} ON {join_condition}
WHERE {where_clause}
GROUP BY {alias_b}.geoid, {alias_b}.namelsad, {alias_b}.stateaabrv, {alias_b}.geom"""
    else:
        # No WHERE clause - return all neighbors
        new_sql = f"""SELECT {alias_b}.namelsad, {alias_b}.stateaabrv, {alias_b}.geoid, {alias_b}.geom, 'neighbor' AS county_type
FROM counties {alias_a} JOIN counties {alias_b} ON {join_condition}
GROUP BY {alias_b}.geoid, {alias_b}.namelsad, {alias_b}.stateaabrv, {alias_b}.geom"""
    
    return new_sql, True


def convert_simple_count_to_rows(sql: str) -> Tuple[str, bool]:
    """
    Convert a simple COUNT(*) query over counties into a row-level selection
    that returns county attributes and geometry so results can be visualized.

    Examples handled:
    - SELECT COUNT(*) FROM counties WHERE ...
    - SELECT COUNT(*) AS n FROM counties WHERE ...
    - SELECT COUNT(DISTINCT geoid) FROM counties WHERE ...

    We purposely ignore GROUP BY here; if present, we do not rewrite.
    """
    # Must be a COUNT query without GROUP BY
    if not re.search(r"\bCOUNT\s*\(", sql, flags=re.IGNORECASE):
        return sql, False
    if re.search(r"\bGROUP\s+BY\b", sql, flags=re.IGNORECASE):
        return sql, False

    # Capture optional alias after FROM counties
    m_from = re.search(r"(?is)\bFROM\s+counties\s*([a-zA-Z_][a-zA-Z0-9_]*)?", sql)
    if not m_from:
        return sql, False

    alias = m_from.group(1)
    alias_prefix = f"{alias}." if alias else ""

    # Extract WHERE clause (up to GROUP BY/ORDER BY/LIMIT or end)
    m_where = re.search(r"(?is)\bWHERE\s+(.+?)(\bGROUP\s+BY\b|\bORDER\s+BY\b|\bLIMIT\b|$)", sql)
    where_clause = m_where.group(1).strip() if m_where else ""

    # Build a row-level select with geom for visualization
    select_cols = f"{alias_prefix}namelsad, {alias_prefix}stateaabrv, {alias_prefix}geoid, {alias_prefix}geom"
    new_sql = f"SELECT {select_cols} FROM counties"
    if alias:
        new_sql += f" {alias}"
    if where_clause:
        new_sql += f" WHERE {where_clause}"

    # Ensure a safe LIMIT to bound payload (added only if absent)
    if not re.search(r"\bLIMIT\s+\d+\b", new_sql, flags=re.IGNORECASE):
        new_sql += " LIMIT 500"

    return new_sql, True


def add_geom_to_select_if_needed(sql: str) -> Tuple[str, bool]:
    """
    Append geom to the SELECT list if it is not present.
    Skip aggregate queries and queries that already select geometry.
    For JOIN queries, qualifies geom with the appropriate alias.
    """
    if re.search(r"\b(COUNT|SUM|AVG|MIN|MAX)\s*\(|\bGROUP\s+BY\b", sql, flags=re.IGNORECASE):
        return sql, False

    m = re.search(r"(?is)^\s*SELECT\s+(.*?)\s+FROM\s", sql)
    if not m:
        return sql, False

    select_clause = m.group(1)

    # If geometry (or a prior geojson cast) is already present, do nothing.
    if re.search(
        r"\b((?:[A-Za-z_][A-Za-z0-9_]*\.)?geom|geojson|ST_AsGeoJSON\s*\(\s*geom\s*\))\b",
        select_clause,
        flags=re.IGNORECASE,
    ):
        return sql, False

    # Determine which alias to use for geom
    # For JOIN queries, use the first alias from FROM clause
    geom_col = "geom"
    # Match "FROM counties alias" where alias is followed by JOIN, WHERE, ORDER, etc.
    # Only match if there's actually an alias (not a SQL keyword)
    from_match = re.search(r"FROM\s+counties\s+([a-zA-Z_][a-zA-Z0-9_]*)(?:\s+(?:JOIN|WHERE|ORDER|GROUP|HAVING|LIMIT|UNION))", sql, flags=re.IGNORECASE)
    if from_match:
        alias = from_match.group(1)
        # Double-check it's not a keyword (shouldn't happen with the lookahead, but be safe)
        if alias.upper() not in ('WHERE', 'ORDER', 'GROUP', 'HAVING', 'LIMIT', 'UNION', 'INTERSECT', 'EXCEPT', 'JOIN'):
            geom_col = f"{alias}.geom"
    # If no alias found, use unqualified geom (which is fine for single-table queries)

    new_select = select_clause.strip() + f", {geom_col}"
    start, end = m.span(1)
    new_sql = sql[:start] + new_select + sql[end:]
    return new_sql, True


def fix_distinct_with_json(sql: str) -> Tuple[str, bool]:
    """
    Fix DISTINCT usage when JSON columns are present.
    PostgreSQL cannot use DISTINCT directly on JSON types.
    Replace DISTINCT with GROUP BY on non-JSON columns, or remove DISTINCT if not needed.
    """
    # Check if query has DISTINCT and JSON columns
    if not re.search(r"\bSELECT\s+DISTINCT\b", sql, flags=re.IGNORECASE):
        return sql, False
    
    if not re.search(r"ST_AsGeoJSON.*::json|geojson", sql, flags=re.IGNORECASE):
        return sql, False
    
    # Extract SELECT clause
    m = re.search(r"(?is)^\s*SELECT\s+DISTINCT\s+(.*?)\s+FROM\s", sql)
    if not m:
        return sql, False
    
    select_clause = m.group(1)
    
    # Find non-JSON columns to use in GROUP BY
    # Simple approach: split by comma and extract first identifier from each part
    columns = []
    parts = select_clause.split(',')
    for part in parts:
        part = part.strip()
        # Skip if this part contains JSON functions or geojson alias
        if re.search(r"ST_AsGeoJSON|geojson", part, re.IGNORECASE):
            continue
        # Extract column name (first identifier, possibly qualified)
        col_match = re.search(r"([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)?)", part, re.IGNORECASE)
        if col_match:
            columns.append(col_match.group(1))
    
    if columns:
        # Use DISTINCT ON instead of GROUP BY for JSON columns
        # DISTINCT ON works with any column types including JSON
        # Use the first non-JSON column for DISTINCT ON
        distinct_col = columns[0]
        new_sql = sql.replace("SELECT DISTINCT", f"SELECT DISTINCT ON ({distinct_col})", 1)
        return new_sql, True
    
    # If no suitable columns found, just remove DISTINCT (less safe but might work)
    new_sql = sql.replace("SELECT DISTINCT", "SELECT", 1)
    return new_sql, True


def fix_round_numeric_cast(sql: str) -> Tuple[str, bool]:
    """
    Ensure ROUND(expr, n) or ROUND(expr) casts the expression to numeric
    for PostgreSQL. If only one argument is provided, use two digits.
    Also fixes malformed ROUND expressions with incorrect cast syntax.
    """
    rewrote_any = False

    # Fix malformed patterns like ROUND(expr::numeric, 2)/divisor
    # This should be ROUND((expr/divisor)::numeric, 2)
    malformed_div_pattern = re.compile(
        r"ROUND\s*\(\s*([^,\)]+?)\s*::\s*numeric\s*,\s*(\d+)\s*\)\s*/\s*([0-9.e+-]+)",
        re.IGNORECASE
    )
    def _fix_div_malformed(m: re.Match) -> str:
        nonlocal rewrote_any
        expr = m.group(1).strip()
        digits = m.group(2)
        divisor = m.group(3).strip()
        rewrote_any = True
        return f"ROUND(({expr}/{divisor})::numeric, {digits})"
    
    sql = malformed_div_pattern.sub(_fix_div_malformed, sql)

    # Fix malformed patterns like ROUND(((expr::numeric, 2)/divisor)
    malformed_nested_pattern = re.compile(
        r"ROUND\s*\(\s*\(\s*\(\s*([^,\)]+?)\s*::\s*numeric\s*,\s*\d+\s*\)\s*([^)]*)\)",
        re.IGNORECASE
    )
    def _fix_nested_malformed(m: re.Match) -> str:
        nonlocal rewrote_any
        expr = m.group(1).strip()
        remainder = m.group(2).strip()
        rewrote_any = True
        # Extract digits if present in remainder, otherwise use 2
        digits_match = re.search(r",\s*(\d+)", m.group(0))
        digits = digits_match.group(1) if digits_match else "2"
        return f"ROUND(({expr}{remainder})::numeric, {digits})"
    
    sql = malformed_nested_pattern.sub(_fix_nested_malformed, sql)

    # Pattern to match ROUND(expr, digits) where expr may contain nested parentheses
    # Use a more sophisticated approach: find ROUND( and match balanced parentheses
    def _fix_round_expr(sql_str: str) -> Tuple[str, bool]:
        changed = False
        # Normalize whitespace first
        normalized = re.sub(r'\s+', ' ', sql_str)
        result = []
        i = 0
        while i < len(normalized):
            # Look for ROUND(
            if normalized[i:].upper().startswith("ROUND("):
                # Find the matching closing parenthesis
                depth = 0
                start = i
                i += 6  # Skip "ROUND("
                expr_start = i
                while i < len(normalized):
                    if normalized[i] == '(':
                        depth += 1
                    elif normalized[i] == ')':
                        if depth == 0:
                            # Found the closing paren for ROUND
                            break
                        depth -= 1
                    i += 1
                
                if i < len(normalized):
                    # Extract the content between ROUND( and )
                    content = normalized[expr_start:i].strip()
                    # Check if it contains a comma (two-arg ROUND)
                    # Find the last comma that's not inside parentheses
                    comma_pos = -1
                    depth = 0
                    for j in range(len(content) - 1, -1, -1):
                        if content[j] == ')':
                            depth += 1
                        elif content[j] == '(':
                            depth -= 1
                        elif content[j] == ',' and depth == 0:
                            comma_pos = j
                            break
                    
                    if comma_pos > 0:
                        # Two-argument ROUND
                        expr = content[:comma_pos].strip()
                        digits = content[comma_pos+1:].strip()
                        # Check if already has numeric cast
                        if not re.search(r"::\s*numeric\b", expr, flags=re.IGNORECASE):
                            changed = True
                            result.append(f"ROUND(({expr})::numeric, {digits})")
                        else:
                            result.append(normalized[start:i+1])
                    else:
                        # Single-argument ROUND
                        expr = content.strip()
                        if not re.search(r"::\s*numeric\b", expr, flags=re.IGNORECASE):
                            changed = True
                            result.append(f"ROUND(({expr})::numeric, 2)")
                        else:
                            result.append(f"ROUND(({expr}), 2)")
                    i += 1
                else:
                    result.append(normalized[start])
                    i += 1
            else:
                result.append(normalized[i])
                i += 1
        return ''.join(result), changed
    
    sql, changed = _fix_round_expr(sql)
    rewrote_any = rewrote_any or changed
    return sql, rewrote_any


def fix_common_function_typos(sql: str) -> Tuple[str, bool]:
    """Correct common PostGIS misspellings the LM might emit (e.g., ST_Aarea → ST_Area)."""
    new_sql = re.sub(r"\bST_Aarea\b", "ST_Area", sql, flags=re.IGNORECASE)
    return new_sql, new_sql != sql


def qualify_ambiguous_columns(sql: str) -> Tuple[str, bool]:
    """
    When self joining the counties table, qualify unqualified columns
    such as namelsad, name, geoid, state, stateaabrv with alias a.
    Also fix ORDER BY namelsad when b.namelsad is the selected value.
    """
    if not re.search(r"\bJOIN\s+counties\b", sql, flags=re.IGNORECASE):
        return sql, False

    # Only attempt this normalization when using aliases a and b; otherwise skip to avoid corrupting c1/c2, etc.
    if not (re.search(r"(?i)\bfrom\s+counties\s+a\b", sql) and re.search(r"(?i)\bjoin\s+counties\s+b\b", sql)):
        return sql, False

    rewrote = False

    # SELECT list: prefix ambiguous bare columns with alias a.
    m_sel = re.search(r"(?is)^\s*SELECT\s+(.*?)\s+FROM\s", sql)
    if m_sel:
        sel = m_sel.group(1)
        ambiguous_cols = ["namelsad", "name", "geoid", "state", "stateaabrv"]
        sel_new = sel
        for col in ambiguous_cols:
            sel_new = re.sub(rf"(?i)(?<![a-z_\.])\b{col}\b(?!\s*\()", fr"a.{col}", sel_new)
        if sel_new != sel:
            rewrote = True
            start, end = m_sel.span(1)
            sql = sql[:start] + sel_new + sql[end:]

    # GROUP BY: keep qualifiers consistent with SELECT rewrite (avoid double-qualifying).
    m_gb = re.search(r"(?is)\bGROUP\s+BY\s+(.+?)(\bORDER\s+BY\b|\bLIMIT\b|$)", sql)
    if m_gb:
        gb = m_gb.group(1)
        gb_new = gb
        for col in ["namelsad", "name", "geoid", "state", "stateaabrv"]:
            # Do not match if already qualified like a.col or b.col
            gb_new = re.sub(rf"(?i)(?<![a-z_]\.)\b{col}\b", fr"a.{col}", gb_new)
        if gb_new != gb:
            rewrote = True
            start, end = m_gb.span(1)
            sql = sql[:start] + gb_new + sql[end:]

    # ORDER BY: prefer a.namelsad unless we explicitly selected b.namelsad (avoid double-qualifying).
    m_ob = re.search(r"(?is)\bORDER\s+BY\s+(.+?)(\bLIMIT\b|$)", sql)
    if m_ob:
        ob = m_ob.group(1)
        if re.search(r"(?i)\bSELECT\s+DISTINCT\s+b\.namelsad\b", sql):
            ob_new = re.sub(r"(?i)(?<![a-z_]\.)\bnamelsad\b", "b.namelsad", ob)
        else:
            ob_new = re.sub(r"(?i)(?<![a-z_]\.)\bnamelsad\b", "a.namelsad", ob)
        for col in ["name", "geoid", "state", "stateaabrv"]:
            ob_new = re.sub(rf"(?i)(?<![a-z_]\.)\b{col}\b", fr"a.{col}", ob_new)
        if ob_new != ob:
            rewrote = True
            start, end = m_ob.span(1)
            sql = sql[:start] + ob_new + sql[end:]

    return sql, rewrote


def sanitize_sql(sql: str) -> str:
    """
    Enforce a single, safe SELECT that reads from counties.
    - Keeps only the first statement; trims semicolons.
    - Forbids DDL/DML; strips comments and smart quotes.
    - Normalizes schema prefix; prefers stateaabrv for 2-letter states.
    - Adds LIMIT to non-aggregates if absent (to protect UI).
    """
    sql = strip_code_fences(sql).strip()
    # Normalize “smart quotes” to ASCII quotes before parsing.
    sql = sql.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    sql = strip_sql_comments(sql).strip()

    # Split on semicolons and take only the first statement.
    parts = sqlparse.split(sql)
    if not parts:
        raise ValueError("Invalid or empty SQL.")
    sql = parts[0].strip()

    sql = sql.rstrip(";").strip()

    parsed = sqlparse.parse(sql)
    if not parsed or len(parsed) != 1:
        raise ValueError("Multiple statements are not allowed.")

    stmt = parsed[0]
    # Allow SELECT statements, including UNION ALL queries
    if not stmt.tokens or stmt.token_first(skip_cm=True).normalized.upper() not in ("SELECT", "WITH"):
        raise ValueError("Only SELECT statements are allowed.")

    # Guardrails against obvious destructive or privileged commands.
    forbidden = r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE|CREATE|GRANT|REVOKE|COPY|EXECUTE|VACUUM)\b"
    if re.search(forbidden, sql, flags=re.IGNORECASE):
        raise ValueError("Forbidden SQL keyword detected.")

    if not re.search(r"\bFROM\s+([A-Za-z0-9_\.]+)", sql, flags=re.IGNORECASE):
        raise ValueError("Missing FROM clause.")

    # Normalize qualified table to bare 'counties'.
    sql = re.sub(r"\b(?:USCountyDB\.|public\.)counties\b", "counties", sql, flags=re.IGNORECASE)

    # Prefer the 2-letter state code column when users write UPPER(state)='XX' or state='XX'.
    sql = re.sub(r"\bUPPER\s*\(\s*state\s*\)\s*=\s*'([A-Z]{2})'", r"stateaabrv='\1'", sql, flags=re.IGNORECASE)
    sql = re.sub(r"\bstate\s*=\s*'([A-Z]{2})'", r"stateaabrv='\1'", sql, flags=re.IGNORECASE)

    # Ensure the query ultimately reads from counties (directly or via JOIN).
    reads_counties = re.search(r"\bFROM\s+counties\b", sql, flags=re.IGNORECASE) or re.search(
        r"\bJOIN\s+counties\b", sql, flags=re.IGNORECASE
    )
    if not reads_counties:
        raise ValueError("Query must ultimately read from table 'counties'.")

    # Add a LIMIT for non-aggregate SELECTs to keep payloads/UI manageable.
    if not re.search(r"\bSELECT\s+COUNT\(|\bGROUP\s+BY\b", sql, flags=re.IGNORECASE):
        if not re.search(r"\bLIMIT\s+\d+\b", sql, flags=re.IGNORECASE):
            sql += " LIMIT 500"

    return sql


def ensure_geojson_projection(sql: str) -> Tuple[str, bool]:
    """
    If the SELECT list contains geom, replace just that target with
    ST_AsGeoJSON(geom)::json AS geojson. Predicates referencing geom
    (e.g., where/join) are untouched.
    Handles UNION ALL queries by processing each SELECT separately.
    """
    # Check if this is a UNION ALL query
    if "UNION ALL" in sql.upper():
        parts = re.split(r"\s+UNION\s+ALL\s+", sql, flags=re.IGNORECASE)
        if len(parts) == 2:
            part1, part2 = parts[0], parts[1]
            part1_converted, changed1 = _convert_geom_to_geojson_in_select(part1)
            part2_converted, changed2 = _convert_geom_to_geojson_in_select(part2)
            if changed1 or changed2:
                return f"{part1_converted} UNION ALL {part2_converted}", True
        return sql, False
    
    return _convert_geom_to_geojson_in_select(sql)


def _convert_geom_to_geojson_in_select(sql: str) -> Tuple[str, bool]:
    """Helper function to convert geom to GeoJSON in a single SELECT statement."""
    m = re.search(r"(?is)^\s*SELECT\s+(.*?)\s+FROM\s", sql)
    if not m:
        return sql, False

    select_clause = m.group(1)
    # Match 'geom' only when it appears as a SELECT output column (not in functions or predicates).
    # Handle both qualified (a.geom) and unqualified (geom) references
    pattern = re.compile(
        r"(^|,\s*)((?:[A-Za-z_][A-Za-z0-9_]*\.)?geom)(\s+AS\s+\w+)?(?=\s*(,|$))",
        re.IGNORECASE,
    )

    def _repl(mm: re.Match) -> str:
        prefix = mm.group(1) or ""
        geom_ref = mm.group(2)  # This includes the alias if present (e.g., "a.geom" or "geom")
        return f"{prefix}ST_AsGeoJSON({geom_ref})::json AS geojson"

    new_select = pattern.sub(_repl, select_clause)
    if new_select == select_clause:
        return sql, False

    start, end = m.span(1)
    return sql[:start] + new_select + sql[end:], True


def pg_connect():
    """Open a new psycopg2 connection using environment configuration."""
    return psycopg2.connect(
        host=PGHOST, port=PGPORT, dbname=PGDATABASE, user=PGUSER, password=PGPASSWORD
    )


def run_sql(sql: str) -> Tuple[List[Dict[str, Any]], int]:
    """Execute a SELECT and return (rows_as_dicts, db_duration_ms)."""
    t0 = time.perf_counter_ns()
    with pg_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            cols = [desc.name for desc in cur.description]
            rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    t1 = time.perf_counter_ns()
    return rows, int((t1 - t0) / 1e6)


def run_sql_params(sql: str, params: Tuple[Any, ...]) -> Tuple[List[Dict[str, Any]], int]:
    """Execute a parameterized SELECT and return (rows_as_dicts, db_duration_ms)."""
    t0 = time.perf_counter_ns()
    with pg_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            cols = [desc.name for desc in cur.description]
            rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    t1 = time.perf_counter_ns()
    return rows, int((t1 - t0) / 1e6)


def get_county_by_geoid(geoid: str) -> Answer:
    """Direct DB endpoint: return county rows for a given GEOID with GeoJSON, no LM call."""
    # Use parameterized SQL to prevent injection and keep a stable plan
    sql = (
        "SELECT namelsad, stateaabrv, geoid, ST_AsGeoJSON(geom)::json AS geojson "
        "FROM counties WHERE geoid = %s ORDER BY stateaabrv LIMIT 500"
    )
    rows, db_ms = run_sql_params(sql, (geoid,))
    # For UI readability, include an inlined SQL string with the literal value
    display_sql = (
        "SELECT namelsad, stateaabrv, ST_AsGeoJSON(geom)::json AS geojson FROM counties "
        f"WHERE geoid = '{geoid}' ORDER BY stateaabrv LIMIT 500"
    )
    return answer_for_rows(rows, display_sql, db_ms)


def write_log(entry: LogEntry) -> None:
    """Append a single LogEntry to the JSONL run log."""
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")


# System prompt builder
POSTGIS_KNOWLEDGE = (
    "SDBMS: PostGIS (SRID 4269). Use correct PostGIS names: "
    "ST_Area(geom::geography), ST_Perimeter(geom::geography), ST_Touches(a,b), "
    "ST_Intersects(a,b), ST_Contains(a,b), ST_Centroid(geom), ST_Distance(a::geography,b::geography), "
    "ST_NumInteriorRings(geom), ST_NumGeometries(geom). Cast to ::geography for meters; "
    "convert meters to miles with /1609.34 and m² to mi² with /2.59e6."
)

GEOGRAPHIC_DESCRIPTION = (
    "This database contains only United States county polygons; no provinces or non-US regions."
)

FEW_SHOT = f"""
EXAMPLES:
1) List counties in Florida (prefer stateaabrv or full state name):
SELECT name, namelsad FROM counties WHERE stateaabrv='FL' OR UPPER(state)='FLORIDA' ORDER BY name;

2) Counties starting with 'San ' in California:
SELECT namelsad FROM counties WHERE stateaabrv='CA' AND namelsad ILIKE 'San %' ORDER BY namelsad;

3) Area (km^2 and mi^2) of Riverside County, CA (with numeric casts):
SELECT
  ROUND((ST_Area(geom::geography)/1e6)::numeric, 2)   AS area_km2,
  ROUND((ST_Area(geom::geography)/2.59e6)::numeric, 2) AS area_mi2
FROM counties
WHERE namelsad='Riverside County' AND stateaabrv='CA';

4) CA counties touching Nevada (state boundary touch using ST_Touches):
SELECT DISTINCT c1.namelsad FROM counties c1 JOIN counties c2 ON ST_Touches(c1.geom, c2.geom)
WHERE c1.stateaabrv='CA' AND c2.stateaabrv='NV' ORDER BY c1.namelsad;

5) Visualize Madison County in all states (include geometry for map):
SELECT namelsad, stateaabrv, geom FROM counties WHERE namelsad='Madison County' ORDER BY stateaabrv;

6) How many counties are called Madison County?
SELECT COUNT(*) AS n FROM counties WHERE namelsad='Madison County';

7) Three most frequent county names (group by name, not namelsad):
SELECT name, COUNT(*) AS freq FROM counties GROUP BY name ORDER BY freq DESC LIMIT 3;

8) Counties whose name equals their state (text-only):
SELECT namelsad, stateaabrv FROM counties WHERE UPPER(name)=UPPER(state) ORDER BY namelsad;

9) Multi-word county names in Minnesota (text regex):
SELECT namelsad FROM counties WHERE stateaabrv='MN' AND name ~ '\\s' ORDER BY namelsad;

10) Rank all counties in Arizona by area (mi^2):
SELECT namelsad, ROUND((ST_Area(geom::geography)/2.59e6)::numeric, 2) AS area_mi2
FROM counties WHERE stateaabrv='AZ' ORDER BY area_mi2 DESC;

11) Nationwide counties with area < 100 mi^2:
SELECT namelsad, ROUND((ST_Area(geom::geography)/2.59e6)::numeric, 2) AS area_mi2
FROM counties WHERE (ST_Area(geom::geography)/2.59e6) < 100 ORDER BY area_mi2 ASC;

12) Perimeter of Orange County, CA (mi and km):
SELECT
  ROUND((ST_Perimeter(geom::geography)/1609.34)::numeric, 2) AS perim_mi,
  ROUND((ST_Perimeter(geom::geography)/1000)::numeric, 2)    AS perim_km
FROM counties WHERE namelsad='Orange County' AND stateaabrv='CA';

13) South Dakota counties with perimeter > 800 mi:
SELECT namelsad FROM counties WHERE stateaabrv='SD' AND ST_Perimeter(geom::geography)/1609.34 > 800 ORDER BY namelsad;

14) Counties with holes (interior rings) using dumped polygons:
WITH dumped AS (
  SELECT namelsad, (ST_Dump(geom)).geom AS poly
  FROM counties
)
SELECT DISTINCT namelsad FROM dumped WHERE ST_NumInteriorRings(poly) > 0 ORDER BY namelsad;

15) Multipart counties (non-contiguous MultiPolygons):
SELECT namelsad FROM counties WHERE ST_NumGeometries(geom) > 1 ORDER BY namelsad;

16) Counties whose centroid falls outside their polygon:
SELECT namelsad FROM counties WHERE NOT ST_Covers(geom, ST_Centroid(geom)) ORDER BY namelsad;

17) How many neighbors does Utah County, UT have? (touching counties):
SELECT COUNT(DISTINCT b.geoid) AS neighbor_count
FROM counties a JOIN counties b ON ST_Touches(a.geom, b.geom)
WHERE a.namelsad='Utah County' AND a.stateaabrv='UT';

18) Which county in AL has the most neighbors?
SELECT a.namelsad, COUNT(DISTINCT b.geoid) AS neighbor_count
FROM counties a JOIN counties b ON ST_Touches(a.geom, b.geom)
WHERE a.stateaabrv='AL'
GROUP BY a.namelsad
ORDER BY neighbor_count DESC
LIMIT 1;

19) Counties in CA with exactly two neighbors:
WITH nbr AS (
  SELECT a.geoid, a.namelsad, COUNT(DISTINCT b.geoid) AS n
  FROM counties a JOIN counties b ON ST_Touches(a.geom, b.geom)
  WHERE a.stateaabrv='CA'
  GROUP BY a.geoid, a.namelsad
)
SELECT namelsad FROM nbr WHERE n = 2 ORDER BY namelsad;

20) Madison County, Idaho (attribute filter):
SELECT namelsad, stateaabrv FROM counties WHERE namelsad='Madison County' AND stateaabrv='ID';
 
21) Visualize county by GEOID (return geometry for map overlays):
SELECT namelsad, stateaabrv, geom FROM counties WHERE geoid='16065';
""".strip()


def build_system_prompt(schema_tips: Optional[str] = None) -> str:
    """Compose a strict system prompt (schema + routing hints + few-shot) for the LM."""
    base = f"""
You are a strict PostGIS SQL assistant.
We are already connected to the database USCountyDB.

SCHEMA: {SCHEMA_TEXT}
GEOGRAPHY: {GEOGRAPHIC_DESCRIPTION}
KNOWLEDGE: {POSTGIS_KNOWLEDGE}

HARD RULES:
- Output ONLY one SQL statement; no markdown, no backticks, no prose.
- Use the table name counties (do NOT prefix with DB/schema). The only table is counties.
- SELECT-only. Never DDL/DML. No multiple statements.
- Prefer exact matches on stateaabrv for abbreviations like 'CA', 'FL', etc.
- For full state names, use UPPER(state)='FLORIDA' style for case-insensitivity.
- If the question requests geometry, include the column name geom in SELECT; backend will return it as GeoJSON.
- Return only the columns needed to answer; avoid SELECT * unless necessary.

ROUTING HINTS:
- If the prompt says 'visualize', 'map', or asks for geometry, include column geom in SELECT; backend converts it to GeoJSON.
- If it mentions neighbor/adjacent/touching, use a self-join with ST_Touches(a.geom, b.geom).
- Area units: mi^2 = ST_Area(geom::geography)/2.59e6; km^2 = ST_Area(geom::geography)/1e6. Use ROUND((...)::numeric, 2) when rounding.
- Perimeter units: miles = ST_Perimeter(geom::geography)/1609.34; km = /1000. Use ROUND((...)::numeric, 2) when rounding.
- Multipart geometries: ST_NumGeometries(geom) > 1.
- Holes/interior rings: prefer applying ST_NumInteriorRings to dumped polygon parts from ST_Dump(geom).
- Centroid outside polygon: use NOT ST_Covers(geom, ST_Centroid(geom)).
- Name frequency/grouping: group by name (not namelsad, which includes 'County').
- For equality of county name and state, use UPPER(name)=UPPER(state) or UPPER(namelsad)=UPPER(state || ' County').

{FEW_SHOT}
""".strip()
    if schema_tips:
        tips = schema_tips.strip().replace("\n", " ")
        base += f"\nSCHEMA TIPS: {tips}"
    return base


def is_ddl_nl_attempt(nlq: str) -> bool:
    """Detect obvious attempts to get the system to run DDL/DML via natural language."""
    q = nlq.lower()
    return bool(re.search(r"\b(create|drop|alter|truncate|insert|update|delete)\b", q))


def extract_schema_tip(db_error_text: str) -> str:
    """
    Produce a short schema hint from a Postgres/PostGIS error message.
    Helps the retry prompt steer the model toward a valid query.
    """
    msg = db_error_text.lower()

    m = re.search(r"column\s+\"?([a-z0-9_]+)\"?\s+does not exist", msg)
    if m:
        col = m.group(1)
        return f"Column '{col}' does not exist in counties; use columns from {SCHEMA_TEXT}."

    m = re.search(r"relation\s+\"?([a-z0-9_\.]+)\"?\s+does not exist", msg)
    if m:
        rel = m.group(1)
        return f"Only table is counties; do not use '{rel}'. Use counties."

    if "function" in msg and "does not exist" in msg:
        return (
            "Use correct PostGIS functions and argument types for SRID 4269. "
            "Use ST_Area(geom::geography), ST_Perimeter(geom::geography); not misspelled names like ST_Aarea."
        )

    if "operator does not exist" in msg and "geometry" in msg:
        return "Use spatial predicates like ST_Intersects(a,b) or ST_Touches(a,b) instead of '=' on geometry."

    if "operator does not exist" in msg and ("numeric" in msg or "record" in msg):
        return (
            "Cast expressions to numeric correctly. Use ROUND((expression)::numeric, digits), "
            "not ROUND(expression::numeric, digits). Apply division/arithmetic before casting to numeric."
        )

    if "ambiguous" in msg and "column" in msg:
        return "Qualify ambiguous columns with table aliases such as a.namelsad, or use explicit aliases in ORDER BY."

    if "syntax error" in msg:
        return "Fix SQL syntax and ensure a single SELECT targeting counties."

    return "Target table is counties; use columns per schema and valid PostGIS functions."


# Core function
def answer_query(nl_query: str, model: Optional[str] = None, provider: Optional[str] = None) -> Answer:
    """
    End-to-end pipeline for NL→SQL:
    - Validate scope/intent → call LM → sanitize/patch SQL → execute → return Answer.
    - Logs every step to JSONL for a frontend log panel.
    """
    log = LogEntry(
        timestamp=now_iso(),
        nl_query=nl_query,
        lm_model=OLLAMA_MODEL,
        system_prompt_hash="",
        lm_duration_ms=0,
        lm_raw_response="",
        sql_candidate="",
        sql_valid=False,
        sql_final=None,
        sql_rewrite_reason=None,
        db_duration_ms=None,
        row_count=None,
        errors=[],
    )

    # Scope guard for non US prompts
    if is_out_of_scope(nl_query):
        ans = Answer(ok=False, error="USA counties dataset only", scope_rejected=True)
        write_log(log)
        return ans

    # Block obvious DDL or DML phrased as natural language
    if is_ddl_nl_attempt(nl_query):
        log.errors.append("NL DDL/DML attempt blocked")
        write_log(log)
        return Answer(ok=False, error="SQL validation error: DDL/DML not allowed.")

    # Ask the language model for a single SELECT
    try:
        system_prompt = build_system_prompt()
        log.system_prompt_hash = sha1(system_prompt)
        lm_text, lm_ms, eff_model = call_lm(system_prompt, nl_query, model=model, provider=provider)
        log.lm_model = eff_model
        log.lm_duration_ms = lm_ms
        log.lm_raw_response = lm_text
    except Exception as e:
        log.errors.append(f"LM error: {e}")
        write_log(log)
        return Answer(ok=False, error=f"LM error: {e}")

    # Sanitize, validate, and optionally rewrite SQL
    try:
        sql_cand = strip_code_fences(lm_text)
        log.sql_candidate = sql_cand
        sql_final = sanitize_sql(sql_cand)

        # For spatial queries (even COUNT queries), add geometry for visualization
        # This must happen BEFORE other geometry processing
        if is_spatial_query(nl_query):
            sql_final2, added = add_geom_to_spatial_count_query(sql_final, nl_query)
            if added:
                sql_final = sql_final2
                reason = "converted spatial COUNT to visualization query"
                log.sql_rewrite_reason = (log.sql_rewrite_reason + ", " if log.sql_rewrite_reason else "") + reason
            else:
                # Try regular geom addition if not a COUNT query
                sql_final2, added = add_geom_to_select_if_needed(sql_final)
                if added:
                    sql_final = sql_final2
                    reason = "added geom for spatial query visualization"
                    log.sql_rewrite_reason = (log.sql_rewrite_reason + ", " if log.sql_rewrite_reason else "") + reason

        # If this is a simple COUNT over counties (non-spatial), convert to row visualization
        if re.search(r"\bSELECT\s+COUNT\s*\(", sql_final, flags=re.IGNORECASE) and not re.search(r"\bJOIN\s+counties\b", sql_final, flags=re.IGNORECASE):
            sql_vis, changed = convert_simple_count_to_rows(sql_final)
            if changed:
                sql_final = sql_vis
                reason = "converted COUNT to row visualization"
                log.sql_rewrite_reason = (log.sql_rewrite_reason + ", " if log.sql_rewrite_reason else "") + reason

        # Add geometry when the user indicates mapping/visualization intent.
        geom_added_via_intent = False
        if should_include_geom(nl_query):
            sql_final2, added = add_geom_to_select_if_needed(sql_final)
            if added:
                sql_final = sql_final2
                geom_added_via_intent = True
                reason = "added geom for visualization"
                log.sql_rewrite_reason = (log.sql_rewrite_reason + ", " if log.sql_rewrite_reason else "") + reason
        
        # For any non-aggregate SELECT query that returns county data, add geometry
        # This ensures all county listing queries are visualized on the map
        # Only do this if geometry wasn't already added above
        if not geom_added_via_intent and not re.search(r"\bSELECT\s+COUNT\s*\(", sql_final, flags=re.IGNORECASE):
            # Check if this is a query that returns county rows (not just aggregates)
            if re.search(r"\bFROM\s+counties\b", sql_final, flags=re.IGNORECASE):
                # Check if geometry is already present in SELECT clause
                select_match = re.search(r"(?is)^\s*SELECT\s+(.*?)\s+FROM\s", sql_final)
                if select_match:
                    select_clause = select_match.group(1)
                    if not re.search(r"\b((?:[A-Za-z_][A-Za-z0-9_]*\.)?geom|geojson|ST_AsGeoJSON\s*\(\s*geom\s*\))\b", select_clause, flags=re.IGNORECASE):
                        sql_final2, added = add_geom_to_select_if_needed(sql_final)
                        if added:
                            sql_final = sql_final2
                            reason = "added geom for county data visualization"
                            log.sql_rewrite_reason = (log.sql_rewrite_reason + ", " if log.sql_rewrite_reason else "") + reason

        # Small resilience tweaks for typical LM output quirks.
        sql_final, _ = fix_common_function_typos(sql_final)
        sql_final, _ = fix_round_numeric_cast(sql_final)
        sql_final, _ = qualify_ambiguous_columns(sql_final)

        # Convert returned geom to GeoJSON for the map UI (predicates remain untouched).
        sql_final, rewrote = ensure_geojson_projection(sql_final)
        if rewrote:
            reason = "projected geom to GeoJSON (geojson)"
            log.sql_rewrite_reason = (log.sql_rewrite_reason + ", " if log.sql_rewrite_reason else "") + reason
        else:
            # If no geom was found, try adding it one more time as a fallback
            if not re.search(r"\bSELECT\s+COUNT\s*\(", sql_final, flags=re.IGNORECASE):
                sql_final2, added = add_geom_to_select_if_needed(sql_final)
                if added:
                    sql_final = sql_final2
                    sql_final, rewrote2 = ensure_geojson_projection(sql_final)
                    if rewrote2:
                        reason = "added geom and projected to GeoJSON"
                        log.sql_rewrite_reason = (log.sql_rewrite_reason + ", " if log.sql_rewrite_reason else "") + reason
        
        # Fix DISTINCT with JSON after GeoJSON conversion
        sql_final, _ = fix_distinct_with_json(sql_final)

        log.sql_valid = True
        log.sql_final = sql_final

    except Exception as e:
        # Second-chance pass: provide a targeted schema hint back to the LM.
        first_val_err = str(e)
        log.errors.append(f"SQL validation error (attempt 1): {first_val_err}")
        try:
            tip = (
                "Previous output was invalid. Output ONLY one SELECT statement, no comments, "
                "no markdown, targeting table counties. Follow routing hints and examples."
            )
            system_prompt_retry = build_system_prompt(schema_tips=tip)
            log.system_prompt_hash = sha1(system_prompt_retry)
            lm_text2, lm_ms2, eff_model2 = call_lm(system_prompt_retry, nl_query, model=model, provider=provider)
            log.lm_model = eff_model2
            log.lm_duration_ms += lm_ms2
            log.lm_raw_response = f"{log.lm_raw_response}\n--- VALIDATION RETRY RESPONSE ---\n{lm_text2}"

            sql_cand2 = strip_code_fences(lm_text2)
            log.sql_candidate = sql_cand2
            sql_final2 = sanitize_sql(sql_cand2)
            
            # Apply spatial query conversion if needed
            if is_spatial_query(nl_query):
                sql_final2_spatial, added_spatial = add_geom_to_spatial_count_query(sql_final2, nl_query)
                if added_spatial:
                    sql_final2 = sql_final2_spatial
                    reason = "converted spatial COUNT to visualization query (retry)"
                    log.sql_rewrite_reason = (log.sql_rewrite_reason + ", " if log.sql_rewrite_reason else "") + reason
                else:
                    sql_final2_geom, added_geom = add_geom_to_select_if_needed(sql_final2)
                    if added_geom:
                        sql_final2 = sql_final2_geom
                        reason = "added geom for spatial query visualization (retry)"
                        log.sql_rewrite_reason = (log.sql_rewrite_reason + ", " if log.sql_rewrite_reason else "") + reason

            # Convert simple COUNT to row visualization on retry
            if re.search(r"\bSELECT\s+COUNT\s*\(", sql_final2, flags=re.IGNORECASE) and not re.search(r"\bJOIN\s+counties\b", sql_final2, flags=re.IGNORECASE):
                sql_vis2, changed2 = convert_simple_count_to_rows(sql_final2)
                if changed2:
                    sql_final2 = sql_vis2
                    reason = "converted COUNT to row visualization (retry)"
                    log.sql_rewrite_reason = (log.sql_rewrite_reason + ", " if log.sql_rewrite_reason else "") + reason

            # Ensure non-aggregate county listings include geometry (retry)
            if not re.search(r"\bSELECT\s+COUNT\s*\(", sql_final2, flags=re.IGNORECASE):
                if re.search(r"\bFROM\s+counties\b", sql_final2, flags=re.IGNORECASE):
                    m_sel = re.search(r"(?is)^\s*SELECT\s+(.*?)\s+FROM\s", sql_final2)
                    if m_sel and not re.search(r"\b((?:[A-Za-z_][A-Za-z0-9_]*\.)?geom|geojson|ST_AsGeoJSON\s*\(\s*geom\s*\))\b", m_sel.group(1), flags=re.IGNORECASE):
                        sql_tmp, added = add_geom_to_select_if_needed(sql_final2)
                        if added:
                            sql_final2 = sql_tmp
                            reason = "added geom for county data visualization (retry)"
                            log.sql_rewrite_reason = (log.sql_rewrite_reason + ", " if log.sql_rewrite_reason else "") + reason
            
            sql_final2, rewrote2 = ensure_geojson_projection(sql_final2)
            if rewrote2:
                reason = "projected geom to GeoJSON (geojson)"
                log.sql_rewrite_reason = (log.sql_rewrite_reason + ", " if log.sql_rewrite_reason else "") + reason
            
            # Fix DISTINCT with JSON after GeoJSON conversion
            sql_final2, _ = fix_distinct_with_json(sql_final2)
            
            log.sql_valid = True
            log.sql_final = sql_final2
        except Exception as e2:
            log.errors.append(f"SQL validation error (attempt 2): {e2}")
            write_log(log)
            return Answer(ok=False, error=f"SQL validation error: {e2}", lm_ms=log.lm_duration_ms)

    # Execute the SQL
    try:
        rows, db_ms = run_sql(log.sql_final)
        log.db_duration_ms = db_ms
        log.row_count = len(rows)
        # If the result includes GeoJSON and the total size is modest, return all rows
        # so the frontend can visualize every feature (not just the first 20).
        has_geojson = bool(rows) and isinstance(rows[0], dict) and ("geojson" in rows[0])
        preview = rows if (has_geojson and len(rows) <= 500) else rows[:20]

        ans = Answer(
            ok=True,
            rows_preview=preview,
            rows_total=len(rows),
            sql=log.sql_final,
            lm_ms=log.lm_duration_ms,
            db_ms=db_ms,
        )
        write_log(log)
        return ans

    except Exception as e:
        # Retry path with schema-tip regeneration if this is not a connection error.
        first_db_error = str(e)
        log.errors.append(f"DB error (attempt 1): {first_db_error}")

        low = first_db_error.lower()
        if any(s in low for s in ["fe_sendauth", "password", "could not connect", "connection refused", "timeout expired"]):
            write_log(log)
            return Answer(
                ok=False,
                error=f"DB connection/config error: {first_db_error}",
                sql=log.sql_final,
                lm_ms=log.lm_duration_ms,
                db_ms=log.db_duration_ms or 0,
            )

        try:
            tip = extract_schema_tip(first_db_error)
            system_prompt_retry = build_system_prompt(schema_tips=tip)
            log.system_prompt_hash = sha1(system_prompt_retry)
            lm_text2, lm_ms2, eff_model2 = call_lm(system_prompt_retry, nl_query, model=model, provider=provider)
            log.lm_model = eff_model2
            log.lm_duration_ms += lm_ms2
            log.lm_raw_response = f"{log.lm_raw_response}\n--- RETRY 1 RESPONSE ---\n{lm_text2}"

            sql_cand2 = strip_code_fences(lm_text2)
            log.sql_candidate = sql_cand2
            sql_final2 = sanitize_sql(sql_cand2)
            
            # Apply spatial query conversion if needed
            if is_spatial_query(nl_query):
                sql_final2_spatial, added_spatial = add_geom_to_spatial_count_query(sql_final2, nl_query)
                if added_spatial:
                    sql_final2 = sql_final2_spatial
                    reason = "converted spatial COUNT to visualization query (db retry)"
                    log.sql_rewrite_reason = (log.sql_rewrite_reason + ", " if log.sql_rewrite_reason else "") + reason
                else:
                    sql_final2_geom, added_geom = add_geom_to_select_if_needed(sql_final2)
                    if added_geom:
                        sql_final2 = sql_final2_geom
                        reason = "added geom for spatial query visualization (db retry)"
                        log.sql_rewrite_reason = (log.sql_rewrite_reason + ", " if log.sql_rewrite_reason else "") + reason

            # Convert simple COUNT to row visualization on DB retry
            if re.search(r"\bSELECT\s+COUNT\s*\(", sql_final2, flags=re.IGNORECASE) and not re.search(r"\bJOIN\s+counties\b", sql_final2, flags=re.IGNORECASE):
                sql_vis2, changed2 = convert_simple_count_to_rows(sql_final2)
                if changed2:
                    sql_final2 = sql_vis2
                    reason = "converted COUNT to row visualization (db retry)"
                    log.sql_rewrite_reason = (log.sql_rewrite_reason + ", " if log.sql_rewrite_reason else "") + reason

            # Ensure non-aggregate county listings include geometry (db retry)
            if not re.search(r"\bSELECT\s+COUNT\s*\(", sql_final2, flags=re.IGNORECASE):
                if re.search(r"\bFROM\s+counties\b", sql_final2, flags=re.IGNORECASE):
                    m_sel = re.search(r"(?is)^\s*SELECT\s+(.*?)\s+FROM\s", sql_final2)
                    if m_sel and not re.search(r"\b((?:[A-Za-z_][A-Za-z0-9_]*\.)?geom|geojson|ST_AsGeoJSON\s*\(\s*geom\s*\))\b", m_sel.group(1), flags=re.IGNORECASE):
                        sql_tmp, added = add_geom_to_select_if_needed(sql_final2)
                        if added:
                            sql_final2 = sql_tmp
                            reason = "added geom for county data visualization (db retry)"
                            log.sql_rewrite_reason = (log.sql_rewrite_reason + ", " if log.sql_rewrite_reason else "") + reason
            
            sql_final2, rewrote2 = ensure_geojson_projection(sql_final2)
            if rewrote2:
                reason = "projected geom to GeoJSON (geojson)"
                log.sql_rewrite_reason = (log.sql_rewrite_reason + ", " if log.sql_rewrite_reason else "") + reason
            
            # Fix DISTINCT with JSON after GeoJSON conversion
            sql_final2, _ = fix_distinct_with_json(sql_final2)
            
            log.sql_valid = True
            log.sql_final = sql_final2

            rows2, db_ms2 = run_sql(log.sql_final)
            log.db_duration_ms = db_ms2
            log.row_count = len(rows2)
            log.sql_rewrite_reason = (log.sql_rewrite_reason + ", " if log.sql_rewrite_reason else "") + f"retry_with_schema_tip: {tip}"
            ans2 = Answer(
                ok=True,
                rows_preview=rows2[:20],
                rows_total=len(rows2),
                sql=log.sql_final,
                lm_ms=log.lm_duration_ms,
                db_ms=db_ms2,
            )
            write_log(log)
            return ans2

        except ValueError as v:
            log.errors.append(f"SQL validation error on retry: {v}")
            write_log(log)
            return Answer(
                ok=False,
                error=f"SQL validation error: {v}",
                sql=log.sql_final,
                lm_ms=log.lm_duration_ms,
                db_ms=log.db_duration_ms or 0,
            )
        except Exception as e2:
            log.errors.append(f"DB error (attempt 2): {e2}")
            write_log(log)
            return Answer(
                ok=False,
                error=f"DB error: {e2}",
                sql=log.sql_final,
                lm_ms=log.lm_duration_ms,
                db_ms=log.db_duration_ms or 0,
            )


# Small CLI for quick testing
def _print(obj: Any) -> None:
    """Pretty-print a dict-like object as JSON (CLI helper)."""
    print(json.dumps(obj, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Milestone 2 NL→SQL backend")
    ap.add_argument("--query", "-q", required=True, help="Natural-language query")
    ap.add_argument("--model", "-m", required=False, help="Override model (e.g., llama3.1:8b or hf:meta-llama/Llama-3.3-70B-Instruct:cerebras)")
    ap.add_argument("--provider", "-p", required=False, choices=["ollama", "hf"], help="LM provider override")
    args = ap.parse_args()

    ans = answer_query(args.query, model=args.model, provider=args.provider)
    _print(ans.dict())
