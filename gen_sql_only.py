from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Tuple, Optional

from dotenv import load_dotenv

# Reuse functions from backend without executing the database step
from nl2sql_backend import (
    build_system_prompt,
    call_lm,
    strip_code_fences,
    sanitize_sql,
    ensure_geojson_projection,
    add_geom_to_select_if_needed,
    fix_common_function_typos,
    fix_round_numeric_cast,
    qualify_ambiguous_columns,
    should_include_geom,
)


def nl_to_sql(nl_query: str, model: Optional[str] = None, provider: Optional[str] = None) -> Tuple[Dict[str, Any], List[str]]:
    """Generate and sanitize SQL for a single NL query without running DB."""
    errors: List[str] = []
    result: Dict[str, Any] = {
        "nl_query": nl_query,
        "lm_ms": 0,
        "sql_candidate": None,
        "sql_final": None,
        "sql_rewrite_reason": None,
    }

    try:
        system_prompt = build_system_prompt()
        lm_text, lm_ms, eff_model = call_lm(system_prompt, nl_query, model=model, provider=provider)
        result["lm_ms"] = lm_ms
        result["sql_candidate"] = strip_code_fences(lm_text)
    except Exception as e:
        errors.append(f"LM error: {e}")
        return result, errors

    def _sanitize(sql_text: str) -> str:
        sqlx = sanitize_sql(sql_text)  # base validation/cleanup

        # Visualization intent: add geom if needed
        if should_include_geom(nl_query):
            sql2, added = add_geom_to_select_if_needed(sqlx)
            if added:
                sqlx = sql2
                reason = "added geom for visualization"
                result["sql_rewrite_reason"] = (
                    (result["sql_rewrite_reason"] + ", ") if result["sql_rewrite_reason"] else ""
                ) + reason

        # Minor LM quirks fixes
        sqlx, _ = fix_common_function_typos(sqlx)
        # Note: skip fix_round_numeric_cast here to avoid over-rewriting; backend handles cautiously
        sqlx, _ = qualify_ambiguous_columns(sqlx)

        # Project geom to GeoJSON for frontend mapping
        sqlx, rewrote = ensure_geojson_projection(sqlx)
        if rewrote:
            reason = "projected geom to GeoJSON (geojson)"
            result["sql_rewrite_reason"] = (
                (result["sql_rewrite_reason"] + ", ") if result["sql_rewrite_reason"] else ""
            ) + reason

        return sqlx

    try:
        sql = _sanitize(result["sql_candidate"])  # base validation/cleanup
        result["sql_final"] = sql
    except Exception as e:
        # Retry once with a stricter reminder like the backend
        try:
            tip = (
                "Previous output was invalid. Output ONLY one SELECT statement, no comments, "
                "no markdown, targeting table counties. Follow routing hints and examples."
            )
            sys_prompt_retry = build_system_prompt(schema_tips=tip)
            lm_text2, lm_ms2, _ = call_lm(sys_prompt_retry, nl_query, model=model, provider=provider)
            result["lm_ms"] = int(result["lm_ms"]) + int(lm_ms2)
            result["sql_candidate"] = strip_code_fences(lm_text2)
            sql = _sanitize(result["sql_candidate"])  # sanitize retry
            result["sql_final"] = sql
        except Exception as e2:
            errors.append(f"SQL validation error: {e2}")

    return result, errors


def main() -> None:
    load_dotenv()  # pick up OLLAMA_URL/MODEL from .env unless overridden

    ap = argparse.ArgumentParser(description="Generate sanitized SQL for NL prompts (no DB exec)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--query", "-q", help="Single natural-language query")
    src.add_argument("--file", "-f", help="Path to a text file with one prompt per line")
    ap.add_argument("--jsonl", action="store_true", help="Emit results as JSONL instead of pretty JSON")
    ap.add_argument("--model", "-m", help="Override model (e.g., llama3.1:8b or hf:meta-llama/Llama-3.3-70B-Instruct:cerebras)")
    ap.add_argument("--provider", "-p", choices=["ollama", "hf"], help="LM provider override")
    args = ap.parse_args()

    prompts: List[str]
    if args.query:
        prompts = [args.query]
    else:
        try:
            with open(args.file, "r", encoding="utf-8") as fh:
                prompts = [ln.strip() for ln in fh if ln.strip() and not ln.strip().startswith("#")]
        except Exception as e:
            print(f"Error reading prompts file: {e}", file=sys.stderr)
            sys.exit(2)

    any_err = False
    for p in prompts:
        res, errs = nl_to_sql(p, model=args.model, provider=args.provider)
        out = {**res, "errors": errs}
        if args.jsonl:
            print(json.dumps(out, ensure_ascii=False))
        else:
            print(json.dumps(out, indent=2, ensure_ascii=False))
        if errs:
            any_err = True

    sys.exit(1 if any_err else 0)


if __name__ == "__main__":
    main()
