def get_assert(output: str, context) -> dict:
    from ven_eval._meta import get_metadata

    ids = (get_metadata(context).get("budget") or {}).get("selected_source_ids") or []
    ok = len(ids) > 0
    return {"pass": ok, "score": 1.0 if ok else 0.0, "reason": f"selected_source_ids={len(ids)}"}
