def get_assert(output: str, context) -> dict:
    from ven_eval._meta import get_metadata

    used = (get_metadata(context).get("budget") or {}).get("used_tokens")
    ok = isinstance(used, int) and used > 0
    return {"pass": ok, "score": 1.0 if ok else 0.0, "reason": f"used_tokens={used}"}
