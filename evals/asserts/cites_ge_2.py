def get_assert(output: str, context) -> dict:
    from ven_eval._meta import get_metadata
    n = int(get_metadata(context).get("citations_count", 0))
    return {"pass": n >= 2, "score": 1.0 if n >= 2 else 0.0,
             "reason": f"citations_count={n} (need >= 2)"}
