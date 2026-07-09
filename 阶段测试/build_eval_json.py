"""Build a balanced eval json (edit instruction + source video) for the edit eval system.

Samples a small, class-balanced subset from a ReCo edit manifest (default: the
held-out `distill_for_bernini_eval-80` set, whose source videos exist on disk) and
writes `阶段测试/eval_data.json` with the fields the evaluator consumes:

    [{"prompt": <edit instruction>, "source": <abs mp4>, "refs": [...], "kind": "add"}, ...]

Usage:
    python 阶段测试/build_eval_json.py                 # default: 2 per add/remove/replace
    python 阶段测试/build_eval_json.py --per_kind 3 --kinds add remove replace convert
    python 阶段测试/build_eval_json.py --manifest /path/edit_manifest.json --out /path/eval_data.json
"""
import argparse
import json
import os

DEFAULT_MANIFEST = (
    "/apdcephfs_hzlf/share_1227201/xinyu/Dataset/ReCo/"
    "distill_for_bernini_eval-80/edit_manifest.json"
)
HERE = os.path.dirname(os.path.abspath(__file__))


def edit_kind(prompt: str) -> str:
    """First token of the instruction = edit type (add / remove / replace / ...)."""
    toks = str(prompt).strip().lstrip("*").strip().split()
    return toks[0].lower() if toks else ""


def pick_spread(pool, k):
    """Deterministically pick k items evenly spread across a list."""
    if k <= 0 or not pool:
        return []
    if k >= len(pool):
        return list(pool)
    return [pool[int(i * (len(pool) - 1) / max(k - 1, 1))] for i in range(k)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST,
                    help="source ReCo edit_manifest.json")
    ap.add_argument("--out", default=os.path.join(HERE, "eval_data.json"))
    ap.add_argument("--per_kind", type=int, default=2,
                    help="number of cases per edit kind")
    ap.add_argument("--kinds", nargs="*", default=["add", "remove", "replace"],
                    help="edit kinds to include")
    args = ap.parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)

    by_kind = {k: [] for k in args.kinds}
    for it in manifest:
        k = edit_kind(it.get("prompt", ""))
        if k in by_kind:
            by_kind[k].append(it)

    out = []
    for k in args.kinds:
        for it in pick_spread(by_kind[k], args.per_kind):
            src = it.get("source", "")
            if not os.path.exists(src):
                print(f"[build] WARN skip (missing source): {src}")
                continue
            out.append({
                "prompt": it.get("prompt", ""),
                "source": src,
                "refs": it.get("refs", []) or [],
                "task_type": it.get("task_type", "v2v"),
                "kind": k,
            })

    with open(args.out, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    counts = {}
    for it in out:
        counts[it["kind"]] = counts.get(it["kind"], 0) + 1
    print(f"[build] wrote {len(out)} cases -> {args.out}")
    print(f"[build] per-kind: {counts}")


if __name__ == "__main__":
    main()
