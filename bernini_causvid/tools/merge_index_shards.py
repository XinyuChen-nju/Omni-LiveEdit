"""Merge per-shard index files (index.shard*.json) produced by a sharded
gen_edit_targets.py run into a single index.json the dataset can consume.

Run:
    python bernini_causvid/tools/merge_index_shards.py --out_dir <dir with index.shard*.json>
"""

import argparse
import glob
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    shards = sorted(glob.glob(os.path.join(args.out_dir, "index.shard*.json")))
    if not shards:
        raise SystemExit(f"no index.shard*.json under {args.out_dir}")

    merged = []
    for s in shards:
        items = json.load(open(s))
        merged.extend(items)
        print(f"[merge] {os.path.basename(s)}: {len(items)} items")

    out = os.path.join(args.out_dir, "index.json")
    json.dump(merged, open(out, "w"), indent=2)
    print(f"[merge] wrote {out} ({len(merged)} items from {len(shards)} shards)")


if __name__ == "__main__":
    main()
