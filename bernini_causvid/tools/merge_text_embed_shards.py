"""Fold per-shard `text_embed` fields (from a sharded gen_text_embeds.py run)
back into the original index.json, preserving its order.

Each shard wrote index.txtshard*.json containing the entries it processed, each
with a `text_embed` field. We match those back onto the original index by the
`source` latent filename (unique per entry) and write the augmented index in
place (original backed up to <index>.bak).

Run:
    python bernini_causvid/tools/merge_text_embed_shards.py --index <dir>/index.json
"""

import argparse
import glob
import json
import os
import shutil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True, help="original index.json to augment in place")
    args = ap.parse_args()

    root = os.path.dirname(os.path.abspath(args.index))
    shards = sorted(glob.glob(os.path.join(root, "index.txtshard*.json")))
    if not shards:
        raise SystemExit(f"no index.txtshard*.json under {root}")

    src2embed = {}
    for s in shards:
        items = json.load(open(s))
        for it in items:
            if "text_embed" in it and it.get("source") is not None:
                src2embed[it["source"]] = it["text_embed"]
        print(f"[txtmerge] {os.path.basename(s)}: {len(items)} items")

    index = json.load(open(args.index))
    filled, missing = 0, 0
    for it in index:
        emb = src2embed.get(it.get("source"))
        if emb is None:
            missing += 1
            continue
        # sanity: the embed file should actually exist next to the latents.
        if not os.path.exists(os.path.join(root, emb)):
            missing += 1
            continue
        it["text_embed"] = emb
        filled += 1

    bak = args.index + ".bak"
    if not os.path.exists(bak):
        shutil.copyfile(args.index, bak)
    with open(args.index, "w") as f:
        json.dump(index, f, indent=2)
    print(
        f"[txtmerge] filled text_embed on {filled}/{len(index)} entries "
        f"(missing {missing}); updated {args.index} (backup at {bak})"
    )


if __name__ == "__main__":
    main()
