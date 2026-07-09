"""Token-level attention mask visualization, replicating the exact logic in
   causal_edit_model.py: _prepare_edit_tf_attn_mask (TF) and _prepare_edit_attn_mask (no-TF).
   Padding-to-128 is dropped (it carries no semantics, only flex-attn alignment)."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

# ---- representative params -------------------------------------------------
COND_LEN = 4          # source/ref condition tokens
NUM_FRAMES = 3        # target frames
FRAME_SEQLEN = 2      # tokens per frame (latent h*w, shrunk for display)
NFPB = 1              # num_frame_per_block  -> attn_block = FRAME_SEQLEN*NFPB
ATTN_BLOCK = FRAME_SEQLEN * NFPB
NOISY_LEN = NUM_FRAMES * FRAME_SEQLEN


# ============================= TF mask =====================================
def build_tf_mask():
    clean_start = COND_LEN
    noisy_start = COND_LEN + NOISY_LEN
    total = COND_LEN + 2 * NOISY_LEN
    clean_end = np.zeros(total, dtype=int)
    clean_ctx_end = np.zeros(total, dtype=int)
    nn_start = np.zeros(total, dtype=int)
    nn_end = np.zeros(total, dtype=int)
    for bi, start in enumerate(range(0, NOISY_LEN, ATTN_BLOCK)):
        cs = clean_start + start
        clean_end[cs: cs + ATTN_BLOCK] = cs + ATTN_BLOCK
        ns = noisy_start + start
        nn_start[ns: ns + ATTN_BLOCK] = ns
        nn_end[ns: ns + ATTN_BLOCK] = ns + ATTN_BLOCK
        clean_ctx_end[ns: ns + ATTN_BLOCK] = clean_start + bi * ATTN_BLOCK

    M = np.zeros((total, total), dtype=bool)
    for q in range(total):
        for k in range(total):
            q_cond = q < COND_LEN
            kv_cond = k < COND_LEN
            q_clean = (q >= clean_start) and (q < noisy_start)
            q_noisy = q >= noisy_start
            cond_rule = q_cond and kv_cond
            clean_rule = q_clean and (kv_cond or (clean_start <= k < clean_end[q]))
            noisy_rule = q_noisy and (
                kv_cond
                or (clean_start <= k < clean_ctx_end[q])
                or (nn_start[q] <= k < nn_end[q]))
            M[q, k] = cond_rule or clean_rule or noisy_rule or (q == k)
    return M, total


# =========================== no-TF mask ====================================
def build_notf_mask():
    total = COND_LEN + NOISY_LEN
    tgt_end = np.zeros(total, dtype=int)
    for start in range(0, NOISY_LEN, ATTN_BLOCK):
        tgt_end[COND_LEN + start: COND_LEN + start + ATTN_BLOCK] = COND_LEN + start + ATTN_BLOCK
    M = np.zeros((total, total), dtype=bool)
    for q in range(total):
        for k in range(total):
            q_cond = q < COND_LEN
            kv_cond = k < COND_LEN
            cond_rule = q_cond and kv_cond
            causal_tgt = (not kv_cond) and (k < tgt_end[q]) and (k >= 0)
            tgt_rule = (not q_cond) and (kv_cond or causal_tgt)
            M[q, k] = cond_rule or tgt_rule or (q == k)
    return M, total


def labels_tf():
    lab = []
    for i in range(COND_LEN):
        lab.append(f"Cond{i}")
    for f in range(NUM_FRAMES):
        for s in range(FRAME_SEQLEN):
            lab.append(f"cB{f}.{s}")
    for f in range(NUM_FRAMES):
        for s in range(FRAME_SEQLEN):
            lab.append(f"nB{f}.{s}")
    return lab


def labels_notf():
    lab = []
    for i in range(COND_LEN):
        lab.append(f"Cond{i}")
    for f in range(NUM_FRAMES):
        for s in range(FRAME_SEQLEN):
            lab.append(f"nB{f}.{s}")
    return lab


def draw(ax, M, labels, title, seps):
    n = M.shape[0]
    ax.imshow(M, cmap="Greens", vmin=0, vmax=1, aspect="equal")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Key (attended-to)", fontsize=9)
    ax.set_ylabel("Query (attends)", fontsize=9)
    ax.set_title(title, fontsize=11, pad=10)
    # grid
    ax.set_xticks(np.arange(-.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-.5, n, 1), minor=True)
    ax.grid(which="minor", color="#cccccc", linewidth=0.5)
    # region separators
    for s in seps:
        ax.axvline(s - 0.5, color="red", linewidth=1.6)
        ax.axhline(s - 0.5, color="red", linewidth=1.6)
    # annotate ✓
    for q in range(n):
        for k in range(n):
            if M[q, k]:
                ax.text(k, q, "✓", ha="center", va="center", fontsize=6, color="#114411")


fig, axes = plt.subplots(1, 2, figsize=(16, 8))

M1, _ = build_tf_mask()
draw(axes[0], M1, labels_tf(),
     "Teacher Forcing mask\n[ Cond | Clean(GT history) | Noisy ]",
     seps=[COND_LEN, COND_LEN + NOISY_LEN])

M2, _ = build_notf_mask()
draw(axes[1], M2, labels_notf(),
     "No-TF mask (inference / Stage3)\n[ Cond | Noisy(block-causal) ]",
     seps=[COND_LEN])

fig.suptitle("Edit attention visibility — token level  (cond=4, frames=3, frame_seqlen=2, nfpb=1)",
             fontsize=13, y=0.98)
fig.tight_layout(rect=[0, 0, 1, 0.95])
out = "/apdcephfs_hzlf/share_1227201/xinyu/Causal-Forcing/tf_mask_visualization.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print("saved:", out)
