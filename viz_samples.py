import sys, json, os, random
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import imageio

sys.path.insert(0, '/opt/dlami/nvme/chenxinyu/project/Universal-Edit-Forcing')
from utils.wan_wrapper import WanVAEWrapper

VAE_PATH = '/opt/dlami/nvme/chenxinyu/project/Causal-Forcing/wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth'
META_BASE = '/opt/dlami/nvme/chenxinyu/data/Universal-Edit-Metadata/datasets'
OUT_DIR = '/opt/dlami/nvme/chenxinyu/project/Universal-Edit-Forcing/viz_samples'
N_PER = 20
N_REF = 2
CAP_H = 64
FPS = 8
SEED = 42

def decode_video(vae, path):
    lt = torch.load(path, map_location='cpu')
    with torch.no_grad():
        px = vae.decode_to_pixel(lt.unsqueeze(0).cuda())
    rgb = px[0].float()
    rgb = ((rgb + 1) / 2 * 255).clamp(0, 255).byte()
    rgb = rgb.permute(0, 2, 3, 1).cpu().numpy()
    return rgb

def decode_image(vae, path):
    lt = torch.load(path, map_location='cpu')
    with torch.no_grad():
        px = vae.decode_to_pixel(lt.unsqueeze(0).cuda())
    rgb = px[0, 0].float()
    rgb = ((rgb + 1) / 2 * 255).clamp(0, 255).byte()
    rgb = rgb.permute(1, 2, 0).cpu().numpy()
    return rgb

def resize(frame, w, h):
    img = Image.fromarray(frame)
    img = img.resize((w, h), Image.BILINEAR)
    return np.array(img)

def caption_image(text, width, bg=(15, 15, 18), fg=(255, 255, 255)):
    img = Image.new('RGB', (width, CAP_H), bg)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 22)
    except Exception:
        font = ImageFont.load_default()
    while text and draw.textlength(text, font=font) > width - 20:
        text = text[:-1]
    draw.text((10, 18), text, fill=fg, font=font)
    return np.array(img)

def t2v_visualize(vae, s, out_path):
    tgt = decode_video(vae, s['target'])
    H, W = tgt.shape[1], tgt.shape[2]
    text = '[generate] {}'.format(s.get('prompt', ''))
    frames = []
    for t in range(tgt.shape[0]):
        cap = caption_image(text, W)
        frame = np.concatenate([tgt[t], cap], axis=0)
        frames.append(frame)
    imageio.mimsave(out_path, frames, fps=FPS)

def v2v_visualize(vae, s, out_path):
    src = decode_video(vae, s['source'])
    tgt = decode_video(vae, s['target'])
    T = min(src.shape[0], tgt.shape[0])
    src, tgt = src[:T], tgt[:T]
    H, W = src.shape[1], src.shape[2]
    landscape = W >= H
    text = '[{}] {}'.format(s.get('edit_type', 'v2v'), s.get('prompt', ''))
    frames = []
    for t in range(T):
        if landscape:
            body = np.concatenate([src[t], tgt[t]], axis=0)
            cap = caption_image(text, body.shape[1])
            frame = np.concatenate([body, cap], axis=0)
        else:
            body = np.concatenate([src[t], tgt[t]], axis=1)
            cap = caption_image(text, body.shape[1])
            frame = np.concatenate([body, cap], axis=0)
        frames.append(frame)
    imageio.mimsave(out_path, frames, fps=FPS)

def rv2v_visualize(vae, s, out_path):
    ref_paths = [p for p in s.get('refs', []) if os.path.exists(p)][:N_REF]
    refs = [decode_image(vae, p) for p in ref_paths]
    src = decode_video(vae, s['source'])
    tgt = decode_video(vae, s['target'])
    T = min(src.shape[0], tgt.shape[0])
    src, tgt = src[:T], tgt[:T]
    H, W = src.shape[1], src.shape[2]
    landscape = W >= H
    prompt = s.get('prompt', '')
    text = '[{}] {}'.format(s.get('edit_type', 'rv2v'), prompt if prompt else '(empty prompt)')
    frames = []
    for t in range(T):
        if landscape:
            n = len(refs)
            ref_row_w = W // max(n, 1)
            ref_imgs = [resize(r, ref_row_w, ref_row_w) for r in refs]
            ref_row = np.concatenate(ref_imgs, axis=1) if n > 0 else np.zeros((ref_row_w, W, 3), dtype=np.uint8)
            if ref_row.shape[1] != W:
                ref_row = resize(ref_row, W, ref_row.shape[0])
            body = np.concatenate([ref_row, src[t], tgt[t]], axis=0)
            cap = caption_image(text, body.shape[1])
            frame = np.concatenate([body, cap], axis=0)
        else:
            n = len(refs)
            ref_col_h = H // max(n, 1)
            ref_imgs = [resize(r, ref_col_h, ref_col_h) for r in refs]
            ref_col = np.concatenate(ref_imgs, axis=0) if n > 0 else np.zeros((H, H, 3), dtype=np.uint8)
            if ref_col.shape[0] != H:
                ref_col = resize(ref_col, ref_col.shape[1], H)
            body = np.concatenate([ref_col, src[t], tgt[t]], axis=1)
            cap = caption_image(text, body.shape[1])
            frame = np.concatenate([body, cap], axis=0)
        frames.append(frame)
    imageio.mimsave(out_path, frames, fps=FPS)

def main():
    random.seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)
    print('loading VAE ...', flush=True)
    vae = WanVAEWrapper(VAE_PATH).cuda().eval()
    print('VAE loaded', flush=True)

    datasets = sys.argv[1:] if len(sys.argv) > 1 else ['reco', 'mocha_replacement']

    for name in datasets:
        idx = json.load(open(os.path.join(META_BASE, name, 'index.json')))
        random.shuffle(idx)
        if name == 'reco':
            by_et = {}
            for s in idx:
                by_et.setdefault(s.get('edit_type', 'unknown'), []).append(s)
            picked = []
            for et, lst in by_et.items():
                picked.extend(lst[: (N_PER // max(len(by_et), 1)) + 1])
            picked = picked[:N_PER]
        else:
            # 过滤 latent 未复制的样本
            picked = []
            for s in idx:
                ok = os.path.exists(s.get('target', ''))
                if ok and len(picked) < N_PER:
                    picked.append(s)
                if len(picked) >= N_PER:
                    break

        sub_out = os.path.join(OUT_DIR, name)
        os.makedirs(sub_out, exist_ok=True)
        print('=== {} : {} samples ==='.format(name, len(picked)), flush=True)
        for i, s in enumerate(picked):
            out_path = os.path.join(sub_out, '{}_{}.mp4'.format(s.get('edit_type', 't'), s.get('sample_id', i)))
            try:
                if s['task_type'] == 't2v':
                    t2v_visualize(vae, s, out_path)
                elif s['task_type'] == 'rv2v':
                    rv2v_visualize(vae, s, out_path)
                else:
                    v2v_visualize(vae, s, out_path)
                print('  [{}/{}] done: {}'.format(i + 1, len(picked), os.path.basename(out_path)), flush=True)
            except Exception as e:
                print('  [{}/{}] FAIL {}: {}'.format(i + 1, len(picked), s.get('sample_id', i), str(e)[:120]), flush=True)
    print('ALL DONE', flush=True)

if __name__ == '__main__':
    main()
