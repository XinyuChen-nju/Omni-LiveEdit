"""Stage 1: autoregressive (teacher-forcing) diffusion for the edit student.

This is the editing counterpart of `model/diffusion.py`. It turns the bidirectional
Bernini weights into a *causal* multi-step editing model: the target stream is
denoised block-causally while the clean target history is teacher-forced and the
source/reference latents are carried as an always-visible condition prefix.

  loss = E_t || flow_pred - (noise - clean_target) ||^2   (flow-matching, per-block t)

The output ar_diffusion checkpoint initialises Stage 2 (Causal ODE / CF++ CD).
"""
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.wan_wrapper import WanTextEncoder, WanVAEWrapper

from .edit_wrapper import EditDiffusionWrapper
from .ckpt import load_edit_generator_state, report_load_state


class EditDiffusion(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.device = device
        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32

        model_name = getattr(config, "model_name", "Bernini-R-1.3B")
        model_path = getattr(config, "model_path", None)
        text_encoder_path = getattr(config, "text_encoder_path", None)
        tokenizer_path = getattr(config, "tokenizer_path", None)
        vae_path = getattr(config, "vae_path", None)
        self.num_frame_per_block = getattr(config, "num_frame_per_block", 1)
        tshift = getattr(config, "timestep_shift", 5.0)

        self.generator = EditDiffusionWrapper(
            model_name=model_name, timestep_shift=tshift,
            num_frame_per_block=self.num_frame_per_block, bidirectional=False,
            model_path=model_path)
        self.generator.model.requires_grad_(True)

        ckpt = getattr(config, "generator_ckpt", None)
        if ckpt:
            report_load_state(self.generator, load_edit_generator_state(ckpt),
                              tag="EditDiffusion.generator")

        if getattr(config, "gradient_checkpointing", False):
            self.generator.enable_gradient_checkpointing()

        # With precomputed prompt embeds (cache_text_embeds), the umT5-xxl text
        # encoder is never called during training, so we skip building it entirely
        # to free ~11GB of GPU memory.
        self.use_cached_text_embeds = bool(getattr(config, "cache_text_embeds", False))
        if self.use_cached_text_embeds:
            self.text_encoder = None
        else:
            self.text_encoder = WanTextEncoder(
                text_encoder_path=text_encoder_path,
                tokenizer_path=tokenizer_path).requires_grad_(False)
        self.vae = WanVAEWrapper(vae_path=vae_path).requires_grad_(False)

        self.scheduler = self.generator.get_scheduler()
        self.scheduler.timesteps = self.scheduler.timesteps.to(device)

        self.num_train_timestep = getattr(config, "num_train_timestep", 1000)
        self.teacher_forcing = getattr(config, "teacher_forcing", True)
        self.noise_aug_max_t = getattr(config, "noise_augmentation_max_timestep", 0)
        # 给 source（原视频）condition 注入轻微噪声做正则（防止照抄源视频）。这里是
        # 时间步“值”上界（0~1000，0=关闭），训练时对每帧采样 [0, 此值] 的噪声水平。
        # 含 0 使“干净源”在训练分布内 → 推理用干净 source（source_noise=0）即 in-distribution。
        # 与推理端 EditCausalInferencePipeline.source_noise 同义（都是时间步值）。
        self.source_noise_max_t = int(getattr(config, "source_noise_max_timestep", 0) or 0)
        self.source_timestep_mode = str(
            getattr(config, "source_timestep_mode", "source")).lower()
        if self.source_timestep_mode not in ("source", "target"):
            raise ValueError("source_timestep_mode must be 'source' or 'target'")
        # Reference latents stay clean; this controls only their causal-student
        # timestep embedding. Zero is the clean-condition default.
        self.ref_timestep = float(getattr(config, "ref_timestep", 0) or 0)
        if abs(self.ref_timestep) > 1e-8:
            raise ValueError(
                f"ref_timestep must be 0 for clean ref time embedding, got {self.ref_timestep}")
        if self.source_timestep_mode != "source":
            raise ValueError(
                "source_timestep_mode must be 'source' (clean source time embedding = 0); "
                f"got {self.source_timestep_mode!r}")

        # ---- ReCo-style region-reinforced (latent) loss (arXiv:2512.17650) ----
        # Up-weight the EDITING region (where the clean GT target differs from the
        # source) in the flow loss, so the small-but-important edit is not drowned by
        # the large easy-to-copy background (which otherwise dominates the mean loss
        # and starves the edit region of gradient -> edit fades on later frames). The
        # mask is derived per-frame from |target - source| in latent space, so it also
        # tracks where the edit is on every frame. Defaults keep the loss unchanged.
        self.region_loss = bool(getattr(config, "region_loss", False))
        self.region_edit_weight = float(getattr(config, "region_edit_weight", 4.0))
        self.region_mask_threshold = float(getattr(config, "region_mask_threshold", 0.10))
        self.region_mask_soft = bool(getattr(config, "region_mask_soft", False))
        self.region_weight_normalize = bool(getattr(config, "region_weight_normalize", True))
        # 按“编辑类型”决定是否启用区域加权。类型来自数据 index.json 的显式 `edit_type`
        # 字段（add/remove/replace/convert/...），不解析 prompt。config 的
        # region_loss_by_type 是 {类型: true/false} 映射：false=该类型关闭区域加权、退回
        # 全帧均匀 loss（风格化 convert 整帧都变、无可照抄背景，应设 false）。未在映射中
        # 出现的类型默认 true（启用）。为空/缺省时所有类型都启用，行为与原来一致。
        _rlbt = getattr(config, "region_loss_by_type", None)
        self.region_loss_by_type = (
            {str(k).lower(): bool(v) for k, v in dict(_rlbt).items()} if _rlbt else {}
        )
        # 可选：对（硬）编辑区 mask 做 latent 空间闭运算（先膨胀后腐蚀），填补内部空洞、平滑边界。
        # 默认 0 -> 不做，mask 与原来完全一致。仅作用于硬 mask（region_mask_soft=false）。
        self.region_mask_close_kernel = int(getattr(config, "region_mask_close_kernel", 0) or 0)

        # Noisy target edit-token attention ranking: selected self-attention layers
        # should prefer reference keys over the frame-aligned source key.
        self.ref_attn_loss = bool(getattr(config, "ref_attn_loss", False))
        self.ref_attn_loss_weight = float(getattr(config, "ref_attn_loss_weight", 0.0))
        self.ref_attn_loss_layers = tuple(int(x) for x in getattr(
            config, "ref_attn_loss_layers", (10, 15, 20, 25)))
        self.ref_attn_loss_margin = float(getattr(config, "ref_attn_loss_margin", 0.0))
        self.ref_attn_topk = int(getattr(config, "ref_attn_topk", 16))
        self.ref_attn_max_queries = int(getattr(config, "ref_attn_max_queries", 2048))
        self.ref_attn_query_chunk = int(getattr(config, "ref_attn_query_chunk", 128))
        self.ref_attn_loss_types = {str(x).strip().lower() for x in getattr(
            config, "ref_attn_loss_types", ("tryon",))}

    # per-block timestep indices (same block shares a noise level), like base._get_timestep.
    def _sample_timestep_index(self, b, f, lo, hi):
        idx = torch.randint(int(lo), int(hi), (b, f), device=self.device, dtype=torch.long)
        block = max(1, int(self.num_frame_per_block))
        for start in range(0, f, block):
            idx[:, start:min(start + block, f)] = idx[:, start:start + 1]
        return idx

    # per-frame timestep VALUE in [0, hi] (inclusive), shared within a block. Used for
    # the "slight noise" augmentations (clean context / source): a small `hi` means a
    # small timestep value -> small sigma -> slight noise (since timesteps ~= sigma*1000),
    # and 0 is included so the clean case stays in-distribution. Same value-space
    # convention as inference's context_noise / source_noise.
    def _sample_timestep_value(self, b, f, hi):
        v = torch.randint(0, int(hi) + 1, (b, f), device=self.device)
        block = max(1, int(self.num_frame_per_block))
        for start in range(0, f, block):
            v[:, start:min(start + block, f)] = v[:, start:start + 1]
        return v.to(self.dtype)

    @torch.no_grad()
    def _edit_region_mask(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Per-frame editing-region mask [B, F, 1, H, W] from |target - source| in latent
        space (channel-averaged, then per-sample max-normalised to [0, 1]).

        Hard mask (default): 1 where the normalised diff exceeds `region_mask_threshold`,
        else 0. `region_mask_soft` returns the continuous [0, 1] score instead. Detached
        (no grad): used purely as a per-position loss weight.
        """
        diff = ((target.float() - source.float()) ** 2).mean(dim=2, keepdim=True)  # [B,F,1,H,W]
        denom = diff.amax(dim=(1, 3, 4), keepdim=True).clamp_min(1e-6)
        m = diff / denom
        if self.region_mask_soft:
            return m
        mask = (m > self.region_mask_threshold).to(m.dtype)
        # 可选闭运算（先膨胀后腐蚀）：填补编辑区内部空洞、平滑边界。默认关（kernel <= 1）。
        k = self.region_mask_close_kernel
        if k > 1:
            b, f = mask.shape[:2]
            x = mask.flatten(0, 1)                                       # [B*F, 1, H, W]
            x = F.max_pool2d(x, kernel_size=k, stride=1, padding=k // 2)   # 膨胀
            x = -F.max_pool2d(-x, kernel_size=k, stride=1, padding=k // 2)  # 腐蚀
            mask = x.unflatten(0, (b, f))
        return mask

    def generator_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        clean_latent: torch.Tensor,                 # target latent [B, F, C, H, W]
        unconditional_dict: Optional[dict] = None,
        edit_types: Optional[list] = None,          # per-sample edit type from data (index.json `edit_type`)
    ) -> Tuple[torch.Tensor, dict]:
        b, f = clean_latent.shape[:2]
        clean_latent = clean_latent.to(self.device, self.dtype)
        noise = torch.randn_like(clean_latent)

        # Snapshot the CLEAN source now: the source-noise augmentation below may replace
        # conditional_dict["source_latents"] with a noised copy, but the region mask must
        # be computed from clean source vs clean target.
        src_clean = None
        if self.region_loss or self.ref_attn_loss:
            _sl = conditional_dict.get("source_latents")
            if _sl:
                _s = _sl[0] if isinstance(_sl, list) else _sl
                src_clean = _s.to(self.device, self.dtype)

        index = self._sample_timestep_index(b, f, 0, self.num_train_timestep)
        timestep = self.scheduler.timesteps[index].to(dtype=self.dtype, device=self.device)
        noisy = self.scheduler.add_noise(
            clean_latent.flatten(0, 1), noise.flatten(0, 1), timestep.flatten(0, 1)
        ).unflatten(0, (b, f))
        training_target = self.scheduler.training_target(clean_latent, noise, timestep)

        # optional slight-noise augmentation of the clean teacher-forced context.
        # noise_aug_max_t is a timestep VALUE upper bound (0~1000, 0=off): each frame's
        # context is corrupted at a level sampled in [0, noise_aug_max_t] (0 == clean),
        # and that same level is fed back as aug_t so the model knows the context noise
        # level (matches inference's context_noise). Small value -> slight noise.
        if self.teacher_forcing and self.noise_aug_max_t > 0:
            aug_t = self._sample_timestep_value(b, f, self.noise_aug_max_t)
            clean_ctx = self.scheduler.add_noise(
                clean_latent.flatten(0, 1), noise.flatten(0, 1), aug_t.flatten(0, 1)
            ).unflatten(0, (b, f))
        else:
            aug_t = None
            clean_ctx = clean_latent

        # optional slight-noise augmentation of the SOURCE condition (not the target).
        # Each source frame is corrupted at an independently sampled level in
        # [0, source_noise_max_t] (a timestep value; 0 == clean). This regularises the
        # edit so the model relies less on copying the exact source pixels. Refs and
        # the target/loss are untouched. We shallow-copy conditional_dict so the
        # caller's clean source is not mutated.
        if conditional_dict.get("source_latents"):
            source_inputs = []
            source_timesteps = []

            for s in conditional_dict["source_latents"]:
                s = s.to(self.device, self.dtype)
                sb, sf = s.shape[:2]

                if self.source_noise_max_t > 0:
                    # t_src 始终控制 source latent 的轻噪；模型调制 timestep
                    # 由 source_timestep_mode 在 t_src / target timestep 间选择。
                    t_src = self._sample_timestep_value(
                        sb, sf, self.source_noise_max_t)
                    s_input = self.scheduler.add_noise(
                        s.flatten(0, 1),
                        torch.randn_like(s).flatten(0, 1),
                        t_src.flatten(0, 1),
                    ).unflatten(0, (sb, sf))
                else:
                    # source 不加噪时，对应 clean timestep=0。
                    t_src = torch.zeros(
                        (sb, sf),
                        device=self.device,
                        dtype=self.dtype,
                    )
                    s_input = s

                source_inputs.append(s_input)
                source_timesteps.append(
                    timestep if self.source_timestep_mode == "target" else t_src)

            conditional_dict = {
                **conditional_dict,
                "source_latents": source_inputs,
                "source_timesteps": source_timesteps,
            }

        refs = conditional_dict.get("ref_latents")
        if refs is not None:
            refs = refs if isinstance(refs, (list, tuple)) else [refs]
            conditional_dict = {
                **conditional_dict,
                "ref_timesteps": [self.ref_timestep] * len(refs),
            }

        # Build attention supervision from CLEAN source/target before entering the
        # generator. It is detached data supervision; source noise never changes it.
        if self.ref_attn_loss and self.ref_attn_loss_weight > 0.0 and src_clean is not None \
                and src_clean.shape == clean_latent.shape and conditional_dict.get("ref_latents"):
            ref_mask = self._edit_region_mask(src_clean, clean_latent)
            if edit_types is not None and len(edit_types) == b:
                gate = torch.tensor(
                    [1.0 if str(x or "").strip().lower() in self.ref_attn_loss_types else 0.0
                     for x in edit_types], device=ref_mask.device, dtype=ref_mask.dtype,
                ).view(b, 1, 1, 1, 1)
                ref_mask = ref_mask * gate
            else:
                ref_mask = ref_mask * 0.0
            conditional_dict = {
                **conditional_dict,
                "ref_attn_mask": ref_mask.detach(),
                "ref_attn_config": {
                    "layers": self.ref_attn_loss_layers,
                    "margin": self.ref_attn_loss_margin,
                    "topk": self.ref_attn_topk,
                    "max_queries": self.ref_attn_max_queries,
                    "query_chunk": self.ref_attn_query_chunk,
                },
            }

        generator_out = self.generator(
            noisy_image_or_video=noisy,
            conditional_dict=conditional_dict,
            timestep=timestep,
            clean_x=clean_ctx if self.teacher_forcing else None,
            aug_t=aug_t if self.teacher_forcing else None,
        )
        attn_aux = None
        if len(generator_out) == 3:
            flow_pred, x0_pred, attn_aux = generator_out
        else:
            flow_pred, x0_pred = generator_out

        tw = self.scheduler.training_weight(timestep).unflatten(0, (b, f))    # [B, F]
        per_pos = (flow_pred.float() - training_target.float()) ** 2          # [B, F, C, H, W]

        edit_region_loss = edit_frac = None
        if self.region_loss and src_clean is not None and src_clean.shape == clean_latent.shape:
            mask = self._edit_region_mask(src_clean, clean_latent)           # [B, F, 1, H, W]
            # Every dataset may provide edit_type. Empty means not annotated yet;
            # configured non-empty types may enable region weighting.
            if edit_types is not None and len(edit_types) == b:
                mapping = self.region_loss_by_type or {}
                keys = [str(edit_type or "").strip().lower()
                        for edit_type in edit_types]
                gate = torch.tensor(
                    [
                        1.0
                        if key and (mapping.get(key, False) if mapping else True)
                        else 0.0
                        for key in keys
                    ],
                    device=mask.device, dtype=mask.dtype,
                ).view(b, *([1] * (mask.dim() - 1)))                          # [B,1,1,1,1]
                mask = mask * gate
            w = 1.0 + self.region_edit_weight * mask                         # emphasise edit region
            if self.region_weight_normalize:
                # per-frame renormalise so the mean weight stays 1 -> loss magnitude and
                # effective LR are unchanged, only the spatial emphasis is redistributed.
                w = w / w.mean(dim=(2, 3, 4), keepdim=True).clamp_min(1e-6)
            loss = ((per_pos * w).mean(dim=(2, 3, 4)) * tw).mean()
            # diagnostics: edit-region-only flow MSE + edit-region fraction of the frame.
            # 若本 batch 全是被跳过的全局编辑（mask 全 0），不产出区域诊断（保持 None）。
            if float(mask.sum()) > 0.0:
                msum = mask.sum().clamp_min(1.0)
                edit_region_loss = (per_pos.mean(dim=2, keepdim=True) * mask).sum() / msum
                edit_frac = mask.mean()
        else:
            loss = (per_pos.mean(dim=(2, 3, 4)) * tw).mean()

        log_dict = {"x0_pred": x0_pred.detach(), "timestep": timestep.detach()}
        if edit_region_loss is not None:
            log_dict["edit_region_loss"] = edit_region_loss.detach()
            log_dict["edit_frac"] = edit_frac.detach()
        if attn_aux is not None:
            ref_attn_raw = attn_aux[:, 0].mean()
            loss = loss + self.ref_attn_loss_weight * ref_attn_raw
            log_dict["ref_attn_loss"] = ref_attn_raw.detach()
            log_dict["ref_attn_ref_score"] = attn_aux[:, 1].mean().detach()
            log_dict["ref_attn_src_score"] = attn_aux[:, 2].mean().detach()
            log_dict["ref_attn_delta"] = attn_aux[:, 3].mean().detach()
        return loss, log_dict

    @torch.no_grad()
    def augment_for_debug(self, source_latent=None, target_latent=None):
        """Return the source / TF-context latents *exactly as generator_loss feeds them*
        to the network, i.e. after the source-noise and context-noise augmentations.

        For visual sanity-checking only (decode -> mp4). Pure add_noise ops, no model
        forward / collectives, so it is safe to call on rank0 alone. A fresh noise draw
        is used each call, so the result is a representative sample of the augmentation.
        Returns a dict with optional keys 'source_in' / 'target_ctx_in' ([B,F,C,H,W]).
        """
        out = {}
        if source_latent is not None:
            s = source_latent.to(self.device, self.dtype)
            if self.source_noise_max_t > 0:
                sb, sf = s.shape[:2]
                t_src = self._sample_timestep_value(sb, sf, self.source_noise_max_t)
                s = self.scheduler.add_noise(
                    s.flatten(0, 1), torch.randn_like(s).flatten(0, 1), t_src.flatten(0, 1)
                ).unflatten(0, (sb, sf))
            out["source_in"] = s
        if target_latent is not None:
            c = target_latent.to(self.device, self.dtype)
            if self.teacher_forcing and self.noise_aug_max_t > 0:
                cb, cf = c.shape[:2]
                aug_t = self._sample_timestep_value(cb, cf, self.noise_aug_max_t)
                c = self.scheduler.add_noise(
                    c.flatten(0, 1), torch.randn_like(c).flatten(0, 1), aug_t.flatten(0, 1)
                ).unflatten(0, (cb, cf))
            out["target_ctx_in"] = c
        return out
