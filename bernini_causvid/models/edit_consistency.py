"""Stage 2 (Option B): Causal Consistency Distillation = Causal Forcing++ for editing.

Editing counterpart of `model/naive_consistency.py`. Distils the Stage 1 multi-step
AR editing model into a few-step causal editing model WITHOUT generating ODE pairs:
it only needs the GT (source + edited target) latents.

  * generator     : causal edit student (teacher-forced)            [trainable]
  * generator_ema : EMA of the student, the consistency target       [frozen]
  * teacher       : Stage 1 AR editing model (multi-step)            [frozen]

Loss: consistency between the student's x0 at step t and the EMA's x0 at the next
(less noisy) step t_next reached by one teacher CFG step. The source / reference
latents are carried through `conditional_dict` to every forward.
"""
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.scheduler import FlowMatchScheduler
from utils.wan_wrapper import WanTextEncoder, WanVAEWrapper

from .edit_wrapper import EditDiffusionWrapper
from .ckpt import load_edit_generator_state, report_load_state


class EditNaiveConsistency(nn.Module):
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
        nfpb = getattr(config, "num_frame_per_block", 1)
        tshift = getattr(config, "timestep_shift", 5.0)

        def build(trainable):
            w = EditDiffusionWrapper(model_name=model_name, timestep_shift=tshift,
                                     num_frame_per_block=nfpb, bidirectional=False,
                                     model_path=model_path)
            w.model.requires_grad_(trainable)
            return w

        self.generator = build(True)
        self.generator_ema = build(False)
        self.teacher = build(False)

        ckpt = getattr(config, "generator_ckpt", None)
        ckpt = None if ckpt is None or str(ckpt).lower() in ("", "none", "null") else ckpt
        if ckpt:
            state = load_edit_generator_state(ckpt)
            for w, name in ((self.generator, "generator"),
                            (self.generator_ema, "ema"),
                            (self.teacher, "teacher")):
                report_load_state(w, state, tag=f"EditNaiveConsistency.{name}")
            print(f"[EditNaiveConsistency] initialised generator/ema/teacher from {ckpt}")
        else:
            print("[EditNaiveConsistency] generator_ckpt is empty; using raw Bernini weights from model_path")

        if getattr(config, "gradient_checkpointing", False):
            self.generator.enable_gradient_checkpointing()

        # With precomputed prompt embeds (gen_text_embeds.py) + a cached negative-
        # prompt embed, the umT5-xxl text encoder is dropped from GPU entirely
        # (~11GB resident/gather saved), mirroring Stage-1 AR. The trainer then
        # feeds `prompt_embeds` from the batch and the cached negative embed.
        self.use_cached_text_embeds = bool(getattr(config, "cache_text_embeds", False))
        if self.use_cached_text_embeds:
            self.text_encoder = None
        else:
            self.text_encoder = WanTextEncoder(
                text_encoder_path=text_encoder_path,
                tokenizer_path=tokenizer_path).requires_grad_(False)
        self.vae = WanVAEWrapper(vae_path=vae_path).requires_grad_(False)

        self.guidance_scale = float(getattr(config, "guidance_scale", 3.0))
        self.apg_eta = float(getattr(config, "apg_eta", 0.5))
        raw_thresholds = getattr(config, "apg_norm_thresholds", getattr(config, "apg_norm_threshold", (50.0, 50.0, 50.0)))
        if isinstance(raw_thresholds, (int, float)):
            raw_thresholds = [raw_thresholds] * 3
        self.apg_norm_thresholds = tuple(float(x) for x in raw_thresholds)
        if len(self.apg_norm_thresholds) != 3:
            raise ValueError("apg_norm_thresholds must contain exactly three values")
        self.omega_v = float(getattr(config, "omega_v", 1.25))
        self.omega_i = float(getattr(config, "omega_i", 4.5))
        self.omega_ti = float(getattr(config, "omega_ti", 4.0))
        raw_mode_map = getattr(config, "guidance_mode_by_task_type", None) or {
            "t2v": "t2v",
            "s2v": "v2v_apg",
            "v2v": "v2v_apg",
            "tv2v": "v2v_apg",
            "i2i": "v2v_apg",
            "rv2v": "rv2v_apg",
        }
        self.guidance_mode_by_task_type = {
            str(k).strip().lower(): str(v).strip().lower() for k, v in dict(raw_mode_map).items()
        }
        if "tv2v" not in self.guidance_mode_by_task_type and "v2v" in self.guidance_mode_by_task_type:
            self.guidance_mode_by_task_type["tv2v"] = self.guidance_mode_by_task_type["v2v"]
        valid_modes = {"t2v", "v2v_apg", "rv2v_apg"}
        invalid_modes = sorted(set(self.guidance_mode_by_task_type.values()) - valid_modes)
        if invalid_modes:
            raise ValueError(f"unsupported CD guidance modes: {invalid_modes}")
        self.teacher_forcing = getattr(config, "teacher_forcing", True)
        self.source_timestep_mode = str(
            getattr(config, "source_timestep_mode", "source")
        ).lower()
        if self.source_timestep_mode not in ("source", "target"):
            raise ValueError(
                "source_timestep_mode must be 'source' or 'target'")
        self.ref_timestep = float(getattr(config, "ref_timestep", 0) or 0)
        if abs(self.ref_timestep) > 1e-8:
            raise ValueError(
                f"ref_timestep must be 0 for clean ref time embedding, got {self.ref_timestep}")
        if self.source_timestep_mode != "source":
            raise ValueError(
                "source_timestep_mode must be 'source' (clean source time embedding = 0); "
                f"got {self.source_timestep_mode!r}")
        self.discrete_cd_N = getattr(config, "discrete_cd_N", 48)
        self.scheduler = FlowMatchScheduler(shift=tshift, sigma_min=0.0, extra_one_step=True)
        self.scheduler.set_timesteps(num_inference_steps=self.discrete_cd_N, denoising_strength=1.0)
        self.scheduler.sigmas = self.scheduler.sigmas.to(device)
        self.scheduler.timesteps = self.scheduler.timesteps.to(device)

    @torch.no_grad()
    def update_ema(self, decay: float):
        for p_ema, p in zip(self.generator_ema.parameters(), self.generator.parameters()):
            p_ema.mul_(decay).add_(p.detach(), alpha=1.0 - decay)

    def _condition_at_t(self, conditional_dict, timestep):
        """Attach source time embeddings for one CD model evaluation.

        Source latents stay clean throughout CD. In ``source`` mode their
        embedding is therefore timestep zero. In ``target`` mode each model
        evaluation uses its own target timestep (t for teacher/student and
        t_next for the EMA consistency target).
        """
        cond = dict(conditional_dict)
        sources = conditional_dict.get("source_latents")
        if sources is None:
            cond.pop("source_timesteps", None)
        else:
            if not isinstance(sources, (list, tuple)):
                sources = [sources]

            source_timesteps = []
            for source in sources:
                b, f = source.shape[:2]
                if self.source_timestep_mode == "source":
                    source_t = torch.zeros(
                        (b, f), device=source.device, dtype=source.dtype)
                else:
                    source_t = timestep.to(
                        device=source.device, dtype=source.dtype)
                    if source_t.dim() == 1:
                        source_t = source_t.view(b, 1)
                    if source_t.shape[0] != b:
                        raise ValueError(
                            f"target timestep batch {source_t.shape[0]} != source batch {b}")
                    if source_t.shape[1] == 1 and f > 1:
                        source_t = source_t.expand(b, f)
                    if tuple(source_t.shape) != (b, f):
                        raise ValueError(
                            f"target timestep shape {tuple(source_t.shape)} "
                            f"does not match source frames {(b, f)}")
                source_timesteps.append(source_t)
            cond["source_timesteps"] = source_timesteps

        refs = conditional_dict.get("ref_latents")
        if refs is None:
            cond.pop("ref_timesteps", None)
        else:
            refs = refs if isinstance(refs, (list, tuple)) else [refs]
            cond["ref_timesteps"] = [self.ref_timestep] * len(refs)
        return cond

    def _cond_arg(self, clean):
        return clean if self.teacher_forcing else None

    def _resolve_guidance_mode(self, task_types, batch_size: int) -> str:
        if task_types is None:
            raise ValueError("CD guidance requires batch task_types")
        if len(task_types) != batch_size:
            raise ValueError(
                f"task_types length {len(task_types)} != batch {batch_size}")
        keys = []
        for raw in task_types:
            key = str(raw or "").strip().lower()
            if key == "tv2v":
                key = "v2v"
            if key == "s2v":
                raise NotImplementedError(
                    "s2v guidance is reserved but not implemented in current training scope")
            keys.append(key)
        uniq = set(keys)
        if len(uniq) != 1:
            raise ValueError(
                f"CD batch must be homogeneous in task_type, got {sorted(uniq)}")
        task = keys[0]
        if task not in self.guidance_mode_by_task_type:
            raise ValueError(
                f"no CD guidance_mode for task_type {task!r}; "
                f"configured={sorted(self.guidance_mode_by_task_type)}")
        return self.guidance_mode_by_task_type[task]

    @staticmethod
    def _visual_subset(condition: dict, *, source: bool, refs: bool) -> dict:
        out = dict(condition)
        if not source:
            out.pop("source_latents", None)
            out.pop("source_timesteps", None)
        if not refs:
            out.pop("ref_latents", None)
        return out

    def _apg_delta(
        self,
        pred_cond: torch.Tensor,
        pred_base: torch.Tensor,
        norm_threshold: float,
    ) -> torch.Tensor:
        """Bernini `_normalize_diff` without cross-denoising-step momentum."""
        dims = [-1, -2, -4]
        diff = (pred_cond - pred_base).double()
        if norm_threshold > 0:
            diff_norm = diff.norm(p=2, dim=dims, keepdim=True)
            diff = diff * torch.minimum(
                torch.ones_like(diff_norm),
                norm_threshold / (diff_norm + 1e-12),
            )
        direction = pred_cond.double()
        direction = direction / (
            direction.norm(p=2, dim=dims, keepdim=True) + 1e-12)
        parallel = (diff * direction).sum(dim=dims, keepdim=True) * direction
        orthogonal = diff - parallel
        return (orthogonal + self.apg_eta * parallel).type_as(pred_cond)

    def _apg(self, pred_cond, pred_uncond, scale, norm_threshold):
        return pred_uncond + scale * self._apg_delta(
            pred_cond, pred_uncond, norm_threshold)

    def _apg_chain(self, pred_uncond, preds, scales, norm_thresholds):
        """Bernini chain: each projected delta uses the previous condition."""
        if not (len(preds) == len(scales) == len(norm_thresholds)):
            raise ValueError("APG chain predictions, scales, and thresholds must align")
        result = pred_uncond
        previous = pred_uncond
        for pred_cond, scale, threshold in zip(preds, scales, norm_thresholds):
            result = result + scale * self._apg_delta(
                pred_cond, previous, threshold)
            previous = pred_cond
        return result

    @staticmethod
    def _share_edit_block_mask(source_wrapper, target_wrapper):
        """Reuse an identical causal BlockMask across teacher/student/EMA wrappers."""
        source_module = getattr(source_wrapper, "module", source_wrapper)
        target_module = getattr(target_wrapper, "module", target_wrapper)
        source_model = getattr(source_module, "model", None)
        target_model = getattr(target_module, "model", None)
        if source_model is None or target_model is None:
            raise RuntimeError("cannot access edit backbone for BlockMask sharing")
        mask = getattr(source_model, "_edit_block_mask", None)
        key = getattr(source_model, "_edit_block_mask_key", None)
        if mask is None or key is None:
            raise RuntimeError("source edit backbone has no cached BlockMask to share")
        target_model._edit_block_mask = mask
        target_model._edit_block_mask_key = key

    def generator_loss(
        self,
        conditional_dict: dict,
        unconditional_dict: dict,
        clean_latent: torch.Tensor,                 # target latent [B, F, C, H, W]
    ) -> Tuple[torch.Tensor, dict]:
        clean = clean_latent.to(self.device, self.dtype)
        b, f = clean.shape[:2]
        idx = torch.randint(0, self.discrete_cd_N - 1, (1,), device=self.device).item()
        t = self.scheduler.timesteps[idx]
        t_next = self.scheduler.timesteps[idx + 1]
        timestep = t * torch.ones([b, f], device=self.device, dtype=self.dtype)
        timestep_next = t_next * torch.ones([b, f], device=self.device, dtype=self.dtype)

        noise = torch.randn_like(clean)
        latent_t = self.scheduler.add_noise(
            clean, noise=noise, timestep=t * torch.ones([1], device=self.device)
        ).to(self.dtype)

        conditional_t = self._condition_at_t(conditional_dict, timestep)
        unconditional_t = self._condition_at_t(
            unconditional_dict, timestep)
        conditional_t_next = self._condition_at_t(
            conditional_dict, timestep_next)

        # One type-routed teacher guidance step from t -> t_next. Bernini APG
        # is non-linear, so APG branches are combined in x0 space and converted
        # back to flow before applying the existing Euler update.
        guidance_mode = self._resolve_guidance_mode(
            conditional_dict.get("task_types"), b)
        with torch.no_grad():
            if guidance_mode == "t2v":
                v_cond, _ = self.teacher(
                    latent_t, conditional_t, timestep,
                    clean_x=self._cond_arg(clean))
                v_uncond, _ = self.teacher(
                    latent_t, unconditional_t, timestep,
                    clean_x=self._cond_arg(clean))
                v_pred = v_uncond + self.guidance_scale * (v_cond - v_uncond)
            elif guidance_mode == "v2v_apg":
                _, x0_cond = self.teacher(
                    latent_t, conditional_t, timestep,
                    clean_x=self._cond_arg(clean))
                _, x0_uncond = self.teacher(
                    latent_t, unconditional_t, timestep,
                    clean_x=self._cond_arg(clean))
                x0_guided = self._apg(
                    x0_cond, x0_uncond, self.guidance_scale,
                    self.apg_norm_thresholds[-1])
                v_pred = EditDiffusionWrapper._convert_x0_to_flow_pred(
                    self.scheduler, x0_guided.flatten(0, 1),
                    latent_t.flatten(0, 1), timestep.flatten(0, 1),
                ).unflatten(0, (b, f))
            elif guidance_mode == "rv2v_apg":
                if not conditional_t.get("source_latents"):
                    raise ValueError("rv2v_apg requires source_latents")
                if not conditional_t.get("ref_latents"):
                    raise ValueError("rv2v_apg requires ref_latents")
                cond_0 = self._visual_subset(
                    unconditional_t, source=False, refs=False)
                cond_v = self._visual_subset(
                    unconditional_t, source=True, refs=False)
                cond_vi = self._visual_subset(
                    unconditional_t, source=True, refs=True)
                _, x0_0 = self.teacher(
                    latent_t, cond_0, timestep,
                    clean_x=self._cond_arg(clean))
                _, x0_v = self.teacher(
                    latent_t, cond_v, timestep,
                    clean_x=self._cond_arg(clean))
                _, x0_vi = self.teacher(
                    latent_t, cond_vi, timestep,
                    clean_x=self._cond_arg(clean))
                _, x0_vti = self.teacher(
                    latent_t, conditional_t, timestep,
                    clean_x=self._cond_arg(clean))
                x0_guided = self._apg_chain(
                    x0_0, (x0_v, x0_vi, x0_vti),
                    (self.omega_v, self.omega_i, self.omega_ti),
                    self.apg_norm_thresholds)
                v_pred = EditDiffusionWrapper._convert_x0_to_flow_pred(
                    self.scheduler, x0_guided.flatten(0, 1),
                    latent_t.flatten(0, 1), timestep.flatten(0, 1),
                ).unflatten(0, (b, f))
            else:
                raise RuntimeError(f"unreachable guidance_mode {guidance_mode!r}")

            dt = ((timestep - timestep_next) / 1000.0).reshape(b, f, 1, 1, 1)
            latent_t_next = latent_t - dt * v_pred

        # All three causal wrappers use the same shape/key. Reuse the teacher's
        # sparse BlockMask instead of materializing the same large dense mask again.
        self._share_edit_block_mask(self.teacher, self.generator)
        _, cm_pred_t = self.generator(latent_t, conditional_t, timestep, clean_x=self._cond_arg(clean))
        self._share_edit_block_mask(self.generator, self.generator_ema)
        with torch.no_grad():
            _, cm_pred_t_next = self.generator_ema(
                latent_t_next, conditional_t_next, timestep_next, clean_x=self._cond_arg(clean))

        loss = F.mse_loss(cm_pred_t, cm_pred_t_next.detach(), reduction="mean")
        log_dict = {"t": float(t), "t_next": float(t_next)}
        return loss, log_dict
