"""Editing variants of the Causal-Forcing model classes.

Each subclass overrides ONLY `_initialize_models` to swap the plain
`WanDiffusionWrapper` for `EditWanDiffusionWrapper` (which loads the converted
Bernini edit backbone and threads the source/ref condition). All loss math,
timestep sampling and rollout logic are inherited verbatim from the base
classes -- the editing condition flows through `conditional_dict`
(`source_latents` / `ref_latents`) exactly like any other conditioning, so the
training/inference *details* are identical to Causal-Forcing.

The DMD stage additionally overrides the inference-pipeline factory to use the
edit streaming rollout (`EditSelfForcingTrainingPipeline`).
"""
import torch

from model.diffusion import CausalDiffusion
from model.ode_regression import ODERegression
from model.naive_consistency import NaiveConsistency
from model.dmd import DMD
from utils.wan_edit_wrapper import EditWanDiffusionWrapper
from utils.wan_wrapper import WanTextEncoder, WanVAEWrapper


def _edit_kwargs(args):
    return dict(getattr(args, "model_kwargs", {}))


class EditCausalDiffusion(CausalDiffusion):
    def _initialize_models(self, args, device):
        self.generator = EditWanDiffusionWrapper(**_edit_kwargs(args), is_causal=True)
        self.generator.model.requires_grad_(True)
        self.text_encoder = WanTextEncoder()
        self.text_encoder.requires_grad_(False)
        self.vae = WanVAEWrapper()
        self.vae.requires_grad_(False)
        self.scheduler = self.generator.get_scheduler()
        self.scheduler.timesteps = self.scheduler.timesteps.to(device)


class EditODERegression(ODERegression):
    def _initialize_models(self, args, device):
        self.generator = EditWanDiffusionWrapper(**_edit_kwargs(args), is_causal=True)
        self.generator.model.requires_grad_(True)
        self.text_encoder = WanTextEncoder()
        self.text_encoder.requires_grad_(False)
        self.vae = WanVAEWrapper()
        self.vae.requires_grad_(False)
        self.scheduler = self.generator.get_scheduler()
        self.scheduler.timesteps = self.scheduler.timesteps.to(device)


class EditNaiveConsistency(NaiveConsistency):
    def _initialize_models(self, args, device):
        self.generator = EditWanDiffusionWrapper(**_edit_kwargs(args), is_causal=True)
        self.generator.model.requires_grad_(True)
        self.teacher = EditWanDiffusionWrapper(**_edit_kwargs(args), is_causal=True)
        self.teacher.model.requires_grad_(False)
        self.generator_ema = EditWanDiffusionWrapper(**_edit_kwargs(args), is_causal=True)
        self.generator_ema.model.requires_grad_(False)
        self.text_encoder = WanTextEncoder()
        self.text_encoder.requires_grad_(False)
        self.scheduler = self.generator.get_scheduler()
        self.scheduler.timesteps = self.scheduler.timesteps.to(device)


class EditDMD(DMD):
    """DMD distillation for the causal editing student.

    generator / fake_score are causal edit students; real_score is the frozen
    edit teacher. The Stage-3 rollout uses the edit streaming pipeline so the
    source video is streamed as a causal condition exactly like at inference.
    """

    def _initialize_models(self, args, device):
        self.real_model_name = getattr(args, "real_name", "Bernini-R-1.3B")
        self.fake_model_name = getattr(args, "fake_name", "Bernini-R-1.3B")
        self.iscausal = getattr(args, "causal", True)

        self.generator = EditWanDiffusionWrapper(**_edit_kwargs(args), is_causal=self.iscausal)
        self.generator.model.requires_grad_(True)

        # real_score: frozen teacher. fake_score: trainable critic.
        real_kwargs = _edit_kwargs(args)
        real_kwargs.setdefault("model_name", self.real_model_name)
        self.real_score = EditWanDiffusionWrapper(**real_kwargs, is_causal=self.iscausal)
        self.real_score.model.requires_grad_(False)

        fake_kwargs = _edit_kwargs(args)
        fake_kwargs.setdefault("model_name", self.fake_model_name)
        self.fake_score = EditWanDiffusionWrapper(**fake_kwargs, is_causal=self.iscausal)
        self.fake_score.model.requires_grad_(True)

        self.text_encoder = WanTextEncoder()
        self.text_encoder.requires_grad_(False)
        self.vae = WanVAEWrapper()
        self.vae.requires_grad_(False)
        self.scheduler = self.generator.get_scheduler()
        self.scheduler.timesteps = self.scheduler.timesteps.to(device)

    def _initialize_inference_pipeline(self):
        from pipeline.edit_causal_streaming import EditSelfForcingTrainingPipeline
        self.inference_pipeline = EditSelfForcingTrainingPipeline(
            denoising_step_list=self.denoising_step_list,
            scheduler=self.scheduler,
            generator=self.generator,
            num_frame_per_block=self.num_frame_per_block,
            num_transformer_blocks=len(self.generator.model.blocks),
            frame_seq_length=getattr(self.args, "frame_seq_length", 1560),
            local_attn_size=self.generator.model.local_attn_size,
            context_noise=getattr(self.args, "context_noise", 0),
            same_step_across_blocks=getattr(self.args, "same_step_across_blocks", True),
        )
