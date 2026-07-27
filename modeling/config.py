"""Typed config loading: OmegaConf ``DictConfig`` -> dataclass via dacite.

Holds the version-neutral loader machinery plus the composed root/shared config
dataclasses. Component-level configs live with their components
(``modeling.models.{denoisers,tokenizers}``, ``modeling.transport``,
``modeling.dataset``) and are aggregated here.

Discriminated unions (e.g. ``DenoiserCfg``, ``TokenizerCfg``) are resolved by dacite
automatically: each variant's ``name: Literal[...]`` field makes exactly one arm of
the union match the incoming dict.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Type, TypeVar

from omegaconf import DictConfig, OmegaConf
from dacite import Config, from_dict

from modeling.models.denoisers import DenoiserCfg
from modeling.models.tokenizers import TokenizerCfg
from modeling.transport import TransportCfg, SamplingCfg
from modeling.dataset import DataCfg, EvalDataCfg

T = TypeVar("T")

# dacite type hooks: coerce raw scalars/sequences into richer field types during
# from_dict. OmegaConf yields plain lists, so allow dataclass fields typed as tuple.
TYPE_HOOKS = {
    Path: Path,
    tuple: tuple,
}


def load_typed_config(
    cfg: DictConfig,
    data_class: Type[T],
    extra_type_hooks: dict | None = None,
) -> T:
    """Convert an OmegaConf config into a typed dataclass instance."""
    hooks = {**TYPE_HOOKS, **(extra_type_hooks or {})}
    container = OmegaConf.to_container(cfg, resolve=True)
    return from_dict(data_class, container, config=Config(type_hooks=hooks))


def load_typed_root_config(cfg: DictConfig, cfg_type: Type[T]) -> T:
    """Parse a full run's config into its typed root dataclass.

    ``cfg_type`` selects which root (train / infer / planning / probe). Extra keys
    that hydra leaves in the composed config but the dataclass doesn't declare are
    ignored (dacite runs non-strict by default).
    """
    return load_typed_config(cfg, cfg_type)


#################################################################################
#                           Shared top-level configs                            #
#################################################################################

@dataclass(kw_only=True)
class OptimizerCfg:
    lr: float = 2e-4
    betas: List[float] = field(default_factory=lambda: [0.9, 0.95])
    weight_decay: float = 0.0
    override_lr_on_resume: bool = True


@dataclass(kw_only=True)
class SchedulerCfg:
    lr_schedule: str = "linear"     # "linear" | "cosine"
    final_lr: float = 2e-5


@dataclass(kw_only=True)
class WandbCfg:
    enabled: bool = True
    project: str = "raenwm-training"
    entity: Optional[str] = None
    run_name: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    notes: str = ""


@dataclass(kw_only=True)
class TrainCfg:
    batch_size: int
    num_workers: int
    max_epochs: int = 300
    max_steps: Optional[int] = None   # if set, stop after this many optimizer steps (overrides epochs)
    grad_clip_val: float = 1.0
    mixed_precision: Literal["no", "bf16"] = "bf16"
    compile: bool = True
    debug_shapes: bool = False
    from_checkpoint: Optional[str] = None
    log_every: int = 100
    ckpt_every: int = 2000
    eval_every: int = 5000


@dataclass(kw_only=True)
class EvalCfg:
    num_batches: int = 1
    sampling: SamplingCfg = field(default_factory=SamplingCfg)


@dataclass(kw_only=True)
class ModelCfg:
    tokenizer: TokenizerCfg
    denoiser: DenoiserCfg
    transport: TransportCfg


@dataclass(kw_only=True)
class TrainRootCfg:
    output_dir: str
    run_name: str
    train: TrainCfg
    model: ModelCfg
    data: DataCfg
    seed: int = 0
    optimizer: OptimizerCfg = field(default_factory=OptimizerCfg)
    scheduler: SchedulerCfg = field(default_factory=SchedulerCfg)
    eval: EvalCfg = field(default_factory=EvalCfg)
    wandb: WandbCfg = field(default_factory=WandbCfg)


#################################################################################
#                          Inference (infer.py) config                          #
#################################################################################

@dataclass(kw_only=True)
class CheckpointCfg:
    """Resolves the checkpoint file to load. Precedence: explicit path > name > step."""
    path: Optional[str] = None
    name: Optional[str] = None
    step: str = "0100000"
    results_dir: str = "logs"
    run_name: str = "raenwm"

    def resolve(self) -> str:
        if self.path:
            return self.path
        base = f"{self.results_dir}/{self.run_name}/checkpoints"
        if self.name:
            return f"{base}/{self.name}.pth.tar"
        return f"{base}/{self.step}.pth.tar"


@dataclass(kw_only=True)
class InferRunCfg:
    datasets: str                     # comma-separated dataset names
    eval_type: str = "rollout"        # "rollout" | "time"
    batch_size: int = 16
    num_workers: int = 8
    input_fps: int = 4
    num_sec_eval: int = 5
    rollout_fps_values: List[int] = field(default_factory=lambda: [1, 4])
    max_ids: int = 0
    gt: bool = False
    output_dir: str = "results"


@dataclass(kw_only=True)
class InferRootCfg:
    model: ModelCfg
    data: EvalDataCfg
    run: InferRunCfg
    sampling: SamplingCfg = field(default_factory=SamplingCfg)
    checkpoint: CheckpointCfg = field(default_factory=CheckpointCfg)
    seed: int = 0


#################################################################################
#                      Planning (planning_eval.py) config                       #
#################################################################################

@dataclass(kw_only=True)
class PlanningCfg:
    """Scoring + CEM optimization knobs."""
    score_type: Literal["dino", "lpips"] = "dino"
    traj_sampler: Literal["curve", "line"] = "curve"
    num_samples: int = 10
    rollout_stride: int = 1
    topk: int = 5
    opt_steps: int = 15
    num_repeat_eval: int = 1
    prior_mix: float = 0.0
    backtrack_allow: float = 0.0
    prior_beta: float = 0.8
    plot: bool = False
    plot_topn: int = 10


@dataclass(kw_only=True)
class PlanningRunCfg:
    datasets: str
    output_dir: str = "results_plan"
    batch_size: int = 16
    num_workers: int = 8
    save_preds: bool = False
    subset_items: int = 0
    subset_seed: int = 0
    run_tag: str = ""


@dataclass(kw_only=True)
class PlanningRootCfg:
    model: ModelCfg
    data: EvalDataCfg
    run: PlanningRunCfg
    planning: PlanningCfg = field(default_factory=PlanningCfg)
    checkpoint: CheckpointCfg = field(default_factory=CheckpointCfg)
    seed: int = 0


#################################################################################
#                      Probe (train_probe.py) config                            #
#################################################################################

@dataclass(kw_only=True)
class ProbeTrainCfg:
    batch_size: int = 64
    num_workers: int = 8
    eval_num_batches: int = 1
    # Optional overrides of the pose_probe step cadence (None -> use pose_probe.*).
    epochs: Optional[int] = None
    log_every: Optional[int] = None
    ckpt_every: Optional[int] = None
    eval_every: Optional[int] = None
    bfloat16: bool = True
    torch_compile: bool = False
    resume: str = ""
    exp_dir: str = ""
    test_only: bool = False
    only_spatial_random_test: bool = False
    test_action_modes: str = ""
    action_random_mode: str = "shuffle"


@dataclass(kw_only=True)
class ProbeRootCfg:
    """Root config for train_probe.py. ``encoder`` and ``pose_probe`` are passed
    through as plain dicts (the probe code reads them with ``.get()`` defaults, and
    the encoder is polymorphic), while structural fields are typed."""
    output_dir: str
    run_name: str
    data: DataCfg
    encoder: Dict[str, Any] = field(default_factory=dict)
    pose_probe: Dict[str, Any] = field(default_factory=dict)
    train: ProbeTrainCfg = field(default_factory=ProbeTrainCfg)
    wandb: WandbCfg = field(default_factory=WandbCfg)
    seed: int = 0
    config_path: str = "RAE/configs/stage1/pretrained/DINOv2-B.yaml"
