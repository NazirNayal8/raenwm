"""Dataset components: the trajectory datasets, the balanced distributed sampler, and
typed configs + factories for building them.

Physical per-dataset constants (action_stats, metric_waypoint_spacing) remain in the
standalone ``config/data_config.yaml`` loaded by ``modeling.utils.misc`` /
``training_dataset``; they're calibration data, not experiment config.
"""
from dataclasses import dataclass, field
from typing import Optional

from modeling.dataset.samplers import BalancedDistributedSampler
from modeling.dataset.training_dataset import (
    BaseDataset,
    TrainingDataset,
    EvalDataset,
    TrajectoryEvalDataset,
)


@dataclass(kw_only=True)
class DistanceCfg:
    min_dist_cat: int
    max_dist_cat: int


@dataclass(kw_only=True)
class DatasetSourceCfg:
    """One named data source. ``train``/``test`` are split folder paths (either may be
    absent). Per-source ``distance`` / ``len_traj_pred`` override the DataCfg defaults."""
    data_folder: str
    goals_per_obs: int
    train: Optional[str] = None
    test: Optional[str] = None
    distance: Optional[DistanceCfg] = None
    len_traj_pred: Optional[int] = None


@dataclass(kw_only=True)
class DataCfg:
    """Training data config: global defaults + a dict of named sources."""
    image_size: int
    context_size: int
    normalize: bool
    len_traj_pred: int
    distance: DistanceCfg
    sources: dict[str, DatasetSourceCfg]
    traj_stride: int = 1
    balanced_sampling: bool = True
    balanced_sampling_mode: str = "downsample_only"


@dataclass(kw_only=True)
class EvalDistanceCfg:
    eval_min_dist_cat: int
    eval_max_dist_cat: int


@dataclass(kw_only=True)
class EvalDataCfg:
    """Eval/inference data config (mirrors config/eval_config.yaml).

    ``context_size`` is exposed as an alias of ``eval_context_size`` so the generic
    ``wire_denoiser_to_tokenizer`` (which reads ``.image_size`` + ``.context_size``)
    works with this cfg too.
    """
    image_size: int
    normalize: bool
    eval_context_size: int
    eval_len_traj_pred: int
    eval_distance: EvalDistanceCfg
    eval_datasets: dict[str, DatasetSourceCfg]
    trajectory_eval_context_size: Optional[int] = None
    trajectory_eval_len_traj_pred: Optional[int] = None
    trajectory_eval_distance: Optional[DistanceCfg] = None
    traj_stride: int = 1

    @property
    def context_size(self) -> int:
        return self.eval_context_size


def get_eval_dataset(cfg: EvalDataCfg, dataset_name, eval_type, predefined_index=False):
    """Build an EvalDataset for one source (ports infer.py's get_dataset_eval)."""
    from modeling.utils.misc import transform

    src = cfg.eval_datasets[dataset_name]
    idx = f"data_splits/{dataset_name}/test/{eval_type}.pkl" if predefined_index else None
    return EvalDataset(
        data_folder=src.data_folder,
        data_split_folder=src.test,
        dataset_name=dataset_name,
        image_size=cfg.image_size,
        min_dist_cat=cfg.eval_distance.eval_min_dist_cat,
        max_dist_cat=cfg.eval_distance.eval_max_dist_cat,
        len_traj_pred=cfg.eval_len_traj_pred,
        traj_stride=cfg.traj_stride,
        context_size=cfg.eval_context_size,
        normalize=cfg.normalize,
        transform=transform,
        goals_per_obs=4,
        predefined_index=idx,
        traj_names='traj_names.txt',
    )


def get_datasets(cfg: DataCfg):
    """Build (train_datasets, train_dataset, test_dataset) from a typed DataCfg.

    Ports train.py's ``build_datasets``: per-source overrides fall back to global
    defaults; the test split standardizes goals_per_obs to 4.
    """
    from torch.utils.data import ConcatDataset
    from modeling.utils.misc import transform

    train_datasets, test_datasets = [], []
    for name, src in cfg.sources.items():
        for split in ("train", "test"):
            split_path = getattr(src, split)
            if not split_path:
                continue
            goals_per_obs = 4 if split == "test" else int(src.goals_per_obs)
            dist = src.distance or cfg.distance
            len_traj_pred = src.len_traj_pred if src.len_traj_pred is not None else cfg.len_traj_pred
            ds = TrainingDataset(
                data_folder=src.data_folder,
                data_split_folder=split_path,
                dataset_name=name,
                image_size=cfg.image_size,
                min_dist_cat=dist.min_dist_cat,
                max_dist_cat=dist.max_dist_cat,
                len_traj_pred=len_traj_pred,
                context_size=cfg.context_size,
                normalize=cfg.normalize,
                goals_per_obs=goals_per_obs,
                transform=transform,
                predefined_index=None,
                traj_stride=cfg.traj_stride,
            )
            (train_datasets if split == "train" else test_datasets).append(ds)
            print(f"Dataset: {name} ({split}), size: {len(ds)}")

    print(f"Combining {len(train_datasets)} datasets.")
    return train_datasets, ConcatDataset(train_datasets), ConcatDataset(test_datasets)


def build_sampler(cfg: DataCfg, train_datasets, train_dataset, rank, world_size, seed):
    """Choose the training sampler (balanced across sources, or plain distributed).

    Ports train.py's ``build_sampler``; ``world_size`` is passed in (rather than read
    from the process group) so the factory is decoupled from DDP init.
    """
    from torch.utils.data.distributed import DistributedSampler

    balanced = bool(cfg.balanced_sampling) and len(train_datasets) > 1
    if balanced:
        mode = str(cfg.balanced_sampling_mode).strip().lower()
        min_len = min(len(d) for d in train_datasets)
        desired = len(train_datasets) * min_len if mode == "downsample_only" else len(train_dataset)
        return BalancedDistributedSampler(
            train_datasets,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=seed,
            desired_total_size=desired,
        )
    return DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=seed,
    )


__all__ = [
    "BaseDataset",
    "TrainingDataset",
    "EvalDataset",
    "TrajectoryEvalDataset",
    "BalancedDistributedSampler",
    "DistanceCfg",
    "DatasetSourceCfg",
    "DataCfg",
    "EvalDistanceCfg",
    "EvalDataCfg",
    "get_datasets",
    "build_sampler",
    "get_eval_dataset",
]
