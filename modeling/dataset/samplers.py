# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Custom data samplers for distributed training."""

import math

import torch
import torch.distributed as dist
from torch.utils.data import Sampler as TorchSampler


class BalancedDistributedSampler(TorchSampler):
    """Distributed sampler that draws roughly equal numbers of samples from each
    source dataset, regardless of their differing sizes.

    Given a list of datasets (concatenated externally via ConcatDataset), it emits
    global indices into that concatenation. Smaller datasets are oversampled (with
    replacement) and larger ones are subsampled so every dataset contributes the
    same per-epoch count, then indices are sharded across DDP ranks. This prevents
    large datasets from dominating a mixed-corpus training run.
    """
    def __init__(
        self,
        datasets,
        num_replicas=None,
        rank=None,
        shuffle=True,
        seed=0,
        desired_total_size=None,
    ):
        # Fall back to the active process group's world size / rank when not given.
        if num_replicas is None:
            num_replicas = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        if rank is None:
            rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0

        self.datasets = list(datasets)
        self.lengths = [len(d) for d in self.datasets]
        if any(l <= 0 for l in self.lengths):
            raise ValueError(f"All datasets must be non-empty, got lengths={self.lengths}")

        self.num_datasets = len(self.datasets)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)

        # Starting global index of each dataset within the ConcatDataset, so local
        # per-dataset indices can be mapped back to the concatenated index space.
        offsets = [0]
        for l in self.lengths[:-1]:
            offsets.append(offsets[-1] + int(l))
        self.offsets = offsets

        # Target total samples across all datasets for one epoch (before rank sharding).
        if desired_total_size is None:
            desired_total_size = int(sum(self.lengths))
        self.desired_total_size = int(desired_total_size)

        # Equal quota per dataset, then round the grand total up to a multiple of
        # num_replicas so every rank gets exactly num_samples indices.
        self.samples_per_dataset = int(math.ceil(self.desired_total_size / float(self.num_datasets)))
        base_total = self.samples_per_dataset * self.num_datasets

        self.total_size = int(math.ceil(base_total / float(self.num_replicas)) * self.num_replicas)
        self.num_samples = self.total_size // self.num_replicas

    def __iter__(self):
        # Seed derived from base seed + epoch so every rank produces the *same*
        # global ordering this epoch (sharding below picks disjoint slices).
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        # Step 1: pick exactly samples_per_dataset global indices from each dataset,
        # subsampling large datasets and oversampling small ones with replacement.
        per_dataset = []
        for ds_idx, ds_len in enumerate(self.lengths):
            if self.shuffle:
                base = torch.randperm(ds_len, generator=g).tolist()
            else:
                base = list(range(ds_len))

            if self.samples_per_dataset <= ds_len:
                chosen = base[: self.samples_per_dataset]
            else:
                remaining = self.samples_per_dataset - ds_len
                extra = torch.randint(high=ds_len, size=(remaining,), generator=g).tolist() if remaining > 0 else []
                chosen = base + extra

            per_dataset.append([i + self.offsets[ds_idx] for i in chosen])

        # Step 2: interleave datasets round-robin so batches mix sources evenly.
        indices = []
        for j in range(self.samples_per_dataset):
            for i in range(self.num_datasets):
                indices.append(per_dataset[i][j])

        # Step 3: optionally shuffle the interleaved list, then trim/pad to total_size.
        if self.shuffle:
            perm = torch.randperm(len(indices), generator=g).tolist()
            indices = [indices[k] for k in perm]

        indices = indices[: self.desired_total_size]

        if len(indices) < self.total_size:
            indices += indices[: (self.total_size - len(indices))]

        # Step 4: shard — each rank takes a strided slice so ranks don't overlap.
        indices = indices[self.rank : self.total_size : self.num_replicas]
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        # Must be called each epoch so the seed (and thus the ordering) changes.
        self.epoch = int(epoch)
