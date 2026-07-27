# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Checkpoint save/restore and EMA weight tracking."""

import os
from collections import OrderedDict

import torch


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.

    ema = decay * ema + (1 - decay) * model, applied in-place per parameter.
    The '_orig_mod.' prefix (added by torch.compile) is stripped so compiled and
    uncompiled parameter names match.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace('_orig_mod.', '')
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def resume_from_checkpoint(checkpoint_dir, from_checkpoint, device, model, ema, opt, scaler,
                           base_lr, override_lr_on_resume):
    """Restore training state from an existing checkpoint, if one is available.

    Prefers an in-progress run's latest.pth.tar; otherwise falls back to the explicit
    ``from_checkpoint`` path (may be None). Refuses to do both at once, since that
    would risk clobbering the existing latest.pth.tar. Restores model + EMA weights,
    optimizer state (optionally forcing the LR back to ``base_lr``), and the AMP
    scaler.

    Returns ``(start_epoch, train_steps, checkpoint_dict)``. When nothing is
    resumed, ``checkpoint_dict`` is None and the counters are zero.
    """
    latest_path = os.path.join(checkpoint_dir, "latest.pth.tar")
    print('Searching for model from ', checkpoint_dir)
    start_epoch = 0
    train_steps = 0

    if not (os.path.isfile(latest_path) or from_checkpoint):
        return start_epoch, train_steps, None

    if os.path.isfile(latest_path) and from_checkpoint:
        raise ValueError("Resuming from checkpoint, this might override latest.pth.tar!!")

    resume_path = latest_path if os.path.isfile(latest_path) else from_checkpoint
    print("Loading model from ", resume_path)
    ckpt = torch.load(resume_path, map_location=device, weights_only=False)

    # Restore model + EMA weights (stripping the torch.compile prefix). If the
    # checkpoint predates EMA storage, seed EMA from the loaded model instead.
    if "model" in ckpt:
        model_ckp = {k.replace('_orig_mod.', ''): v for k, v in ckpt['model'].items()}
        res = model.load_state_dict(model_ckp, strict=True)
        print("Loading model weights", res)

        model_ckp = {k.replace('_orig_mod.', ''): v for k, v in ckpt['ema'].items()}
        res = ema.load_state_dict(model_ckp, strict=True)
        print("Loading EMA model weights", res)
    else:
        update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights

    # Restore optimizer state. Optionally force the LR back to the config value
    # (useful when changing the schedule on resume rather than continuing the old one).
    if "opt" in ckpt:
        opt_ckp = {k.replace('_orig_mod.', ''): v for k, v in ckpt['opt'].items()}
        opt.load_state_dict(opt_ckp)
        print("Loading optimizer params")

        if override_lr_on_resume:
            for pg in opt.param_groups:
                pg["lr"] = base_lr
            opt.defaults["lr"] = base_lr
            print(f"Override optimizer lr on resume: lr={base_lr}")

    # Restore training progress counters and the AMP scaler if present.
    if "epoch" in ckpt:
        start_epoch = ckpt['epoch'] + 1

    if "train_steps" in ckpt:
        train_steps = ckpt["train_steps"]

    if "scaler" in ckpt and scaler is not None:
        scaler.load_state_dict(ckpt["scaler"])

    return start_epoch, train_steps, ckpt


def save_checkpoint_with_step(model, ema, opt, args, epoch, train_steps, checkpoint_dir, scaler=None, scheduler=None):
    """Save a full training checkpoint (model, EMA, optimizer, and optional scaler/
    scheduler) to both a step-tagged file and latest.pth.tar for resume."""
    checkpoint = {
        "model": model.module.state_dict(),
        "ema": ema.state_dict(),
        "opt": opt.state_dict(),
        "args": args,
        "epoch": epoch,
        "train_steps": train_steps,
    }

    if scaler is not None:
        checkpoint["scaler"] = scaler.state_dict()

    if scheduler is not None:
        checkpoint["scheduler"] = scheduler.state_dict()

    step_checkpoint_path = f"{checkpoint_dir}/checkpoint_step_{train_steps}.pth.tar"
    torch.save(checkpoint, step_checkpoint_path)
    latest_checkpoint_path = f"{checkpoint_dir}/latest.pth.tar"
    torch.save(checkpoint, latest_checkpoint_path)

    return step_checkpoint_path
