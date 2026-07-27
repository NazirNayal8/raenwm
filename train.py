# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# NoMaD, GNM, ViNT: https://github.com/robodhruv/visualnav-transformer
# --------------------------------------------------------

import torch
# Enable TF32 matmul/conv kernels. Was False during original testing, but True
# gives a large speedup on A100-class GPUs at a small precision cost.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import os, sys

# Make the vendored RAE package importable regardless of the current working dir.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RAE_SRC = os.path.join(PROJECT_ROOT, "RAE", "src")
if RAE_SRC not in sys.path:
    sys.path.append(RAE_SRC)
import matplotlib
matplotlib.use('Agg')  # Headless backend: render eval figures to files, never to a display.
from copy import deepcopy
from time import time
import logging
import random
import matplotlib.pyplot as plt
import numpy as np
import wandb

import hydra
from omegaconf import DictConfig, OmegaConf

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
# Disable the (slow) math SDPA fallback so attention uses flash / mem-efficient kernels.
torch.backends.cuda.enable_math_sdp(False)

from modeling.config import load_typed_root_config, TrainRootCfg
from modeling.models import get_tokenizer, get_denoiser, wire_denoiser_to_tokenizer
from modeling.transport import build_transport, make_transport_sampler, build_sample_fn
from modeling.dataset import get_datasets, build_sampler
from modeling.utils.checkpoint import update_ema, resume_from_checkpoint, save_checkpoint_with_step
from modeling.utils.diagnostics import maybe_print_flash_attn_status_once
from modeling.utils.distributed import init_distributed

#################################################################################
#                             Training Helper Functions                         #
#################################################################################

def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def build_scheduler(scheduler_cfg, opt, base_lr, total_steps, train_steps, override_lr_on_resume, checkpoint, logger):
    """Build the LR scheduler and, when resuming, bring it to the resumed step.

    Supports cosine annealing or a linear decay from base_lr to final_lr over the
    full run. On resume: if overriding the LR we re-derive from config; otherwise
    prefer the saved scheduler state and fall back to fast-forwarding step-by-step
    for older checkpoints without scheduler state.
    """
    final_lr = float(scheduler_cfg.final_lr)
    lr_schedule = str(scheduler_cfg.lr_schedule or 'linear').strip().lower()

    if lr_schedule in {"cosine", "cos", "cosineannealing"}:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps, eta_min=final_lr)
    elif lr_schedule in {"linear", "lambda"}:
        def lr_lambda(step):
            if total_steps <= 0:
                return 1.0
            alpha = step / float(total_steps)
            return (final_lr / base_lr) * alpha + (1 - alpha)
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    else:
        raise ValueError(f"Unknown lr_schedule={lr_schedule}. Use 'linear' or 'cosine'.")

    if train_steps > 0:
        if override_lr_on_resume:
            scheduler.step(max(int(train_steps) - 1, 0))
            logger.info(
                f"Scheduler re-derived from config at resumed step={train_steps}, current lr: {scheduler.get_last_lr()[0]:.6f}"
            )
        elif checkpoint is not None and "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
            logger.info(f"Scheduler state restored from checkpoint, current lr: {scheduler.get_last_lr()[0]:.6f}")
        else:
            for _ in range(train_steps):
                scheduler.step()
            logger.info(f"Scheduler fast-forwarded to step {train_steps}, current lr: {scheduler.get_last_lr()[0]:.6f}")

    return scheduler


#################################################################################
#                                  Training Loop                                #
#################################################################################

@hydra.main(version_base=None, config_path="config", config_name="train")
def main(cfg_dict: DictConfig):
    """
    Trains a new CDiT model. Config is composed by hydra (see config/train.yaml) and
    parsed into a typed ``TrainRootCfg``.
    """
    cfg: TrainRootCfg = load_typed_root_config(cfg_dict, TrainRootCfg)

    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP: one process per GPU. Each rank gets a distinct seed so data
    # augmentation / dropout differ across ranks.
    _, rank, gpu, _ = init_distributed()
    device = torch.device(f"cuda:{gpu}")
    seed = cfg.seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup an experiment folder:
    os.makedirs(cfg.output_dir, exist_ok=True)  # Holds all experiment subfolders
    experiment_dir = f"{cfg.output_dir}/{cfg.run_name}"
    checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
    if rank == 0:
        # Only rank 0 creates dirs, logs, and owns the wandb run.
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")

        if cfg.wandb.enabled:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = cfg.wandb.run_name or f"{cfg.run_name}_{timestamp}"
            wandb.init(
                project=cfg.wandb.project,
                name=run_name,
                entity=cfg.wandb.entity,
                config=OmegaConf.to_container(cfg_dict, resolve=True),
                dir=experiment_dir,
                tags=list(cfg.wandb.tags),
                notes=cfg.wandb.notes,
            )
            logger.info(f"WandB initialized successfully with run name: {run_name}")
    else:
        logger = create_logger(None)

    # Create the frozen stage-1 RAE (DINOv2-based tokenizer). It stays in eval mode
    # and is used only to encode pixels -> latents (and decode back during eval);
    # the diffusion model trains in this latent space.
    tokenizer = get_tokenizer(cfg.model.tokenizer).to(device).eval()
    latent_size = tokenizer.latent_spatial_size(cfg.data.image_size)
    num_cond = cfg.data.context_size  # number of context (conditioning) frames

    # Wire the denoiser to the tokenizer geometry (in_channels / input_size / context)
    # then build the conditional DiT.
    wire_denoiser_to_tokenizer(cfg.model.denoiser, tokenizer, cfg.data)
    model = get_denoiser(cfg.model.denoiser).to(device)
    if rank == 0:
        logger.info(
            f"tokenizer latent_dim={tokenizer.latent_dim}, latent_size={latent_size}, "
            f"learn_sigma={cfg.model.denoiser.learn_sigma}"
        )
        logger.info(
            f"CDiT in_channels={model.in_channels}, out_channels={model.out_channels}, "
            f"patch_size={model.patch_size}, head_width={getattr(model, 'head_width', 'NA')}"
        )
    debug_shapes = bool(cfg.train.debug_shapes)

    # EMA copy of the weights: updated every step, used for evaluation / final model.
    ema = deepcopy(model).to(device)
    requires_grad(ema, False)  # EMA is never optimized directly.

    # Optimizer.
    base_lr = float(cfg.optimizer.lr)
    betas = tuple(cfg.optimizer.betas)
    override_lr_on_resume = bool(cfg.optimizer.override_lr_on_resume)
    opt = torch.optim.AdamW(model.parameters(), lr=base_lr, betas=betas, weight_decay=cfg.optimizer.weight_decay)

    # bf16 mixed precision toggle (paired with a GradScaler below).
    bfloat_enable = cfg.train.mixed_precision == "bf16"

    # Introspect the model's attention shape for the flash-attn probe; fall back to
    # reasonable defaults if the denoiser can't report it.
    try:
        num_heads, head_dim, seqlen = model.attention_shape()
    except Exception:
        num_heads, head_dim, seqlen = 8, 64, 256

    probe_dtype = torch.bfloat16 if bfloat_enable else torch.float16
    maybe_print_flash_attn_status_once(
        device=device,
        dtype=probe_dtype,
        num_heads=num_heads,
        head_dim=head_dim,
        seqlen=seqlen,
        rank=rank,
    )

    scaler = torch.amp.GradScaler() if bfloat_enable else None

    # Restore model/EMA/optimizer/scaler state if a checkpoint exists. The returned
    # checkpoint dict is reused below to restore scheduler state without reloading.
    start_epoch, train_steps, latest_checkpoint = resume_from_checkpoint(
        checkpoint_dir, cfg.train.from_checkpoint, device, model, ema, opt, scaler,
        base_lr, override_lr_on_resume,
    )

    # Optionally JIT-compile the model (~40% speedup, but can regress quality on
    # some torch versions), then wrap in DDP for multi-GPU gradient sync.
    if cfg.train.compile:
        model = torch.compile(model)
    model = DDP(model, device_ids=[device], find_unused_parameters=True)
    try:
        if debug_shapes:
            (model.module if hasattr(model, "module") else model).debug_shapes = True
    except Exception:
        pass

    # Transport defines the flow-matching / diffusion training objective and paths.
    transport = build_transport(cfg.model.transport, tokenizer.latent_dim, latent_size)
    sampler_transport = make_transport_sampler(transport)  # ODE sampler wrapper, used in eval.
    logger.info(f"CDiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Build datasets, sampler, and loader.
    train_datasets, train_dataset, test_dataset = get_datasets(cfg.data)
    sampler = build_sampler(cfg.data, train_datasets, train_dataset, rank, dist.get_world_size(), cfg.seed)
    loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=cfg.train.num_workers > 0,
    )
    logger.info(f"Dataset contains {len(train_dataset):,} images")

    # Build the LR scheduler (and fast-forward it if resuming). max_steps, when set,
    # bounds the run (and the schedule horizon) instead of max_epochs.
    max_steps = cfg.train.max_steps
    total_steps = max_steps if max_steps else cfg.train.max_epochs * len(loader)
    scheduler = build_scheduler(
        cfg.scheduler, opt, base_lr, total_steps, train_steps, override_lr_on_resume, latest_checkpoint, logger
    )

    # Prepare models for training:
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    grad_clip_val = float(cfg.train.grad_clip_val)
    cfg_container = OmegaConf.to_container(cfg_dict, resolve=True)  # stored in checkpoints

    # Variables for monitoring/logging purposes:
    log_steps = 0
    running_loss = 0
    start_time = time()

    logger.info(f"Training for {cfg.train.max_epochs} epochs...")
    for epoch in range(start_epoch, cfg.train.max_epochs):
        sampler.set_epoch(epoch)  # reshuffle for this epoch (see sampler.__iter__)
        logger.info(f"Beginning epoch {epoch}...")

        for batch in loader:
            # Batch is (frames, actions, rel_time[, paths]); paths are unused here.
            x, y, rel_t = batch[:3]
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            rel_t = rel_t.to(device, non_blocking=True)

            with torch.amp.autocast('cuda', enabled=bfloat_enable, dtype=torch.bfloat16):
                # Encode every frame to latents with the frozen RAE. x arrives in
                # [-1,1]; map to [0,1] pixels for the tokenizer. Shape: (B, T, ...).
                with torch.no_grad():
                    B, T = x.shape[:2]
                    x_flat = x.flatten(0, 1)
                    x_pix = x_flat * 0.5 + 0.5
                    x_latent = tokenizer.encode(x_pix)
                    x = x_latent.unflatten(0, (B, T))

                # Split the sequence: first num_cond frames are context, the rest are
                # prediction targets ("goals"). Each goal is paired with the same
                # context block, then batch/goal dims are flattened for the model.
                num_goals = T - num_cond
                x_start = x[:, num_cond:].flatten(0, 1)
                x_cond = x[:, :num_cond].unsqueeze(1).expand(B, num_goals, num_cond, x.shape[2], x.shape[3], x.shape[4]).flatten(0, 1)
                y = y.flatten(0, 1)
                rel_t = rel_t.flatten(0, 1)

                transport_kwargs = dict(y=y, x_cond=x_cond, rel_t=rel_t)

                # Flow-matching / diffusion loss: transport samples a timestep, adds
                # noise to x_start, and asks the model to predict the target velocity.
                transport_terms = transport.training_losses(
                    model,
                    x_start,
                    transport_kwargs,
                )
                loss = transport_terms["loss"].mean()

            # Backward + optimizer step, with optional grad clipping. The bf16 path
            # routes through the GradScaler (unscale before clipping).
            opt.zero_grad()
            if not bfloat_enable:
                loss.backward()
                if grad_clip_val and grad_clip_val > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_val)
                opt.step()
            else:
                scaler.scale(loss).backward()
                if grad_clip_val and grad_clip_val > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_val)
                scaler.step(opt)
                scaler.update()

            # Track the EMA weights and advance the LR schedule once per step.
            update_ema(ema, model.module)
            scheduler.step()

            # Log loss values:
            running_loss += loss.detach().item()
            log_steps += 1
            train_steps += 1
            if train_steps % cfg.train.log_every == 0:
                # Measure training speed (synchronize so timing isn't skewed by async CUDA):
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                samples_per_sec = dist.get_world_size() * x_cond.shape[0] * steps_per_sec
                # Average the loss across all ranks for a global training-loss number.
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}, Samples/Sec: {samples_per_sec:.2f}")

                # Log to wandb
                if rank == 0 and wandb.run is not None:
                    log_dict = {
                        'metrics/training_loss': avg_loss,
                        'performance/steps_per_second': steps_per_sec,
                        'performance/samples_per_second': samples_per_sec,
                        'training/epoch': epoch,
                        'training/step': train_steps,
                        'training/learning_rate': scheduler.get_last_lr()[0],
                    }
                    wandb.log(log_dict, step=train_steps)

                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Periodically checkpoint (rank 0 only) — writes both a step-tagged file
            # and latest.pth.tar for resume.
            if train_steps % cfg.train.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint_path = save_checkpoint_with_step(
                        model, ema, opt, cfg_container, epoch, train_steps, checkpoint_dir,
                        scaler if bfloat_enable else None, scheduler,
                    )
                    logger.info(f"Saved checkpoint to {checkpoint_path}")

            # Periodic evaluation: run the EMA model through the ODE sampler on the
            # test set and log a perceptual (DreamSim) similarity score.
            if train_steps % cfg.train.eval_every == 0 and train_steps > 0:
                eval_start_time = time()
                save_dir = os.path.join(experiment_dir, str(train_steps))
                sim_score = evaluate(
                    ema,
                    tokenizer,
                    sampler_transport,
                    test_dataset,
                    rank,
                    cfg.train.batch_size,
                    cfg.train.num_workers,
                    latent_size,
                    device,
                    save_dir,
                    cfg.seed + train_steps,
                    bfloat_enable,
                    num_cond,
                    cfg.eval.sampling,
                    max_batches=int(cfg.eval.num_batches),
                )
                dist.barrier()
                eval_end_time = time()
                eval_time = eval_end_time - eval_start_time
                logger.info(f"(step={train_steps:07d}) Perceptual Loss: {sim_score:.4f}, Eval Time: {eval_time:.2f}")

                # Log evaluation results to wandb
                if rank == 0 and wandb.run is not None:
                    wandb.log({
                        'eval/perceptual_loss': sim_score,
                        'eval/eval_time': eval_time,
                    }, step=train_steps)

            # Bounded run: stop after max_steps optimizer steps if configured.
            if max_steps and train_steps >= max_steps:
                break

        if max_steps and train_steps >= max_steps:
            break

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    cleanup()  # tear down the DDP process group


@torch.no_grad()
def evaluate(model, rae, sampler_transport, test_dataloaders, rank, batch_size, num_workers, latent_size, device, save_dir, seed, bfloat_enable, num_cond, sampling_cfg, max_batches=1):
    """Sample predictions from the (EMA) model and score them against ground truth.

    For a few test batches: encode frames to latents, run the ODE sampler
    conditioned on the context frames + actions to generate the goal frame, decode
    back to pixels, save a few visual triptychs (context | target | prediction),
    and accumulate a DreamSim perceptual similarity score reduced across ranks.
    """
    # Distributed sampler so each rank evaluates a disjoint shard of the test set.
    sampler = DistributedSampler(
        test_dataloaders,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=seed
    )

    # Seed the loader and each worker so evaluation is reproducible run-to-run.
    g = torch.Generator()
    g.manual_seed(int(seed))

    def seed_worker(worker_id):
        worker_seed = (int(seed) + int(worker_id)) % (2**32)
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    loader = DataLoader(
        test_dataloaders,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=g,
    )
    # DreamSim is the perceptual metric used to compare prediction vs. ground truth.
    from dreamsim import dreamsim
    eval_model, _ = dreamsim(pretrained=True)
    eval_model.eval()
    eval_model = eval_model.to(device)
    score = torch.tensor(0.).to(device)     # running sum of per-sample distances
    n_samples = torch.tensor(0).to(device)  # running count for the average

    if rank == 0:
        os.makedirs(save_dir, exist_ok=True)
    saved = 0  # number of visualization images written so far (capped at 10)

    for batch_idx, batch in enumerate(loader):
        # Evaluate at most max_batches to keep periodic eval cheap.
        if max_batches is not None and int(max_batches) > 0 and batch_idx >= int(max_batches):
            break
        if isinstance(batch, (list, tuple)) and len(batch) >= 3:
            x, y, rel_t = batch[:3]
        else:
            raise ValueError(f"Unexpected eval batch format: {type(batch)} with len={len(batch) if isinstance(batch, (list, tuple)) else 'NA'}")
        x = x.to(device)
        y = y.to(device)
        rel_t = rel_t.to(device).flatten(0, 1)
        with torch.amp.autocast('cuda', enabled=bfloat_enable, dtype=torch.bfloat16):
            B, T = x.shape[:2]
            num_goals = T - num_cond
            if num_goals <= 0:
                # Defensive fallback for short sequences that have no goal frames.
                num_goals = 1
                if rank == 0:
                    logging.getLogger(__name__).warning(
                        f"Eval fallback: T={T}, num_cond={num_cond} => no goals; using num_goals=1"
                    )
            B_flat = B * num_goals
            # Encode frames to latents (same [-1,1] -> [0,1] convention as training).
            x_flat = x.flatten(0, 1)
            x_pix = x_flat * 0.5 + 0.5
            x_latent = rae.encode(x_pix)
            x_latent = x_latent.unflatten(0, (B, T))
            # Start the reverse ODE from pure Gaussian noise in latent space.
            init = torch.randn(B_flat, rae.latent_dim, latent_size, latent_size, device=device)

            # Build the ODE sampling function from the typed sampling config.
            sample_fn = build_sample_fn(sampler_transport, sampling_cfg)
            # Same context-broadcast layout as training so each goal has its context.
            x_cond = x_latent[:, :num_cond].unsqueeze(1).expand(B, num_goals, num_cond, *x_latent.shape[2:]).flatten(0, 1)
            y_target = y.flatten(0, 1)
            rel_t_target = rel_t
            # Unwrap DDP; temporarily disable intermediate-feature returns during
            # sampling (restored in finally) so the sampler gets plain predictions.
            mobj = model.module if hasattr(model, "module") else model
            rif_prev = getattr(mobj, "return_intermediate_features", False)
            try:
                mobj.set_return_intermediate_features(False)
                xs = sample_fn(init, mobj, y=y_target, x_cond=x_cond, rel_t=rel_t_target)
            finally:
                try:
                    mobj.set_return_intermediate_features(rif_prev)
                except Exception:
                    pass
            samples_latent = xs[-1]  # final ODE state = generated latent
            xs = None

            # Decode generated latents back to pixels; sanitize NaNs and clamp to [0,1].
            samples = rae.decode(samples_latent).float()
            samples = torch.nan_to_num(samples)
            samples = samples.clamp(0, 1)

            # Decode the context latents too (for the visualization panels below).
            x_cond_pixels = rae.decode(x_cond.flatten(0, 1)).float()
            x_cond_pixels = torch.nan_to_num(x_cond_pixels).clamp(0, 1).unflatten(0, (B_flat, num_cond))
            # Keep the *original* (undecoded) pixels for context and target frames,
            # so visualizations show true inputs rather than RAE round-trips.
            x_pix_full = x_pix.unflatten(0, (B, T))
            x_cond_raw_pixels = x_pix_full[:, :num_cond].unsqueeze(1).expand(B, num_goals, num_cond, *x_pix_full.shape[2:]).flatten(0, 1)
            if T - num_cond > 0:
                x_start_raw_pixels = x_pix_full[:, num_cond:].flatten(0, 1)
            else:
                x_start_raw_pixels = x_pix_full[:, -1]

            # If sampling diverged (non-finite latents), fall back to the last context frame.
            if not torch.isfinite(samples_latent).all():
                samples = x_cond_pixels[:, -1]
            # Ground-truth goal frames decoded through the RAE (the scoring reference).
            if T - num_cond > 0:
                x_start_latent = x_latent[:, num_cond:].flatten(0, 1)
                x_start_pixels = rae.decode(x_start_latent).float()
            else:
                x_start_latent = x_latent[:, -1]
                x_start_pixels = rae.decode(x_start_latent).float()
            x_start_pixels = torch.nan_to_num(x_start_pixels).clamp(0, 1)

            # Save up to 10 example triptychs: [last context | ground-truth goal | prediction].
            if rank == 0 and saved < 10:
                n_to_save = min(int(samples.shape[0]), 10 - saved)
                for i in range(n_to_save):
                    _, ax = plt.subplots(1, 3, dpi=256)
                    def to_uint8_image(img3ch):
                        img = img3ch.detach().float().clamp(0.0, 1.0)
                        return (img.permute(1, 2, 0).cpu().numpy() * 255.0).astype('uint8')
                    ax[0].imshow(to_uint8_image(x_cond_raw_pixels[i, -1]))
                    ax[0].axis('off')
                    ax[1].imshow(to_uint8_image(x_start_raw_pixels[i]))
                    ax[1].axis('off')
                    ax[2].imshow(to_uint8_image(samples[i]))
                    ax[2].axis('off')
                    plt.tight_layout()
                    plt.savefig(f'{save_dir}/{saved}.png')
                    plt.close()
                    saved += 1

            # Perceptual distance between ground-truth and predicted goal frames.
            res = eval_model(x_start_pixels, samples)
            score += res.sum()
            n_samples += len(res)

    # Aggregate score/count across all ranks to get a global mean.
    dist.all_reduce(score)
    dist.all_reduce(n_samples)
    sim_score = score / n_samples
    # Free the DreamSim model and reclaim GPU memory before returning to training.
    try:
        del eval_model
    except Exception:
        pass
    try:
        import gc
        gc.collect()
        torch.cuda.empty_cache()
    except Exception:
        pass
    return sim_score


if __name__ == "__main__":
    main()
