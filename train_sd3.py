import os
import copy
import shutil
import math
import random
import argparse
import datetime
import subprocess

from pathlib import Path
from tqdm.auto import tqdm
from einops import rearrange
from omegaconf import OmegaConf
from safetensors import safe_open
from typing import Dict, Optional, Tuple

import torch
import torchvision
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import diffusers
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    SD3Transformer2DModel,
    StableDiffusion3Pipeline,
)

import transformers
from transformers import (
    CLIPTokenizer,
    CLIPTextModelWithProjection,
    PretrainedConfig, 
    T5EncoderModel, 
    T5TokenizerFast
)

from irontorch import distributed as dist
from irontorch.utils import set_seed, GradScaler
from irontorch.recorder import Logger

from animatediff.data.niul import WebVid10M
from animatediff.utils.util import save_videos_grid
from animatediff.models.transformer import SD3Transformer3DModel
from animatediff.pipelines.pipeline_animation import AnimationPipeline



def _encode_prompt_with_clip(
    text_encoder,
    tokenizer,
    prompt: str,
    device=None,
    num_images_per_prompt: int = 1,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    )

    text_input_ids = text_inputs.input_ids
    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)

    pooled_prompt_embeds = prompt_embeds[0]
    prompt_embeds = prompt_embeds.hidden_states[-2]
    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape
    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds, pooled_prompt_embeds


def _encode_prompt_with_t5(
    text_encoder,
    tokenizer,
    max_sequence_length,
    prompt=None,
    num_images_per_prompt=1,
    device=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    prompt_embeds = text_encoder(text_input_ids.to(device))[0]

    dtype = text_encoder.dtype
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape

    # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds


def encode_prompt(
    text_encoders,
    tokenizers,
    prompt: str,
    max_sequence_length: int = 77,
    device=None,
    num_images_per_prompt: int = 1,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt

    clip_tokenizers = tokenizers[:2]
    clip_text_encoders = text_encoders[:2]

    clip_prompt_embeds_list = []
    clip_pooled_prompt_embeds_list = []
    for tokenizer, text_encoder in zip(clip_tokenizers, clip_text_encoders):
        prompt_embeds, pooled_prompt_embeds = _encode_prompt_with_clip(
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device if device is not None else text_encoder.device,
            num_images_per_prompt=num_images_per_prompt,
        )
        clip_prompt_embeds_list.append(prompt_embeds)
        clip_pooled_prompt_embeds_list.append(pooled_prompt_embeds)

    clip_prompt_embeds = torch.cat(clip_prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = torch.cat(clip_pooled_prompt_embeds_list, dim=-1)

    t5_prompt_embed = _encode_prompt_with_t5(
        text_encoders[-1],
        tokenizers[-1],
        max_sequence_length,
        prompt=prompt,
        num_images_per_prompt=num_images_per_prompt,
        device=device if device is not None else text_encoders[-1].device,
    )

    clip_prompt_embeds = torch.nn.functional.pad(
        clip_prompt_embeds, (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1])
    )
    prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)

    return prompt_embeds, pooled_prompt_embeds


def get_sigmas(noise_scheduler, timesteps, n_dim=4, dtype=torch.float32):
    sigmas = noise_scheduler.sigmas.to(device=timesteps.device, dtype=dtype)
    schedule_timesteps = noise_scheduler.timesteps.to(timesteps.device)
    # timesteps = timesteps.to(accelerator.device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma


def main(conf):
    check_min_version("0.31.0.dev0")

    # Initialize distributed training
    local_rank      = dist.get_local_rank()
    global_rank     = dist.get_rank()
    num_processes   = dist.get_world_size()

    set_seed(conf.global_seed + global_rank)
    
    # Logging folder
    timestamp = datetime.datetime.now().strftime("-%Y-%m-%dT%H-%M-%S")
    dir_name = "debug" if conf.is_debug else conf.name + timestamp
    output_dir = os.path.join(conf.output_dir, dir_name)
    if conf.is_debug and os.path.exists(output_dir):
        os.system(f"rm -rf {output_dir}")

    # Handle the output folder creation
    if dist.is_primary():
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/samples", exist_ok=True)
        os.makedirs(f"{output_dir}/sanity_check", exist_ok=True)
        os.makedirs(f"{output_dir}/checkpoints", exist_ok=True)
        shutil.copy(conf.config_path, output_dir)
    logger = Logger(save_dir=output_dir, name=conf.name, rank=dist.get_rank(), mode='rich')

    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        conf.pretrained_model_path, subfolder="scheduler"
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)

    vae = AutoencoderKL.from_pretrained(
        conf.pretrained_model_path,
        subfolder="vae",
    )

    # Load the tokenizers
    tokenizer_one = CLIPTokenizer.from_pretrained(
        conf.pretrained_model_path,
        subfolder="tokenizer",
    )
    tokenizer_two = CLIPTokenizer.from_pretrained(
        conf.pretrained_model_path,
        subfolder="tokenizer_2",
    )
    tokenizer_three = T5TokenizerFast.from_pretrained(
        conf.pretrained_model_path,
        subfolder="tokenizer_3",
    )

    text_encoder_one = CLIPTextModelWithProjection.from_pretrained(
        conf.pretrained_model_path, subfolder="text_encoder"
    )
    text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
        conf.pretrained_model_path, subfolder="text_encoder_2"
    )
    text_encoder_three = T5EncoderModel.from_pretrained(
        conf.pretrained_model_path, subfolder="text_encoder_3"
    )

    if conf.image_finetune:
        transformer = SD3Transformer2DModel.from_pretrained(
            conf.pretrained_model_path, subfolder="transformer"
        )
    else:
        transformer = SD3Transformer3DModel.from_pretrained_2d(
            conf.pretrained_model_path, subfolder="transformer",
            transformer_additional_kwargs=conf.transformer_additional_kwargs
            )
        
    if conf.transformer_checkpoint_path != "":
        logger.info(f"from checkpoint: {conf.transformer_checkpoint_path}")
        transformer_checkpoint = torch.load(conf.transformer_checkpoint_path, map_location="cpu")
        if "global_step" in transformer_checkpoint: 
            logger.info(f"global_step: {transformer_checkpoint['global_step']}")
        state_dict = transformer_checkpoint["state_dict"] if "state_dict" in transformer_checkpoint else transformer_checkpoint

        m, u = transformer.load_state_dict(state_dict, strict=False)
        logger.info(f"missing keys: {len(m)}, unexpected keys: {len(u)}")
        assert len(u) == 0
        
    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    text_encoder_three.requires_grad_(False)

    transformer.requires_grad_(False)
    for name, param in transformer.named_parameters():
        for trainable_module_name in conf.trainable_modules:
            if trainable_module_name in name:
                param.requires_grad = True
                break
            
    trainable_params = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=conf.learning_rate,
        betas=(conf.adam_beta1, conf.adam_beta2),
        weight_decay=conf.adam_weight_decay,
        eps=conf.adam_epsilon,
    )

    logger.info(f"trainable params number: {len(trainable_params)}")
    logger.info(f"trainable params scale: {sum(p.numel() for p in trainable_params) / 1e6:.3f} M")

    # Enable gradient checkpointing
    if conf.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    # Move models to GPU
    vae.to(local_rank)
    text_encoder_one.to(local_rank)
    text_encoder_two.to(local_rank)
    text_encoder_three.to(local_rank)

    # Get the training dataset
    train_dataset = WebVid10M(
            **conf.train_data,
            is_image=conf.image_finetune, 
            use_condition=False
    )
    train_sampler = dist.get_data_sampler(
            train_dataset, 
            shuffle=True, 
            distributed=conf.distributed
    )

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=conf.batch_size,
        shuffle=False,
        sampler=train_sampler,
        num_workers=conf.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # Get the training iteration
    if conf.max_train_steps == -1:
        assert conf.max_train_epoch != -1
        conf.max_train_steps = conf.max_train_epoch * len(train_dataloader)
        
    if conf.checkpointing_steps == -1:
        assert conf.checkpointing_epochs != -1
        conf.checkpointing_steps = conf.checkpointing_epochs * len(train_dataloader)

    # Scheduler
    lr_scheduler = get_scheduler(
        conf.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=conf.lr_warmup_steps * conf.gradient_accumulation_steps,
        num_training_steps=conf.max_train_steps * conf.gradient_accumulation_steps,
    )

    # Validation pipeline
    if not conf.image_finetune:
        validation_pipeline = AnimationPipeline(
            vae=vae, 
            tokenizer=tokenizer_one,
            tokenizer_2=tokenizer_two,
            tokenizer_3=tokenizer_three,
            text_encoder=text_encoder_one,
            text_encoder_2=text_encoder_two,
            text_encoder_3=text_encoder_three,
            transformer=transformer,
            scheduler=noise_scheduler,
        ).to("cuda")
    else:
        validation_pipeline = StableDiffusion3Pipeline.from_pretrained(
            conf.pretrained_model_path,
            vae=vae,
            tokenizer=tokenizer_one,
            tokenizer_2=tokenizer_two,
            tokenizer_3=tokenizer_three,
            text_encoder=text_encoder_one,
            text_encoder_2=text_encoder_two,
            text_encoder_3=text_encoder_three,
            transformer=transformer,
        ).to("cuda")

    # samples = []
    
    # generator = torch.Generator(device="cuda")
    # generator.manual_seed(conf.global_seed)
    
    # height = conf.train_data.sample_size[0] if not isinstance(conf.train_data.sample_size, int) else conf.train_data.sample_size
    # width  = conf.train_data.sample_size[1] if not isinstance(conf.train_data.sample_size, int) else conf.train_data.sample_size

    # prompts = conf.validation_data.prompts

    # for idx, prompt in enumerate(prompts):
    #     if not conf.image_finetune:
    #         sample = validation_pipeline(
    #             prompt,
    #             generator    = generator,
    #             video_length = conf.train_data.sample_n_frames,
    #             height       = height,
    #             width        = width,
    #             **conf.validation_data,
    #         ).videos
    #         save_videos_grid(sample, f"{output_dir}/samples/sample-{global_step}/{idx}.gif")
    #         samples.append(sample)
    # import pdb;pdb.set_trace()
    
    # if not conf.image_finetune:
    #     samples = torch.concat(samples)
    #     save_path = f"{output_dir}/samples/sample-{global_step}.gif"
    #     save_videos_grid(samples, save_path)

    # DDP warpper
    transformer.to(local_rank)
    if conf.distributed:
        transformer = DDP(transformer, device_ids=[local_rank], output_device=local_rank)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / conf.gradient_accumulation_steps)
    num_train_epochs = math.ceil(conf.max_train_steps / num_update_steps_per_epoch)
    total_batch_size = conf.batch_size * num_processes * conf.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {conf.batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {conf.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {conf.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, conf.max_train_steps), disable=not dist.is_primary())
    progress_bar.set_description("Steps")

    # Support mixed-precision training
    scaler = GradScaler(conf.mixed_precision)
    # scaler = None

    for epoch in range(first_epoch, num_train_epochs):
        if conf.distributed:
            train_dataloader.sampler.set_epoch(epoch)
        transformer.train()
        
        for step, batch in enumerate(train_dataloader):
            if conf.cfg_random_null_text:
                batch['text'] = [name if random.random() > conf.cfg_random_null_text_ratio else "" for name in batch['text']]
                
            # Data batch sanity check
            if dist.is_primary() and epoch == first_epoch and step == 0:
                pixel_values, texts = batch['pixel_values'].cpu(), batch['text']
                if not conf.image_finetune:
                    pixel_values = rearrange(pixel_values, "b f c h w -> b c f h w")
                    for idx, (pixel_value, text) in enumerate(zip(pixel_values, texts)):
                        pixel_value = pixel_value[None, ...]
                        save_videos_grid(pixel_value, f"{output_dir}/sanity_check/{'-'.join(text.replace('/', '').split()[:10]) if not text == '' else f'{global_rank}-{idx}'}.gif", rescale=True)
                else:
                    for idx, (pixel_value, text) in enumerate(zip(pixel_values, texts)):
                        pixel_value = pixel_value / 2. + 0.5
                        torchvision.utils.save_image(pixel_value, f"{output_dir}/sanity_check/{'-'.join(text.replace('/', '').split()[:10]) if not text == '' else f'{global_rank}-{idx}'}.png")
                    
            ### >>>> Training >>>> ###
            
            # Convert videos to latent space            
            pixel_values = batch["pixel_values"].to(local_rank)
            video_length = pixel_values.shape[1]
            with torch.no_grad():
                if not conf.image_finetune:
                    pixel_values = rearrange(pixel_values, "b f c h w -> (b f) c h w")
                    latents = vae.encode(pixel_values).latent_dist
                    latents = latents.sample()
                    latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
                else:
                    latents = vae.encode(pixel_values).latent_dist
                    latents = latents.sample()
                latents = (latents - vae.config.shift_factor) * vae.config.scaling_factor

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            
            u = compute_density_for_timestep_sampling(
                weighting_scheme=conf.weighting_scheme,
                batch_size=bsz,
                logit_mean=conf.logit_mean,
                logit_std=conf.logit_std,
                mode_scale=conf.mode_scale,
            )
            indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
            timesteps = noise_scheduler_copy.timesteps[indices].to(device=latents.device)

            sigmas = get_sigmas(
                    noise_scheduler_copy, 
                    timesteps, 
                    n_dim=latents.ndim, 
                    dtype=latents.dtype
            )
            noisy_latents = (1.0 - sigmas) * latents + sigmas * noise
            
            # Get the text embedding for conditioning
            tokenizers = [tokenizer_one, tokenizer_two, tokenizer_three]
            text_encoders = [text_encoder_one, text_encoder_two, text_encoder_three]
            with torch.no_grad():
                prompt_embeds, pooled_prompt_embeds = encode_prompt(
                    text_encoders, tokenizers, batch['text'], device=latents.device
                )
                prompt_embeds = prompt_embeds.to(latents.device)
                pooled_prompt_embeds = pooled_prompt_embeds.to(latents.device)

            # Predict the noise residual and compute loss
            # Mixed-precision training
            with torch.amp.autocast("cuda", enabled=conf.mixed_precision):
                model_pred = transformer(
                    hidden_states=noisy_latents,
                    timestep=timesteps,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    return_dict=False,
                )[0]

                if conf.precondition_outputs:
                    model_pred = model_pred * (-sigmas) + noisy_latents

                # these weighting schemes use a uniform timestep sampling
                # and instead post-weight the loss
                weighting = compute_loss_weighting_for_sd3(weighting_scheme=conf.weighting_scheme, sigmas=sigmas)

                # flow matching loss
                if conf.precondition_outputs:
                    target = latents
                else:
                    target = noise - latents 

                # Compute regular loss.
                loss = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                    1,
                )
                loss = loss.mean()
            
            scaler(loss, optimizer, transformer.parameters(), clip_grad=conf.max_grad_norm)
            # if scaler is not None:
            #     scaler(loss, optimizer, transformer.parameters(), clip_grad=conf.max_grad_norm)
            # else:
            #     optimizer.zero_grad()
            #     loss.backward()
            #     torch.nn.utils.clip_grad_norm_(transformer.parameters(), conf.max_grad_norm)
            #     optimizer.step()

            lr_scheduler.step()
            progress_bar.update(1)
            global_step += 1
            
            ### <<<< Training <<<< ###
            
            # Save checkpoint
            if dist.is_primary() and global_step % conf.checkpointing_steps == 0:
                save_path = os.path.join(output_dir, f"checkpoints")
                state_dict = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "state_dict": transformer.module.state_dict() if dist.is_parallel(transformer) else transformer.state_dict(),
                }

                if step == len(train_dataloader) - 1:
                    torch.save(state_dict, os.path.join(save_path, f"checkpoint-epoch-{epoch+1}.ckpt"))
                else:
                    torch.save(state_dict, os.path.join(save_path, f"checkpoint-step-{global_step}.ckpt"))
                logger.info(f"Saved state to {save_path} (global_step: {global_step})")
                
            # Periodically validation
            if dist.is_primary() and (global_step % conf.validation_steps == 0 or global_step in conf.validation_steps_tuple):
                samples = []
                
                generator = torch.Generator(device=latents.device)
                generator.manual_seed(conf.global_seed)
                
                height = conf.train_data.sample_size[0] if not isinstance(conf.train_data.sample_size, int) else conf.train_data.sample_size
                width  = conf.train_data.sample_size[1] if not isinstance(conf.train_data.sample_size, int) else conf.train_data.sample_size

                prompts = conf.validation_data.prompts[:2] if global_step < 1000 and (not conf.image_finetune) else conf.validation_data.prompts

                for idx, prompt in enumerate(prompts):
                    if not conf.image_finetune:
                        sample = validation_pipeline(
                            prompt,
                            generator    = generator,
                            video_length = conf.train_data.sample_n_frames,
                            height       = height,
                            width        = width,
                            **conf.validation_data,
                        ).videos
                        save_videos_grid(sample, f"{output_dir}/samples/sample-{global_step}/{idx}.gif")
                        samples.append(sample)
                        
                    else:
                        sample = validation_pipeline(
                            prompt,
                            generator           = generator,
                            height              = height,
                            width               = width,
                            num_inference_steps = conf.validation_data.get("num_inference_steps", 25),
                            guidance_scale      = conf.validation_data.get("guidance_scale", 8.),
                        ).images[0]
                        sample = torchvision.transforms.functional.to_tensor(sample)
                        samples.append(sample)
                
                if not conf.image_finetune:
                    samples = torch.concat(samples)
                    save_path = f"{output_dir}/samples/sample-{global_step}.gif"
                    save_videos_grid(samples, save_path)
                    
                else:
                    samples = torch.stack(samples)
                    save_path = f"{output_dir}/samples/sample-{global_step}.png"
                    torchvision.utils.save_image(samples, save_path, nrow=4)

                logger.info(f"Saved samples to {save_path}")
                
            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            
            if global_step >= conf.max_train_steps:
                break
            


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path",  type=str, required=True)

    conf = dist.setup_config(parser)
    conf.distributed = conf.n_gpu > 1
    dist.run(main, conf.launch_config.nproc_per_node, conf=conf)
