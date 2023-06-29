import os
import random
import datetime
import torch
from PIL import Image

import diffusers
from diffusers import (StableDiffusionPipeline, 
                       StableDiffusionImg2ImgPipeline)
from diffusers.utils import numpy_to_pil
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput

import torch
import datetime
from PIL import Image
import numpy as np

from lora import load_safetensors_lora

SCHEDULERS = {
    "unipc": diffusers.schedulers.UniPCMultistepScheduler,
    "euler_a": diffusers.schedulers.EulerAncestralDiscreteScheduler,
    "euler": diffusers.schedulers.EulerDiscreteScheduler,
    "ddim": diffusers.schedulers.DDIMScheduler,
    "ddpm": diffusers.schedulers.DDPMScheduler,
    "deis": diffusers.schedulers.DEISMultistepScheduler,
    "dpm2": diffusers.schedulers.KDPM2DiscreteScheduler,
    "dpm2-a": diffusers.schedulers.KDPM2AncestralDiscreteScheduler,
    "dpm++_2s": diffusers.schedulers.DPMSolverSinglestepScheduler,
    "dpm++_2m": diffusers.schedulers.DPMSolverMultistepScheduler,
    "dpm++_2m_karras": diffusers.schedulers.DPMSolverMultistepScheduler,
    "dpm++_sde": diffusers.schedulers.DPMSolverSDEScheduler,
    "dpm++_sde_karras": diffusers.schedulers.DPMSolverSDEScheduler,
    "heun": diffusers.schedulers.HeunDiscreteScheduler,
    "heun_karras": diffusers.schedulers.HeunDiscreteScheduler,
    "lms": diffusers.schedulers.LMSDiscreteScheduler,
    "lms_karras": diffusers.schedulers.LMSDiscreteScheduler,
    "pndm": diffusers.schedulers.PNDMScheduler,
}

def calc_pix_8(x):
    x = int(x)
    return x - x % 8


class StableDiffusionImageGenerator:
    def __init__(
            self,
            sd_model_path: str,
            device: str="cuda",
            dtype: torch.dtype=torch.float16,
            ):
        self.device = torch.device(device)
        self.pipe = StableDiffusionPipeline.from_pretrained(
            sd_model_path,
            torch_dtype=dtype,
        ).to(device)
        self.pipe_i2i = StableDiffusionImg2ImgPipeline.from_pretrained(
            sd_model_path,
            torch_dtype=dtype,
        ).to(device)
        self.pipe.enable_xformers_memory_efficient_attention()
        self.pipe.enable_attention_slicing()
        self.pipe_i2i.enable_xformers_memory_efficient_attention()
        self.pipe_i2i.enable_attention_slicing()
        self.pipe.safety_checker = None
        self.pipe_i2i.safety_checker = None
        return
    
    
    def load_lora(self, safetensor_path, alpha=0.75):
        self.pipe = load_safetensors_lora(self.pipe, safetensor_path, alpha=alpha, device=self.device)
        self.pipe_i2i = load_safetensors_lora(self.pipe_i2i, safetensor_path, alpha=alpha, device=self.device)


    def decode_latents_to_PIL_image(self, latents, decode_factor=0.18215):
        with torch.no_grad():
            latents = 1 / decode_factor * latents
            image = self.pipe.vae.decode(latents).sample
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).float().numpy()
            image = numpy_to_pil(image)
            image = StableDiffusionPipelineOutput(images=image, nsfw_content_detected=None).images[0]
            return image


    def diffusion_from_noise(
            self,
            prompt,
            negative_prompt,
            scheduler_name="dpm++_2m_karras",
            num_inference_steps=20, 
            guidance_scale=9.5,
            width=512,
            height=512,
            output_type="pil",
            decode_factor=0.18215,
            seed=1234,
            save_path=None
            ):

        self.pipe.scheduler = SCHEDULERS[scheduler_name].from_config(self.pipe.scheduler.config)
        self.pipe.scheduler.set_timesteps(num_inference_steps, self.device)
        seed = random.randint(1, 1000000000) if seed == -1 else seed

        with torch.no_grad():
            latents = self.pipe(
                prompt=prompt, 
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps, 
                generator=torch.manual_seed(seed),
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                output_type="latent"
            ).images # 1x4x(W/8)x(H/8)

            if save_path is not None:
                pil_image = self.decode_latents_to_PIL_image(latents, decode_factor)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                pil_image.save(save_path, quality=95)

            if output_type == "latent":
                return latents
            elif output_type == "pil":
                return self.decode_latents_to_PIL_image(latents, decode_factor)
            else:
                raise NotImplementedError()
        
    def diffusion_from_image(
            self,
            prompt,
            negative_prompt,
            image,
            scheduler_name="dpm++_2m_karras",
            num_inference_steps=20,
            denoising_strength=0.58,
            guidance_scale=10,
            output_type="pil",
            decode_factor=0.18215,
            seed=1234,
            save_path=None
            ):

        self.pipe_i2i.scheduler = SCHEDULERS[scheduler_name].from_config(self.pipe_i2i.scheduler.config)
        self.pipe_i2i.scheduler.set_timesteps(num_inference_steps, self.device)
        seed = random.randint(1, 1000000000) if seed == -1 else seed

        with torch.no_grad():
            latents = self.pipe_i2i(
                prompt=prompt, 
                negative_prompt=negative_prompt,
                image=image,
                num_inference_steps=num_inference_steps, 
                strength=denoising_strength,
                generator=torch.manual_seed(seed),
                guidance_scale=guidance_scale,
                output_type="latent"
            ).images # 1x4x(W/8)x(H/8)

            if save_path is not None:
                pil_image = self.decode_latents_to_PIL_image(latents, decode_factor)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                pil_image.save(save_path, quality=95)

            if output_type == "latent":
                return latents
            elif output_type == "pil":
                return self.decode_latents_to_PIL_image(latents, decode_factor)
            else:
                raise NotImplementedError()


    def diffusion_enhance(
            self,
            prompt,
            negative_prompt,
            scheduler_name="dpm++_2m_karras",
            num_inference_steps=20,
            num_inference_steps_enhance=20,
            guidance_scale=10,
            width=512,
            height=512,
            seed=1234,
            upscale_target="latent", # "latent" or "pil"
            interpolate_mode="nearest",
            antialias = True,
            upscale_by=1.8,
            enhance_steps=2, # 2=Hires.fix
            denoising_strength=0.58,
            output_type="pil",
            decode_factor=0.15,
            decode_factor_final=0.18215,
            save_dir="output"
            ):
        
        with torch.no_grad():
            w_init = calc_pix_8(width)
            h_init = calc_pix_8(height)
            w_final = calc_pix_8(w_init * upscale_by)
            h_final = calc_pix_8(h_init * upscale_by)
            resolution_pairs = [(calc_pix_8(x), calc_pix_8(y)) for x, y 
                    in zip(np.linspace(w_init, w_final, enhance_steps),
                            np.linspace(h_init, h_final, enhance_steps))
                    ]
            image = None
            now_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

            if enhance_steps == 1: # Single generation
                image = self.diffusion_from_noise(
                        prompt,
                        negative_prompt,
                        scheduler_name=scheduler_name,
                        num_inference_steps=num_inference_steps, 
                        guidance_scale=guidance_scale,
                        width=w_final,
                        height=h_final,
                        output_type=output_type,
                        decode_factor=decode_factor_final,
                        seed=seed,
                        save_path=os.path.join(save_dir, f"{now_str}.jpg")
                    )
                return image

            
            for i, (w, h) in enumerate(resolution_pairs):

                if image is None: # Step 1: Generate low-quality image
                    image = self.diffusion_from_noise(
                        prompt,
                        negative_prompt,
                        scheduler_name=scheduler_name,
                        num_inference_steps=num_inference_steps, 
                        guidance_scale=guidance_scale,
                        width=w,
                        height=h,
                        output_type=upscale_target,
                        decode_factor=decode_factor,
                        seed=seed,
                        save_path=os.path.join(save_dir, f"{now_str}_{i}.jpg")
                    )
                    continue

                # Step 2: Interpolate latent or image -> PIL image
                if upscale_target == "latent":
                    image = torch.nn.functional.interpolate(
                            image,
                            (h // 8, w // 8),
                            mode=interpolate_mode,
                            antialias=True if antialias and interpolate_mode != "nearest" else False,
                        )
                    image = self.decode_latents_to_PIL_image(image, decode_factor)
                else:
                    image = image.resize((w, h), Image.Resampling.LANCZOS)

                # Step 3: Generate image (i2i) 
                if i < len(resolution_pairs) - 1:
                    image = self.diffusion_from_image(
                        prompt,
                        negative_prompt,
                        image,
                        scheduler_name=scheduler_name,
                        num_inference_steps=int(num_inference_steps_enhance / denoising_strength) + 1, 
                        denoising_strength=denoising_strength,
                        guidance_scale=guidance_scale,
                        output_type=upscale_target,
                        decode_factor=decode_factor,
                        seed=seed,
                        save_path=os.path.join(save_dir, f"{now_str}_{i}.jpg")
                    )

                else: # Final enhance
                    image = self.diffusion_from_image(
                        prompt,
                        negative_prompt,
                        image,
                        scheduler_name=scheduler_name,
                        num_inference_steps=int(num_inference_steps_enhance / denoising_strength) + 1, 
                        denoising_strength=denoising_strength,
                        guidance_scale=guidance_scale,
                        output_type=output_type,
                        decode_factor=decode_factor_final,
                        seed=seed,
                        save_path=os.path.join(save_dir, f"{now_str}_{i}.jpg")
                    )
                    return image
    
