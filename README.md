# S2D2: Simple Stable Diffusion based on Diffusers
Diffusers-based simple generating image module with upscaling features for jupyter notebook, ipython or python interactive shell

## Features
- ☑ Just prepare safetensors files to go
- ☑ Run Hires.fix without [AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui)
  - ☑ Latent Upscaler
  - ☒ GAN models
- ☑ Multi-stage upscaling (extension of Hires.fix)
- ☑ LoRA
- ☒ Controlnet
- ☒ Multi-batch generation (Only single generation is supported)

### Schedule
- Add ControlNet (Convert SD safetensors file to diffusers model files)

# Getting Started
## 1. Install libraries
```bash
pip install -r requirements.txt
```

## 2. Prepare safetensors files of SD models and LoRA(option)
- Only safetensors file is supported.
- Place the files in the directory of your choice.

Ex. 

## 3. Run jupyter notebook
```bash
cd s2d2
jupyter notebook
```

## 4. Import main class and load LoRA(option)
```python
from s2d2 import StableDiffusionImageGenerator
generator = StableDiffusionImageGenerator(
    r"C:\xxx\Counterfeit-V30.safetensors",
)
# Load LoRA (multi files)
generator.load_lora(r"C:\xxx\lora_1.safetensors", alpha=0.2)
generator.load_lora(r"C:\xxx\lora_2.safetensors", alpha=0.15)
```

## 5. Generate image using enhance features(Hires.fix and its extended upscaling)
```python
image = generator.diffusion_enhance(
          prompt,
          negative_prompt,
          scheduler_name="dpm++_2m_karras", # [1]
          num_inference_steps=20, # [2]
          num_inference_steps_enhance=20, # [3]
          guidance_scale=10,  # [4]
          width=700, # [5]
          height=500, # [6]
          seed=-1, # [7]
          upscale_target="latent", # [8] "latent" or "pil". pil mode is temporary implemented.
          interpolate_mode="bicubic", # [9]
          antialias=True, # [10]
          upscale_by=1.8, # [11]
          enhance_steps=2, # [12] 2=Hires.fix
          denoising_strength=0.60, # [13]
          output_type="pil", # [14] "latent" or "pil"
          decode_factor=0.15, # [15] Denominator when decoding latents. Used to adjust the saturation of the image during decoding.
          decode_factor_final=0.18215, # [16] Denominator when decoding final latents.
          )
image.save("generated_image.jpg) # or just "image" to display image in jupyter
```

### Correspondence of web ui and parameters
![image](https://github.com/keisuke-okb/S2D2/assets/70097451/6b37aa08-70f9-4f69-a67a-63ac38a70b81)


### Parameters
🚧🚧🚧🚧🚧Under construction🚧🚧🚧🚧🚧

- Available schedulers are:
```python
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
```



# Generated sample images
- Used [Counterfeit-V30.safetensors](https://huggingface.co/gsdf/Counterfeit-V3.0/tree/main)
- Initial resolution: 696x496
- Upscale factor: 1.8
- Target resolution: 696x496(x1.8, nearest multiple of 8) = 1248x888


### 2-stage upscaling(Hires.fix)
![image](https://github.com/keisuke-okb/S2D2/assets/70097451/c53eb7b6-4878-466e-b769-f44dfdfce7fa)

### N-stage upscaling(Ex.4)
Stepwise upscaling between the initial resolution and the target resolution.
![image](https://github.com/keisuke-okb/S2D2/assets/70097451/fbacf8b8-37e8-41f6-9402-49ada9754522)


# Comparison of generating images without or with latent upscaling
- Without latent upscaling: Single generation@696x496
- With latent upcscaling: 2-stage generation(like Hires.fix, 696x496 to 1248x888)

- Prompt: "1girl, solo, full body, blue eyes, looking at viewer, hairband, bangs, brown hair, long hair, smile, blue eyes, wine-red dress, outdoor, night, moonlight, castle, flowers, garden"
- Negative prompt: "EasyNegative, extra fingers, fewer fingers, bad hands"

![image](https://github.com/keisuke-okb/S2D2/assets/70097451/90632859-07c4-4849-868e-ed7c739c65f1)
![image](https://github.com/keisuke-okb/S2D2/assets/70097451/2bffb9ea-e3c2-417e-8e9e-b3607e246674)
![image](https://github.com/keisuke-okb/S2D2/assets/70097451/5dc21284-fb4f-4687-a833-161b8ef50f95)
![image](https://github.com/keisuke-okb/S2D2/assets/70097451/5b5b7a7a-2352-470c-8293-2e66bdce418a)
![image](https://github.com/keisuke-okb/S2D2/assets/70097451/d2076935-a20b-4b35-8947-b1428532b272)
![image](https://github.com/keisuke-okb/S2D2/assets/70097451/916a6289-7729-4c72-b4c7-2b9541d14f42)
![image](https://github.com/keisuke-okb/S2D2/assets/70097451/bfe37531-7df9-4874-ad6f-9e8283ade6f2)


# References
- [AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui)
- [ddPn08/Radiata](https://github.com/ddPn08/Radiata)
- [huggingface/diffusers/blob/main/scripts/convert_lora_safetensor_to_diffusers.py](https://github.com/huggingface/diffusers/blob/main/scripts/convert_lora_safetensor_to_diffusers.py)
