# S2D2: Simple Stable Diffusion based on Diffusers
Diffusers-based simple generating image module for jupyter notebook or python commandline

## Features
- ☑ Just prepare safetensors files to go (Fast loading)
- ☑ Run Hires.fix without [AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui)
  - ☑ Latent Upscaler
  - ☒ GAN models
- ☑ Multi-level upscaling (extension of Hires.fix)
- ☑ LoRA
- ☒ Controlnet
- ☒ Multi-batch generation (Only single generation is supported)

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
image.save("generate_image.jpg) # or just "image" for display image in jupyter
```

### Correspondence of web ui and parameters
![image](https://github.com/keisuke-okb/S2D2/assets/70097451/0389a313-f000-42ed-a549-ebcb5aeb8eec)

### Parameters
🚧🚧🚧🚧🚧Under construction🚧🚧🚧🚧🚧


# Generated sample images
- Used [Counterfeit-V30.safetensors](https://huggingface.co/gsdf/Counterfeit-V3.0/tree/main)
- Initial resolution: 696x496
- Upscale factor: 1.8
- Target resolution: 696x496(x1.8, nearest multiple of 8) = 1248x888

### Multi-scale upscaling(Ex.4)
Stepwise upscaling between the initial resolution and the target resolution.
![image](https://github.com/keisuke-okb/S2D2/assets/70097451/2f2bb59b-405d-4214-9af5-ac5264cd1d54)

### 2-scale upscaling(Hires.fix)
![image](https://github.com/keisuke-okb/S2D2/assets/70097451/8af64b21-1c32-40f6-be87-618859a2bf1f)


# References
- [AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui)
- [ddPn08/Radiata](https://github.com/ddPn08/Radiata)
- [huggingface/diffusers/blob/main/scripts/convert_lora_safetensor_to_diffusers.py](https://github.com/huggingface/diffusers/blob/main/scripts/convert_lora_safetensor_to_diffusers.py)
