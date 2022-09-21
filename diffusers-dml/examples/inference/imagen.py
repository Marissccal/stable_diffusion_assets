from torch import autocast
import requests
import torch
from PIL import Image
from io import BytesIO

from diffusers import StableDiffusionImg2ImgPipeline


# load the pipeline
#lms = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear")
pipe = StableDiffusionImg2ImgPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", revision="fp16", 
     use_auth_token=True)
torch.manual_seed(42)

# let's download an initial image
url = "https://raw.githubusercontent.com/Marissccal/stable_diffusion_assets/main/assets/king.jpg"

response = requests.get(url)
init_image = Image.open(BytesIO(response.content)).convert("RGB")
init_image = init_image.resize((768, 512))

prompt = "Going king under the mountain of Erebor, photorealistic, trending artstation"

image = pipe(prompt, init_image=init_image, strength=0.75, guidance_scale=10)["sample"][0]
image.save("king.png")

