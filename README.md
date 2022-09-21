### Image-to-Image text-guided generation with Stable Diffusion

The `StableDiffusionImg2ImgPipeline` lets you pass a text prompt and an initial image to condition the generation of new images.

```python
from torch import autocast
import requests
import torch
from PIL import Image
from io import BytesIO

from diffusers import StableDiffusionImg2ImgPipeline

# load the pipeline
device = "cuda"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    revision="fp16", 
    torch_dtype=torch.float16,
    use_auth_token=True
)
pipe = pipe.to(device)

# let's download an initial image
url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"

response = requests.get(url)
init_image = Image.open(BytesIO(response.content)).convert("RGB")
init_image = init_image.resize((768, 512))

prompt = "A fantasy landscape, trending on artstation"

with autocast("cuda"):
    images = pipe(prompt=prompt, init_image=init_image, strength=0.75, guidance_scale=7.5)["sample"]

images[0].save("fantasy_landscape.png")
```
You can also run this example on colab [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/patil-suraj/Notebooks/blob/master/image_2_image_using_diffusers.ipynb)

### In-painting using Stable Diffusion

The `StableDiffusionInpaintPipeline` lets you edit specific parts of an image by providing a mask and text prompt.

```python
from io import BytesIO

from torch import autocast
import torch
import requests
import PIL

from diffusers import StableDiffusionInpaintPipeline

def download_image(url):
    response = requests.get(url)
    return PIL.Image.open(BytesIO(response.content)).convert("RGB")

img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"

init_image = download_image(img_url).resize((512, 512))
mask_image = download_image(mask_url).resize((512, 512))

device = "cuda"
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    revision="fp16", 
    torch_dtype=torch.float16,
    use_auth_token=True
)
pipe = pipe.to(device)

prompt = "a cat sitting on a bench"
with autocast("cuda"):
    images = pipe(prompt=prompt, init_image=init_image, mask_image=mask_image, strength=0.75)["sample"]

images[0].save("cat_on_bench.png")
```

### Running Code

If you want to run the code yourself 💻, you can try out:
- [Text-to-Image Latent Diffusion](https://huggingface.co/CompVis/ldm-text2im-large-256)
```python
# !pip install diffusers transformers
from diffusers import DiffusionPipeline

model_id = "CompVis/ldm-text2im-large-256"

# load model and scheduler
ldm = DiffusionPipeline.from_pretrained(model_id)

# run pipeline in inference (sample random noise and denoise)
prompt = "A painting of a squirrel eating a burger"
images = ldm([prompt], num_inference_steps=50, eta=0.3, guidance_scale=6)["sample"]

# save images
for idx, image in enumerate(images):
    image.save(f"squirrel-{idx}.png")
```
- [Unconditional Diffusion with discrete scheduler](https://huggingface.co/google/ddpm-celebahq-256)
```python
# !pip install diffusers
from diffusers import DDPMPipeline, DDIMPipeline, PNDMPipeline

model_id = "google/ddpm-celebahq-256"

# load model and scheduler
ddpm = DDPMPipeline.from_pretrained(model_id)  # you can replace DDPMPipeline with DDIMPipeline or PNDMPipeline for faster inference

# run pipeline in inference (sample random noise and denoise)
image = ddpm()["sample"]

# save image
image[0].save("ddpm_generated_image.png")
```
- [Unconditional Latent Diffusion](https://huggingface.co/CompVis/ldm-celebahq-256)
- [Unconditional Diffusion with continous scheduler](https://huggingface.co/google/ncsnpp-ffhq-1024)


