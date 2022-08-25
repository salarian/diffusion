import torch
from diffusers import StableDiffusionPipeline
from torch import autocast
from PIL import Image

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

def main():
    # make sure you're logged in with `huggingface-cli login`
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", revision="fp16", torch_dtype=torch.float16, use_auth_token=True)

    num_images = 2
    prompt = ["a photograph of an astronaut riding a horse"] * num_images

    with autocast("cuda"):
        images = pipe(prompt, num_inference_steps=50)["sample"]

    grid = image_grid(images, rows=2, cols=2)
    grid.save('output.jpg')

if __name__ == "__main__":
    main()