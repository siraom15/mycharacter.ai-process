from diffusers import StableDiffusionPipeline
import torch

# Load the Stable Diffusion model
model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", device)

# # Float 16 is used for faster inference (but cannot use with cpu)
# # Use torch.float32 if you want to use float32
# pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)

# pipe.to(device)

# # Define the prompt
# prompt = "a cartoon of a happy cat playing in a garden"

# # Generate the image
# image = pipe(prompt).images[0]

# # Save the image
# image_path = "generated_cartoon.png"
# image.save(image_path)

# print(f"Image saved at {image_path}")
