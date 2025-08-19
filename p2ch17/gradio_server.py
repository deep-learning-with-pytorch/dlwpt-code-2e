import gradio as gr

import torch
from diffusers import DiffusionPipeline

image_generator = None
def load_image_generator():
    global image_generator
    if image_generator is None:
        model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
        pipeline = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipeline.to(device)

        pipeline.enable_vae_tiling()
        pipeline.enable_sequential_cpu_offload()
        
        image_generator = pipeline
    return image_generator
    
def generate_image(prompt):
    image_generator = load_image_generator()
    image = image_generator(prompt).images[0]
    return image

demo = gr.Interface(
    fn=generate_image,
    inputs=gr.Textbox(label="Prompt"),
    outputs=gr.Image(type="pil", label="Generated Image"),
    flagging_mode="auto",
)

demo.launch()