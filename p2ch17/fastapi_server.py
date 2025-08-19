from fastapi import FastAPI
from pydantic import BaseModel
import torch

class TextInput(BaseModel):
    text: str

app = FastAPI()

from transformers import AutoModelForCausalLM, AutoTokenizer

model = None
tokenizer = None
def get_model_and_tokenizer(device):
    global model, tokenizer
    checkpoint = "HuggingFaceTB/SmolLM2-360M-Instruct"
    if model is None:
        model = AutoModelForCausalLM.from_pretrained(checkpoint)
        model = model.to(device)
        model.eval()
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, padding_side="left")
    print(f"Memory footprint: {model.get_memory_footprint() / 1e6:.2f} MB")
    return model, tokenizer

@app.post("/generate")
def generate_text(input_txt: TextInput = TextInput(text="What is PyTorch and why is it cool?")):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = get_model_and_tokenizer(device)
    messages = [{"role": "user", "content": input_txt.text}]
    input_text=tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(inputs, max_new_tokens=256, do_sample=True)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text