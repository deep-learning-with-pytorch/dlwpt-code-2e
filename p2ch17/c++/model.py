from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

checkpoint = "HuggingFaceTB/SmolLM2-360M-Instruct"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
vocab_size = len(tokenizer)
num_words = 10
random_indices = torch.randint(low=0, high=vocab_size, size=(num_words,))
random_words = [tokenizer.convert_ids_to_tokens(idx.item()) for idx in random_indices]
random_text = ' '.join(random_words)
input_ids = tokenizer.encode(random_text, return_tensors='pt')
print(input_ids)

# Load the model
model = AutoModelForCausalLM.from_pretrained(checkpoint)

exported_model = torch.export.export(model, args=(input_ids,))
output_path = torch._inductor.aoti_load_package(
    exported_model,
    package_path=os.path.join(os.getcwd(), "llm.pt2"),
)
