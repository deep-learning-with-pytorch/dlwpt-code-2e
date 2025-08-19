from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import torch
from typing import List, Dict, Tuple, AsyncGenerator
from transformers import AutoModelForCausalLM, AutoTokenizer
import asyncio
import uuid
import time
from threading import Thread, Lock, Event
import queue
import torch.nn.functional as F
from contextlib import asynccontextmanager

shutdown_event = Event()
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # On Startup
    worker_thread = Thread(target=model_worker, daemon=True)
    worker_thread.start()
    yield
    # On Shutdown
    shutdown_event.set()

class TextInput(BaseModel):
    text: str

app = FastAPI(lifespan=lifespan)

model = None
tokenizer = None

# Configuration
MAX_BATCH_SIZE = 4  # Maximum number of requests to process in a batch
BATCH_WAIT_TIME = 3  # Time to wait to collect a batch (seconds)
MAX_TOKENS = 400     # Maximum tokens to generate per request

def get_batch_from_queue(max_batch_size: int = MAX_BATCH_SIZE, batch_wait_time: float = BATCH_WAIT_TIME) -> List[Tuple[str, str]]:
    """
    Retrieve a batch of requests from the queue.
    """
    batch = []
    batch_start_time = time.time()
    try:
        first_request = inference_queue.get(timeout=1.0)
        batch.append(first_request)
        while len(batch) < max_batch_size:
            elapsed = time.time() - batch_start_time
            remaining_timeout = max(0, batch_wait_time - elapsed)
            if remaining_timeout <= 0:
                break
            try:
                request = inference_queue.get(timeout=remaining_timeout)
                batch.append(request)
            except queue.Empty:
                break
    except queue.Empty:
        pass
    if len(batch) > 1:
        print(f"Collected batch of {len(batch)} requests")
    return batch

# Request queues and result storage
inference_queue = queue.Queue()
results = {}  # Maps request_id to generated tokens
results_lock = Lock()  # Add lock for thread safety

def get_model_and_tokenizer(device: torch.device) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    global model, tokenizer
    checkpoint = "HuggingFaceTB/SmolLM2-360M-Instruct"
    if model is None:
        model = AutoModelForCausalLM.from_pretrained(checkpoint)
        model = model.to(device)
        model = torch.compile(model)
        model.eval()
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, padding_side="left")
    print(f"Model size: {model.get_memory_footprint() / 1e6:.2f} MB")
    return model, tokenizer

def model_worker() -> None:
    """Background worker that processes batched inference requests"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = get_model_and_tokenizer(device)
    
    while not shutdown_event.is_set():
        try:
            batch = get_batch_from_queue()
            if shutdown_event.is_set() or not batch:
                continue
            
            formatted_text = []
            for request_id, prompt in batch:
                text = [{"role": "user", "content": prompt}]
                results[request_id] = queue.Queue()
                formatted_text.append(tokenizer.apply_chat_template(text, tokenize=False))

            skip_beginning = [True] * len(batch)
            active_requests = set(range(len(batch)))
            batch_inputs = tokenizer(formatted_text, padding=True, return_tensors="pt").to(device)
            batch_tokens, batch_attention_mask = batch_inputs.input_ids, batch_inputs.attention_mask
            for _ in range(MAX_TOKENS):
                if not active_requests:
                    break  # All requests are done
                
                # Generate as batch
                with torch.no_grad():
                    outputs = model.generate(
                        batch_tokens,
                        attention_mask=batch_attention_mask,
                        max_new_tokens=1,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                generated_token_ids = [output[-1:] for output in outputs]

                for idx in list(active_requests):
                    request_id, _ = batch[idx]
                    new_token_id = generated_token_ids[idx]
                    new_token = tokenizer.decode(new_token_id)
                    
                    # Check for the start of the assistant's response
                    if skip_beginning[idx] and new_token in "<|im_start|>assistant":
                        continue
                    else:
                        skip_beginning[idx] = False
                    
                    # Check if we've reached the end
                    if new_token_id == tokenizer.eos_token_id:
                        active_requests.remove(idx)
                        # Mark the end of generation for this request
                        with results_lock:
                            if request_id in results:
                                print(f"Finished - Request ID: {request_id}")
                                results[request_id].put(None)  # Signal end of generation

                        # Replace the EOS token in the output tensor (some transformer models use EOS and PAD interchangeably
                        # and this will raise warnings when we try to generate again)
                        alternative_id = tokenizer.unk_token_id if hasattr(tokenizer, 'unk_token_id') else 0
                        outputs[idx, -1] = alternative_id
                        continue
                
                    # Put the token in the result queue
                    with results_lock:
                        if request_id in results:
                            results[request_id].put(new_token)

                # Update for next generation step
                batch_tokens = outputs
                batch_attention_mask = F.pad(batch_attention_mask, (0, 1), value=1)

        except Exception as e:
            print(f"Error in model worker: {e}")
            import traceback
            traceback.print_exc()

async def stream_results(request_id: str) -> AsyncGenerator[str, None]:
    """Stream results as they become available"""
    # Wait for the worker to create the result queue
    max_wait_seconds = 60
    wait_interval = 0.1
    wait_attempts = int(max_wait_seconds / wait_interval)
    for _ in range(wait_attempts):
        with results_lock:
            if request_id in results:
                result_queue = results[request_id]
                break
        await asyncio.sleep(wait_interval)
    else:
        raise HTTPException(status_code=408, detail="Request timeout: model processing took too long to start")
    
    # Stream results from queue
    while True:
        try:
            token = result_queue.get(block=False)
            if token is None:  # End of generation
                with results_lock:
                    if request_id in results:
                        del results[request_id]
                break
            yield token
        except queue.Empty:
            await asyncio.sleep(0.01)

@app.post("/generate")
async def generate_text(input_txt: TextInput) -> StreamingResponse:
    # Generate a unique ID for this request
    request_id = str(uuid.uuid4())
    # Put the request in the queue
    inference_queue.put((request_id, input_txt.text))
    print(f"Received - Request ID: {request_id}, Prompt: {input_txt.text}")
    # Return a streaming response
    return StreamingResponse(stream_results(request_id), media_type="text/plain")

@app.get("/")
async def main() -> Dict[str, str]:
    return {"status": "Server is running"}