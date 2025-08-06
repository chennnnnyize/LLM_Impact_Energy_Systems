import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import asyncio
import uuid
import gc
import torch
import signal
import sys
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams

app = FastAPI()
MODEL_PATH = "/home/ubuntu/AI-Power-Finetune3/models/llama-3.1-70b-Instruct"
model = None

# Request models
class PromptRequest(BaseModel):
    prompts: list[str]
    sampling_params: dict = {
        "temperature": 0.8,
        "top_p": 0.9,
        "max_tokens": 4096
    }

class GenerateRequest(BaseModel):
    prompt: str
    sampling_params: dict = {
        "temperature": 0.8,
        "top_p": 0.9,
        "max_tokens": 4096
    }

def load_model():
    global model
    engine_config = AsyncEngineArgs(
        model=MODEL_PATH,
        tensor_parallel_size=8
    )
    model = AsyncLLMEngine.from_engine_args(engine_config)
    print("Model loaded and held in memory.")

@app.on_event("startup")
async def startup_event():
    load_model()

@app.get("/health")
async def health_check():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy"}

@app.post("/generate")
async def generate(request: GenerateRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
        
    try:
        sampling_params = SamplingParams(**request.sampling_params)
        req_id = str(uuid.uuid4())
        
        final_output = None
        async for request_output in model.generate(request.prompt, sampling_params, req_id):
            final_output = request_output
            
        if final_output is None or not final_output.outputs:
            return {"output": ""}
            
        generated_text = final_output.outputs[0].text
        
        print(f"Debug - Generated text: {generated_text}")  # Debug print
        
        return {"output": generated_text.strip()}
        
    except Exception as e:
        print(f"Error in generate endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Sequential processing endpoint
@app.post("/run-sequential")
async def run_sequential(request: PromptRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
        
    sampling_params = SamplingParams(**request.sampling_params)
    responses = []
    
    for prompt in request.prompts:
        try:
            req_id = str(uuid.uuid4())
            final_output = None
            async for request_output in model.generate(prompt, sampling_params, req_id):
                final_output = request_output
            
            if final_output is None or not final_output.outputs:
                responses.append("")
                continue
                
            generated_text = final_output.outputs[0].text
            responses.append(generated_text.strip())
            
        except Exception as e:
            print(f"Error processing prompt: {e}")
            responses.append("")
            
    return {"responses": responses}

# Parallel processing endpoint
@app.post("/run-parallel")
async def run_parallel(request: PromptRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
        
    sampling_params = SamplingParams(**request.sampling_params)
    
    async def process_single_prompt(prompt, idx):
        try:
            print(f"\nProcessing prompt {idx}:")
            print(f"Input prompt: {prompt[:200]}...")  # Print first 200 chars
            
            req_id = str(uuid.uuid4())
            final_output = None
            async for request_output in model.generate(prompt, sampling_params, req_id):
                final_output = request_output
                print(f"Intermediate output {idx}: {request_output}")
                
            if final_output is None or not final_output.outputs:
                print(f"No output generated for prompt {idx}")
                return ""
                
            text = final_output.outputs[0].text.strip()
            print(f"Final output {idx}: '{text}'")
            return text
            
        except Exception as e:
            print(f"Error processing prompt {idx}: {e}")
            return ""

    tasks = [process_single_prompt(prompt, i) for i, prompt in enumerate(request.prompts)]
    responses = await asyncio.gather(*tasks)
    
    # Debug output
    print("\nAll responses:")
    for i, resp in enumerate(responses):
        print(f"Response {i}: '{resp}'")
    
    # If we got empty responses, try with modified parameters
    if any(not r.strip() for r in responses):
        print("\nDetected empty responses, trying with modified parameters...")
        modified_params = SamplingParams(
            temperature=0.7,
            top_p=1.0,
            max_tokens=16,
            stop=["\n", "[/INST]"]  # Add stop tokens
        )
        
        tasks = [process_single_prompt(prompt, i) for i, prompt in enumerate(request.prompts)]
        responses = await asyncio.gather(*tasks)
        
        print("\nRetry responses:")
        for i, resp in enumerate(responses):
            print(f"Retry response {i}: '{resp}'")
    
    return {"responses": responses}

# Shutdown event to free VRAM when the application is stopping
@app.on_event("shutdown")
async def shutdown_event():
    cleanup()

def cleanup():
    global model
    if model is not None:
        print("Unloading model and freeing VRAM...")
        model = None  # Clear model reference to free VRAM
        gc.collect()  # Force garbage collection
        torch.cuda.empty_cache()  # Free up the VRAM
    print("Shutdown complete.")

# Set up signal handlers
signal.signal(signal.SIGINT, lambda s, f: cleanup())
signal.signal(signal.SIGTERM, lambda s, f: cleanup())

if __name__ == "__main__":
    uvicorn.run("model_server:app", host="0.0.0.0", port=8000, log_level="debug")