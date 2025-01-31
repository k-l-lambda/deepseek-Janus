from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Union
import base64
import io
from PIL import Image
import torch
from transformers import AutoConfig, AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
import numpy as np
import time
import requests
import uuid
from dataclasses import dataclass, field, asdict
import json

# Import the multimodal understanding functionality
from demo.fastapi_app import generate_image

class ModelData(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str
    permission: List[dict] = []

class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelData]

#model_path = "deepseek-ai/Janus-1.3B"
model_path = "deepseek-ai/Janus-Pro-7B"
config = AutoConfig.from_pretrained(model_path)
language_config = config.language_config
language_config._attn_implementation = 'eager'
vl_gpt = AutoModelForCausalLM.from_pretrained(model_path,
                                             language_config=language_config,
                                             trust_remote_code=True)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda()

vl_chat_processor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer
cuda_device = 'cuda' if torch.cuda.is_available() else 'cpu'

app = FastAPI(title="Janus OpenAI-Compatible API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ImageContent(BaseModel):
    type: str = "image_url"  # or "image_base64"
    image_url: Optional[str] = None
    base64: Optional[str] = None

class Message(BaseModel):
    role: str
    content: Union[str, List[Union[str, ImageContent]]]

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.1
    top_p: Optional[float] = 0.95
    seed: Optional[int] = 42

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[dict]
    usage: dict

class ImageGenerationRequest(BaseModel):
    prompt: str
    model: Optional[str] = "janus-1.3b"
    n: Optional[int] = 1
    size: Optional[str] = "1024x1024"
    response_format: Optional[str] = "b64_json"
    seed: Optional[int] = None
    guidance: Optional[float] = 5.0

class CompletionRequest(BaseModel):
    model: str
    prompt: str
    max_tokens: Optional[int]	= 512
    temperature: Optional[float]	= 0.0
    top_p: Optional[float] = 1.0
    n: Optional[int]	= 1
    image_data: Optional[str] = None

@dataclass
class CompletionChoice:
    text: str
    index: int
    logprobs: Optional[dict] = None
    finish_reason: Optional[str] = None

@dataclass
class CompletionUsage:
    prompt_tokens: str
    completion_tokens: str
    total_tokens: str

class CompletionResponse(BaseModel):
    id: str = field(default_factory=lambda: f"cmpl-{str(uuid.uuid4())}")
    object: str = "text_completion"
    created: int = field(default_factory=lambda: int(time.time()))
    model: str = None
    choices: List[CompletionChoice] = field(default_factory=list)
    usage: CompletionUsage = None


@torch.inference_mode()
def multimodal_understanding(image_data=None, question=None, prompt=None, seed=None, top_p=1, temperature=1, max_new_tokens=512):
    torch.cuda.empty_cache()
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed(seed)

    conversation = [
        {
            "role": "User",
            "content": f"<image_placeholder>\n{question}",
            "images": [image_data] if image_data is not None else [],
        },
        {"role": "Assistant", "content": ""},
    ] if question is not None else None

    pil_images = [Image.open(io.BytesIO(image_data))] if image_data is not None else []
    prepare_inputs = vl_chat_processor(
        conversations=conversation, prompt=prompt, images=pil_images, force_batchify=True
    ).to(cuda_device, dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16)
    
    inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)
    outputs = vl_gpt.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=max_new_tokens,
        do_sample=False if temperature == 0 else True,
        use_cache=True,
        temperature=temperature,
        top_p=top_p,
    )
    
    answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
    return answer

@app.get("/v1/models")
async def list_models():
    models = [
        #ModelData(
        #    id="janus-1.3b",
        #    created=1709251200,  # March 2024
        #    owned_by="deepseek-ai",
        #    permission=[{
        #        "id": "modelperm-janus-1.3b",
        #        "object": "model_permission",
        #        "created": 1709251200,
        #        "allow_create_engine": False,
        #        "allow_sampling": True,
        #        "allow_logprobs": True,
        #        "allow_search_indices": False,
        #        "allow_view": True,
        #        "allow_fine_tuning": False,
        #        "organization": "*",
        #        "group": None,
        #        "is_blocking": False
        #    }]
        #),
        ModelData(
            id="janus-pro-7b",
            created=1709251200,
            owned_by="deepseek-ai",
            permission=[{
                "id": model_path,
                "object": "model_permission",
                "created": 1709251200,
                "allow_create_engine": False,
                "allow_sampling": True,
                "allow_logprobs": True,
                "allow_search_indices": False,
                "allow_view": True,
                "allow_fine_tuning": False,
                "organization": "*",
                "group": None,
                "is_blocking": False
            }]
        )
    ]
    return ModelList(data=models)

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    try:
        # Extract image and question from the messages
        messages = request.messages
        if not messages:
            raise HTTPException(status_code=400, detail="No messages provided")

        # Process the last user message
        last_msg = messages[-1]
        if last_msg.role != "user":
            raise HTTPException(status_code=400, detail="Last message must be from user")

        # Handle content processing
        content = last_msg.content
        if isinstance(content, list):
            image_content = None
            question = ""
            #print(f'{content=}')

            for item in content:
                if isinstance(item, BaseModel) and item.type == "image_url":
                    image_content = item
                elif isinstance(item, str):
                    question += item

            if not image_content:
                raise HTTPException(status_code=400, detail="No image content found")

            # Extract image data from URL or data URL
            if image_content.image_url:
                image_url = image_content.image_url
                if image_url.startswith('data:image/'):
                    # Handle data URL
                    header, encoded = image_url.split(",", 1)
                    image_data = base64.b64decode(encoded)
                else:
                    # Handle regular URL
                    response = requests.get(image_url)
                    image_data = response.content
            elif image_content.base64:
                image_data = base64.b64decode(image_content.base64)
            else:
                raise HTTPException(status_code=400, detail="Invalid image format")

            # Process the image and question
            response = multimodal_understanding(
                image_data=image_data,
                question=question,
                seed=request.seed,
                top_p=request.top_p,
                temperature=request.temperature
            )

            return ChatCompletionResponse(
                id=f"chatcmpl-{hash(response)}",
                created=int(time.time()),
                model=request.model,
                choices=[{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response
                    },
                    "finish_reason": "stop"
                }],
                usage={
                    "prompt_tokens": len(question),
                    "completion_tokens": len(response),
                    "total_tokens": len(question) + len(response)
                }
            )
        else:
            raise HTTPException(status_code=400, detail="Invalid message format")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/images/generations")
async def create_image(request: ImageGenerationRequest):
    try:
        images = generate_image(
            prompt=request.prompt,
            seed=request.seed,
            guidance=request.guidance,
            size=request.size,
            parallel_size=request.n,
        )

        response_images = []
        for img in images:
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            if request.response_format == "b64_json":
                response_images.append({"b64_json": img_str})
            else:
                # Implement URL storage and return URLs if needed
                response_images.append({"url": f"data:image/png;base64,{img_str}"})

        return {
            "created": int(time.time()),
            "data": response_images
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/v1/completions')
async def completion(request: CompletionRequest):
    try:
        # Extract parameters
        max_tokens = request.max_tokens
        temperature = request.temperature
        top_p = request.top_p
        n = request.n
        prompt = request.prompt

        # Call multimodal_understanding instead of direct generate
        model_response = multimodal_understanding(
            prompt=prompt,
            image_data=request.image_data,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )

        choices = [
            CompletionChoice(
                text=model_response,
                index=i,
                finish_reason="stop",
            ) for i in range(n)
        ]

        usage = CompletionUsage(
            prompt_tokens=prompt,
            completion_tokens=model_response,
            total_tokens=prompt + model_response,
        )

        return CompletionResponse(
            model=request.model,
            choices=choices,
            usage=usage,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    args = parser.parse_args()
    
    uvicorn.run(app, host="0.0.0.0", port=args.port)
