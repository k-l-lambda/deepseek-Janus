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

# Import the multimodal understanding functionality
from demo.fastapi_app import generate_image

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

@torch.inference_mode()
def multimodal_understanding(image_data, question, seed, top_p, temperature):
    torch.cuda.empty_cache()
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)

    conversation = [
        {
            "role": "User",
            "content": f"<image_placeholder>\n{question}",
            "images": [image_data],
        },
        {"role": "Assistant", "content": ""},
    ]

    pil_images = [Image.open(io.BytesIO(image_data))]
    prepare_inputs = vl_chat_processor(
        conversations=conversation, images=pil_images, force_batchify=True
    ).to(cuda_device, dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16)
    
    inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)
    outputs = vl_gpt.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=512,
        do_sample=False if temperature == 0 else True,
        use_cache=True,
        temperature=temperature,
        top_p=top_p,
    )
    
    answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
    return answer

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
            guidance=request.guidance
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
