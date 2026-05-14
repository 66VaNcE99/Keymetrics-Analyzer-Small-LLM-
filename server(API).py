from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer
from awq import AutoAWQForCausalLM
import torch
import os
import traceback

os.environ.pop("AWQ_FORCE_TRITON", None)

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct-AWQ"

app = FastAPI()

print("Loading AWQ model...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

model = AutoAWQForCausalLM.from_quantized(
    MODEL_NAME,
    device_map="auto",
    fuse_layers=True
)

model.eval()

class Request(BaseModel):
    prompt: str
    max_new_tokens: int = 256


@app.post("/generate")
def generate(req: Request):
    try:
        messages = [
            {"role": "user", "content": req.prompt}
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = tokenizer(
            text,
            return_tensors="pt"
        ).to("cuda:0")

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=req.max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )

        generated = output[0][inputs["input_ids"].shape[1]:]

        response = tokenizer.decode(
            generated,
            skip_special_tokens=True
        )

        return {"response": response}

    except Exception as e:
        print("FULL SERVER ERROR:")
        traceback.print_exc()
        return {"error": str(e)}
