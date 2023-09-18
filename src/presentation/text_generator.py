from contextlib import asynccontextmanager

import torch
from fastapi import APIRouter, Response, FastAPI
from transformers import AutoModelForCausalLM, AutoTokenizer

ml_models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    ml_models["tokenizer"] = AutoTokenizer.from_pretrained("bigscience/bloom-7b1")
    # optimize model
    ml_models["generator"] = AutoModelForCausalLM.from_pretrained(
        "bigscience/bloom-7b1",
        device_map="auto",  # auto infer where to store model parts (gpu, cpu or disk)
        load_in_8bit=True,  # load model in mixed int
    )
    yield
    # Clean up the ML models and release the resources
    ml_models.clear()


class TextGenerator:
    router = APIRouter(prefix="", tags=["Image Moderation"])

    @staticmethod
    @router.get("/generate",
                summary="Generate text")
    async def generate_text(response: Response,
                            text: str | None = None,
                            max_length: int = 50
                            ) -> dict:
        """
        Generate text
        """
        inputs = torch.tensor([ml_models["tokenizer"].encode(text)], dtype=torch.long)
        generate_text = ml_models["generator"].generate(
            inputs,
            max_length=max_length,
            do_sample=True,
            top_k=50,
            top_p=0.9
        )[0]

        return {"result": generate_text}