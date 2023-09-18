# initialize fast api service

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

from src.presentation.text_generator import TextGenerator


class ZaloHomeTestGateway:
    """
        REST API Gateway of Hometext
    """
    api: FastAPI = FastAPI(title="Zalo Home Test",
                           version="1.0.0",
                           description="Zalo Home Test",
                           contact={"name": "Hai pham", "email": "haipham1996@gmail.com"},
                            root_path="")
    api.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    api.include_router(TextGenerator().router, prefix='/v1')


if __name__ == '__main__':
    Instrumentator().instrument(ZaloHomeTestGateway.api).expose(ZaloHomeTestGateway.api)
    uvicorn.run(app=ZaloHomeTestGateway.api, port=8000, host='0.0.0.0')
