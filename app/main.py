import app  # telemetry patch must run first

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import GROQ_MODEL
from app.models.schemas import HealthResponse, SessionClearResponse
from app.routers import chat, image
from app.services.memory import init_db, clear_session

init_db()

api = FastAPI(
    title="Christianity AI Assistant",
    description="Scripture-grounded, denomination-aware Christian AI with safety layer.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

api.include_router(chat.router)
api.include_router(image.router)


@api.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", model=GROQ_MODEL, version="1.0.0")


@api.delete("/session/{session_id}", response_model=SessionClearResponse)
def delete_session(session_id: str):
    clear_session(session_id)
    return SessionClearResponse(cleared=True, session_id=session_id)


@api.get("/")
def root():
    return {
        "name": "Christianity AI Assistant",
        "docs": "/docs",
        "health": "/health",
    }
