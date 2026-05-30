from fastapi import APIRouter, HTTPException
from app.models.schemas import ChatRequest, ChatResponse, VerseSource
from app.services.pipeline import run

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
def chat(req: ChatRequest):
    try:
        result = run(
            user_message=req.message,
            session_id=req.session_id,
            denomination=req.denomination,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return ChatResponse(
        reply=result["reply"],
        sources=[VerseSource(**v) for v in result["sources"]],
        unverified=result["unverified"],
        session_id=result["session_id"],
        denomination=result["denomination"],
        provider=result["provider"],
        blocked=result["blocked"],
        block_category=result["block_category"],
        output_flagged=result["output_flagged"],
        is_image=result["is_image"],
        image_url=result["image_url"],
    )
