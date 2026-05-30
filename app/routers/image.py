from fastapi import APIRouter, HTTPException
from app.models.schemas import ImageRequest, ImageResponse
from app.services.image_gen import generate
from app.services.memory import get_or_create_session, add_message

router = APIRouter(prefix="/image", tags=["image"])


@router.post("", response_model=ImageResponse)
def image(req: ImageRequest):
    try:
        session_id = get_or_create_session(req.session_id)
        result = generate(req.prompt)
        reply = result.get("message", "")
        add_message(session_id, "user", req.prompt)
        add_message(session_id, "assistant", reply or result.get("sanitised_prompt", ""))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return ImageResponse(
        success=result["success"],
        image_url=result["image_url"],
        sanitised_prompt=result["sanitised_prompt"],
        block_reason=result["block_reason"],
        message=result.get("message", ""),
        session_id=session_id,
    )
