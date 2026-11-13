"""FastAPI сервер для проекта Antigravity.

Предоставляет REST API для взаимодействия с ReAct-агентом,
получения состояния сцены для фронтенда и ручного редактирования.
"""

import os
from contextlib import asynccontextmanager
from typing import Any, Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from ar.AgentCore import AgentCore, ContextMode
from ar.renderer import CompositionRenderer
from ar.scene_session import SceneSession, SceneValidationError
from ar.svg_store import SVGStore

load_dotenv()

# Глобальные объекты для MVP (в продакшене это хранилось бы в Redis/Postgres по session_id)
store: SVGStore = None
session: SceneSession = None
renderer: CompositionRenderer = None
agent: AgentCore = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global store, session, renderer, agent
    print("Инициализация сервисов...")
    store = SVGStore(db_path="data/lancedb", png_dir="data/png")
    session = SceneSession(get_image=store.get)
    renderer = CompositionRenderer(get_image=store.get)
    
    agent = AgentCore(
        store=store,
        session=session,
        renderer=renderer,
        context_mode=ContextMode.TEXT_ONLY,
        temperature=0.1,
    )
    yield
    print("Остановка сервисов...")


app = FastAPI(title="Antigravity API", lifespan=lifespan)


# ==========================================
# Модели запросов
# ==========================================

class ChatRequest(BaseModel):
    prompt: str


class DiffRequest(BaseModel):
    operations: list[dict[str, Any]]
    commit_message: str = "Ручное редактирование через UI"


# ==========================================
# Эндпоинты
# ==========================================

@app.post("/api/chat")
async def chat_with_agent(req: ChatRequest):
    """Отправляет запрос LLM-агенту. Возвращает текстовый ответ и новое состояние сцены."""
    try:
        # Если сцена пустая, вызываем создание, иначе — редактирование
        if not session.composition.items:
            answer = agent.generate_composition(req.prompt)
        else:
            answer = agent.edit_composition(req.prompt)
            
        return {
            "answer": answer,
            "scene": renderer.to_frontend_payload(session.composition)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/scene")
async def get_scene():
    """Возвращает текущее состояние сцены (JSON) для отрисовки на фронтенде."""
    return renderer.to_frontend_payload(session.composition)


@app.post("/api/scene/diff")
async def apply_manual_diff(req: DiffRequest):
    """Применяет ручные изменения с фронтенда (drag & drop, resize)."""
    try:
        entry = session.apply_diff(req.operations, commit_message=req.commit_message)
        return {
            "status": "ok",
            "step_id": entry.step_id,
            "scene": renderer.to_frontend_payload(session.composition)
        }
    except SceneValidationError as e:
        # Если фронтенд прислал кривые координаты (коллизия), сервер это отклонит
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/scene/undo")
async def undo_step():
    """Откат на один шаг назад."""
    try:
        session.undo()
        return renderer.to_frontend_payload(session.composition)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/scene/redo")
async def redo_step():
    """Возврат на один шаг вперед."""
    try:
        session.redo()
        return renderer.to_frontend_payload(session.composition)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/library/search")
async def search_library(q: str, limit: int = 10):
    """Поиск по базе SVG компонентов (для боковой панели фронтенда)."""
    results = store.search(q, limit=limit)
    return [
        {
            "id": img.id,
            "name": img.name,
            "description": img.description,
            "width": img.width,
            "height": img.height,
            "svg": img.svg,
        }
        for img in results
    ]


@app.get("/api/export")
async def export_scene(fmt: Literal["svg", "png"] = "png"):
    """Скачивание финальной композиции."""
    if not session.composition.items:
        raise HTTPException(status_code=400, detail="Сцена пуста")
        
    if fmt == "svg":
        svg_str = renderer.render_composition_svg(session.composition)
        return Response(content=svg_str, media_type="image/svg+xml")
    else:
        png_bytes = renderer.render_composition_png(session.composition)
        return Response(content=png_bytes, media_type="image/png")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
