"""FastAPI сервер для проекта Antigravity:
- синхронные эндпоинты (выполняются в threadpool и не блокируют сервер во время работы агента);
- корректные HTTP-коды ошибок (404 для ненайденных элементов, 400 для ошибок валидации, 500 для сбоев LLM).
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel

from AgentCore import AgentCore, ContextMode
from composition import CompositionError
from palette import PRESET_PALETTES, Palette, load_custom_palettes, register_palette
from renderer import CompositionRenderer
from scene_session import SceneSession, SceneValidationError
from svg_store import SVGStore

load_dotenv()

store: SVGStore | None = None
session: SceneSession | None = None
renderer: CompositionRenderer | None = None
agent: AgentCore | None = None


DEFAULT_SVGS: list[dict[str, Any]] = [
    {
        "name": "Открытая книга (L)",
        "concept_key": "book",
        "detail_level": "L",
        "role": "main",
        "description": "Детализированная открытая книга с закладкой, обучение, документация, знания, гайд",
        "svg": '<svg viewBox="0 0 180 140" xmlns="http://www.w3.org/2000/svg"><path d="M20 35 L90 50 L160 35 L160 115 L90 130 L20 115 Z" fill="#ddd6fe" stroke="#1e293b" stroke-width="4"/><path d="M28 25 L90 42 L90 120 L28 105 Z" fill="#ffffff" stroke="#1e293b" stroke-width="4"/><path d="M152 25 L90 42 L90 120 L152 105 Z" fill="#f8fafc" stroke="#1e293b" stroke-width="4"/><polygon points="112,36 126,32 126,76 119,68 112,76" fill="#7c3aed"/></svg>',
    },
    {
        "name": "Открытая книга (S)",
        "concept_key": "book",
        "detail_level": "S",
        "role": "main",
        "description": "Компактная книга, обучение, чтение, справочник",
        "svg": '<svg viewBox="0 0 110 90" xmlns="http://www.w3.org/2000/svg"><path d="M14 20 L55 30 L96 20 L96 72 L55 82 L14 72 Z" fill="#ffffff" stroke="#1e293b" stroke-width="4"/><line x1="55" y1="30" x2="55" y2="82" stroke="#1e293b" stroke-width="4"/><rect x="66" y="25" width="10" height="26" fill="#7c3aed"/></svg>',
    },
]


def seed_default_library(st: SVGStore) -> None:
    if st.count() > 0:
        return
    for item in DEFAULT_SVGS:
        try:
            st.add(
                item["svg"],
                name=item["name"],
                description=item["description"],
                detail_level=item["detail_level"],
                role=item["role"],
                concept_key=item["concept_key"],
            )
        except Exception as exc:
            print(f"Не удалось добавить стартовый SVG {item['name']}: {exc}")


def get_or_init_agent() -> AgentCore:
    global agent
    if agent is None:
        assert store is not None and session is not None and renderer is not None
        agent = AgentCore(
            store=store,
            session=session,
            renderer=renderer,
            context_mode=ContextMode.TEXT_ONLY,
            library_mode="full_catalog",
            rag_top_k=10,
            temperature=0.1,
        )
    return agent


@asynccontextmanager
async def lifespan(app: FastAPI):
    global store, session, renderer, agent
    print("🚀 Инициализация сервисов Antigravity...")
    load_custom_palettes("palettes.json")

    store = SVGStore(db_path="data/lancedb", png_dir="data/png")
    seed_default_library(store)

    session = SceneSession(
        get_image=store.get,
        find_variant=store.find_variant,
        find_role_variant=store.find_role_variant,
    )
    session.create_scene(
        width=800.0,
        height=800.0,
        grid_columns=3,
        grid_rows=3,
        detail_level="L",
        uniform_scale=1.0,
        strict_bounds=False,
        layers=[
            {"name": "Background", "strict_bounds": False, "allow_internal_overlap": True},
            {"name": "Main", "strict_bounds": False, "allow_internal_overlap": True},
        ],
        palette="lavender_mist",
        commit_message="Инициализация сцены (Размер L, Лавандовая схема)",
    )
    renderer = CompositionRenderer(get_image=store.get)

    try:
        agent = get_or_init_agent()
        print("✅ AgentCore успешно подключён")
    except Exception as exc:
        print(f"⚠️ AgentCore будет инициализирован при наличии ключа ({exc})")
    yield
    print("🛑 Остановка сервисов Antigravity...")


app = FastAPI(title="Antigravity SVG Composer API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _build_full_scene_response() -> dict[str, Any]:
    assert session is not None and renderer is not None
    payload = renderer.to_frontend_payload(session.composition)
    payload["step_id"] = session.current_step_id
    payload["warnings"] = list(session.last_warnings)
    payload["active_palette"] = session.active_palette.to_dict() if session.active_palette else None
    payload["history"] = session.get_history()
    report = session.composition.validate(session.get_image)
    color_errors = session._validate_color_rules(session.composition, session.active_palette)
    payload["validation"] = {
        "ok": report.ok and not color_errors,
        "out_of_bounds": list(report.out_of_bounds),
        "role_violations": list(report.role_violations),
        "scale_violations": list(report.scale_violations),
        "color_violations": color_errors,
        "overlaps": [
            {"layer_id": ov.layer_id, "first": ov.first, "second": ov.second, "area": round(ov.area, 1)}
            for ov in report.overlaps
        ],
    }
    return payload


class ChatRequest(BaseModel):
    prompt: str
    context_mode: Literal["text_only", "multimodal"] | None = None
    library_mode: Literal["full_catalog", "rag"] | None = None
    rag_top_k: int | None = None


class DiffRequest(BaseModel):
    operations: list[dict[str, Any]]
    commit_message: str = "Ручное редактирование через UI"
    strict_validation: bool = True


class RevertRequest(BaseModel):
    step_id: int


class RegisterPaletteRequest(BaseModel):
    name: str
    title: str
    bg_colors: list[str]
    accent_colors: list[str]
    canvas_bg: str = "#ffffff"


class AddLibraryItemRequest(BaseModel):
    svg: str
    name: str
    description: str | None = None
    detail_level: Literal["L", "S"] = "L"
    role: Literal["main", "background"] = "main"
    concept_key: str | None = None


@app.get("/api/health")
def health_check():
    return {
        "status": "ok",
        "library_count": store.count() if store else 0,
        "agent_ready": agent is not None,
    }


@app.post("/api/chat")
def chat_with_agent(req: ChatRequest):
    """Синхронный эндпоинт: FastAPI выполняет его в threadpool, не блокируя event loop."""
    assert session is not None
    try:
        ag = get_or_init_agent()
        if req.context_mode:
            ag.context_mode = ContextMode(req.context_mode)
        if req.library_mode:
            ag.set_search_config(req.library_mode, req.rag_top_k or ag.tool_registry.rag_top_k)
        elif req.rag_top_k:
            ag.set_search_config(ag.tool_registry.library_mode, req.rag_top_k)

        if not session.composition.items:
            answer = ag.generate_composition(req.prompt)
        else:
            answer = ag.edit_composition(req.prompt)

        return {
            "answer": answer,
            "scene": _build_full_scene_response(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/scene")
def get_scene():
    return _build_full_scene_response()


@app.post("/api/scene/diff")
def apply_manual_diff(req: DiffRequest):
    assert session is not None
    try:
        entry = session.apply_diff(
            req.operations,
            commit_message=req.commit_message,
            strict_validation=req.strict_validation,
        )
        return {
            "status": "ok",
            "step_id": entry.step_id,
            "warnings": list(session.last_warnings),
            "scene": _build_full_scene_response(),
        }
    except SceneValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except (CompositionError, KeyError) as e:
        err_msg = str(e)
        status = 404 if "не найден" in err_msg.lower() or "not found" in err_msg.lower() else 400
        raise HTTPException(status_code=status, detail=err_msg)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/scene/undo")
def undo_step():
    assert session is not None
    try:
        session.undo()
        return _build_full_scene_response()
    except CompositionError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/scene/redo")
def redo_step():
    assert session is not None
    try:
        session.redo()
        return _build_full_scene_response()
    except CompositionError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/scene/revert")
def revert_step(req: RevertRequest):
    assert session is not None
    try:
        session.revert_to_step(req.step_id)
        return _build_full_scene_response()
    except CompositionError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/api/palettes")
def get_palettes():
    return [pal.to_dict() for pal in PRESET_PALETTES.values()]


@app.post("/api/palettes/register")
def create_custom_palette(req: RegisterPaletteRequest):
    pal = Palette(
        name=req.name,
        title=req.title,
        bg_colors=req.bg_colors,
        accent_colors=req.accent_colors,
        canvas_bg=req.canvas_bg,
    )
    register_palette(pal)
    return pal.to_dict()


@app.get("/api/library/search")
def search_library(
    q: str = "",
    detail_level: Literal["L", "S"] | None = None,
    role: Literal["main", "background"] | None = None,
    limit: int = 50,
):
    assert store is not None
    results = store.search(q, limit=limit, detail_level=detail_level, role=role)
    return [
        {
            "id": img.id,
            "name": img.name,
            "description": img.description,
            "detail_level": img.detail_level,
            "role": img.role,
            "concept_key": img.concept_key,
            "width": img.width,
            "height": img.height,
            "colors": list(img.colors),
            "svg": img.svg,
        }
        for img in results
    ]


@app.post("/api/library/add")
def add_library_item(req: AddLibraryItemRequest):
    assert store is not None
    try:
        img = store.add(
            req.svg,
            name=req.name,
            description=req.description,
            detail_level=req.detail_level,
            role=req.role,
            concept_key=req.concept_key,
        )
        return {
            "id": img.id,
            "name": img.name,
            "description": img.description,
            "detail_level": img.detail_level,
            "role": img.role,
            "concept_key": img.concept_key,
            "width": img.width,
            "height": img.height,
            "colors": list(img.colors),
            "svg": img.svg,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/export")
def export_scene(fmt: Literal["svg", "png"] = "svg", background: str | None = None):
    assert session is not None and renderer is not None
    if fmt == "svg":
        svg_str = renderer.render_composition_svg(session.composition, background=background)
        return Response(
            content=svg_str,
            media_type="image/svg+xml",
            headers={"Content-Disposition": 'attachment; filename="antigravity_scene.svg"'},
        )
    png_bytes = renderer.render_composition_png(session.composition, background=background)
    return Response(
        content=png_bytes,
        media_type="image/png",
        headers={"Content-Disposition": 'attachment; filename="antigravity_scene.png"'},
    )

from pathlib import Path
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

for candidate_dist in (Path("dist"), Path("../dist"), Path("../../dist")):
    if candidate_dist.is_dir() and (candidate_dist / "index.html").is_file():
        assets_dir = candidate_dist / "assets"
        if assets_dir.is_dir():
            app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

        @app.get("/{full_path:path}")
        def serve_react_spa(full_path: str, _dist: Path = candidate_dist):
            file_path = _dist / full_path
            if full_path and file_path.is_file():
                return FileResponse(file_path)
            return FileResponse(_dist / "index.html")
        break

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)