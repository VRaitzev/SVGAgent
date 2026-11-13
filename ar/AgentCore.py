"""Ядро AI-агента для автоматической сборки и редактирования SVG-композиций.

Реализует ReAct-агента на базе LangGraph, который умеет:
- Искать компоненты в библиотеке по метафоре/названию.
- Собирать сцену с нуля (сетка, размеры, слои).
- Вносить транзакционные диффы (двигать, масштабировать, красить).
- Проверять сцену (математически и визуально через PNG-превью).
- Откатывать изменения, если что-то пошло не так.

Поддерживает два режима контекста:
1. "text_only" — только текстовые описания и параметры (быстро, дешево).
2. "multimodal" — передает PNG-превью компонентов и сцены в LLM (максимальное качество).
"""

from __future__ import annotations

import base64
import json
import os
from enum import Enum
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from ar.agent_tools import ToolRegistry
from ar.renderer import CompositionRenderer
from ar.scene_session import SceneSession
from ar.svg_store import SVGStore


class ContextMode(str, Enum):
    """Режим передачи контекста в LLM."""
    TEXT_ONLY = "text_only"
    MULTIMODAL = "multimodal"


class AgentCore:
    """Оркестратор ReAct-агента для работы с SVG-композициями."""

    def __init__(
        self,
        store: SVGStore,
        session: SceneSession,
        renderer: CompositionRenderer,
        *,
        context_mode: ContextMode = ContextMode.TEXT_ONLY,
        visual_review: bool = True,
        model_name: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.0,
    ) -> None:
        self.store = store
        self.session = session
        self.renderer = renderer
        self.context_mode = context_mode
        self.visual_review = visual_review

        # Настройка LLM (по умолчанию DeepSeek)
        self.model_name = model_name or os.getenv("LLM_MODEL") or "deepseek-chat"
        self.api_key = api_key or os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("LLM_BASE_URL") or "https://api.deepseek.com/v1"

        if not self.api_key:
            raise ValueError("Не задан API-ключ для LLM (LLM_API_KEY или OPENAI_API_KEY).")

        self.llm = ChatOpenAI(
            model=self.model_name,
            api_key=self.api_key,
            base_url=self.base_url,
            temperature=temperature,
            max_retries=3,
        )

        # Инициализация инструментов и агента
        self.tool_registry = ToolRegistry(session, store, renderer)
        self.tools = self.tool_registry.get_tools()
        self.agent = create_react_agent(self.llm, self.tools)

    # ---------- Вспомогательные методы для подготовки контекста ----------

    def _get_system_prompt(self) -> str:
        """Собирает системный промпт: статическая роль + динамическая документация тулов."""
        static_role = (
            "Ты — эксперт по графическому дизайну и векторной графике. "
            "Твоя задача — собирать красивые иллюстрации из готовых SVG-компонентов, "
            "используя предоставленные инструменты.\n\n"
            "ПРАВИЛА:\n"
            "1. Действуй пошагово: сначала найди компоненты, потом собери сцену, потом проверь результат.\n"
            "2. Не бойся делать несколько итераций, если сцена выглядит неидеально.\n"
            "3. Всегда проверяй сцену через `inspect_and_validate_scene` после важных изменений.\n"
            "4. Если что-то сломалось — используй историю и откат.\n\n"
        )

        tools_doc = self.tool_registry.get_tools_documentation()
        return static_role + tools_doc

    def _prepare_multimodal_content(self, query: str) -> list[dict[str, Any]]:
        """Собирает контент сообщения (текст + картинки) для мультимодальной модели."""
        content: list[dict[str, Any]] = [{"type": "text", "text": query}]

        candidates = self.session.candidates
        if candidates:
            content.append({"type": "text", "text": "\nВизуальные превью доступных кандидатов:"})
            for cid in candidates[:8]:  # Ограничиваем 8 картинками
                try:
                    img = self.store.get(cid)
                    if img.png_path and os.path.exists(img.png_path):
                        with open(img.png_path, "rb") as f:
                            b64 = base64.b64encode(f.read()).decode("utf-8")
                        content.append({"type": "text", "text": f"Кандидат {cid} ({img.name}):"})
                        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
                except Exception:
                    pass

        if self.visual_review:
            try:
                png_bytes = self.renderer.render_composition_png(self.session.composition)
                b64_scene = base64.b64encode(png_bytes).decode("utf-8")
                content.append({"type": "text", "text": "\nТекущее состояние сцены (если объекты уже есть):"})
                content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_scene}"}})
            except Exception:
                pass

        return content

    # ---------- Публичные методы запуска ----------

    def generate_composition(self, user_request: str) -> str:
        return self._run_agent(user_request, is_editing=False)

    def edit_composition(self, user_request: str) -> str:
        return self._run_agent(user_request, is_editing=True)

    def _run_agent(self, user_request: str, *, is_editing: bool = False) -> str:
        scene_summary = self.session.get_scene_summary()
        scene_json = json.dumps(scene_summary, ensure_ascii=False, indent=2)

        messages: list[Any] = [SystemMessage(content=self._get_system_prompt())]

        prompt_text = (
            f"Текущее состояние сцены (JSON):\n{scene_json}\n\n"
            f"Запрос пользователя: {user_request}"
        )

        if self.context_mode == ContextMode.MULTIMODAL:
            content = self._prepare_multimodal_content(prompt_text)
            messages.append(HumanMessage(content=content))
        else:
            messages.append(HumanMessage(content=prompt_text))

        try:
            # ИСПРАВЛЕНИЕ: Добавлен recursion_limit для предотвращения обрыва на сложных правках
            result = self.agent.invoke({"messages": messages}, config={"recursion_limit": 50})
            final_message = result["messages"][-1]
            return final_message.content
        except Exception as e:
            return f"Ошибка во время выполнения агента: {e}"