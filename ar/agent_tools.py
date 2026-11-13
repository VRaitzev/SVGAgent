"""Реестр инструментов (Tools) для ReAct-агента на LangChain."""

from __future__ import annotations

import base64
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from ar.renderer import CompositionRenderer
from ar.scene_session import SceneSession, SceneValidationError
from ar.svg_store import SVGStore


class SearchComponentsArgs(BaseModel):
    query: str = Field(..., description="Поисковый запрос: метафора, название или описание нужных компонентов.")
    top_k: int = Field(5, description="Максимальное количество возвращаемых кандидатов.")


class CreateSceneArgs(BaseModel):
    width: float = Field(1200.0, description="Ширина холста.")
    height: float = Field(800.0, description="Высота холста.")
    grid_columns: int = Field(3, description="Количество столбцов сетки.")
    grid_rows: int = Field(3, description="Количество строк сетки.")
    # ИСПРАВЛЕНИЕ: Используем общий словарь вместо вложенной модели, чтобы LangChain не путался
    placements: list[dict[str, Any]] = Field(default_factory=list, description="Список словарей для первоначального размещения объектов.")
    palette_name: str | None = Field(None, description="Имя пресета палитры.")


class ApplySceneDiffArgs(BaseModel):
    operations: list[dict[str, Any]] = Field(
        ...,
        description="Список операций диффа (op: 'add_item', 'update_item' и т.д.)."
    )
    commit_message: str = Field("Применение диффа", description="Комментарий к шагу истории.")


class RevertToStepArgs(BaseModel):
    step_id: int = Field(..., description="ID шага истории, к которому нужно откатиться.")


class ToolRegistry:
    """Фабрика инструментов для LangChain-агента."""

    def __init__(
        self,
        session: SceneSession,
        store: SVGStore,
        renderer: CompositionRenderer,
    ) -> None:
        self.session = session
        self.store = store
        self.renderer = renderer

    def get_tools(self) -> list[StructuredTool]:
        return [
            self._make_search_tool(),
            self._make_create_scene_tool(),
            self._make_apply_diff_tool(),
            self._make_inspect_scene_tool(),
            *self._make_history_tools(), # ИСПРАВЛЕНИЕ: распаковка списка
        ]

    def _make_search_tool(self) -> StructuredTool:
        def search_components(query: str, top_k: int = 5) -> str:
            images = self.store.search(query, limit=top_k)
            # ИСПРАВЛЕНИЕ: Добавляем найденных кандидатов в сессию, чтобы VLM могла их увидеть
            self.session.update_candidates([img.id for img in images])
            
            if not images:
                return "В библиотеке не найдено подходящих компонентов."
            
            lines = [f"Найдено {len(images)} компонентов:"]
            for img in images:
                lines.append(
                    f"- image_id: {img.id!r} | Название: {img.name or 'Без названия'} | "
                    f"Размер: {img.width:.0f}x{img.height:.0f}"
                )
            return "\n".join(lines)

        return StructuredTool(
            name="search_components",
            description="Ищет подходящие SVG-компоненты в библиотеке по названию, описанию или метафоре.",
            args_schema=SearchComponentsArgs,
            func=search_components,
        )

    def _make_create_scene_tool(self) -> StructuredTool:
        def create_scene(
            width: float = 1200.0,
            height: float = 800.0,
            grid_columns: int = 3,
            grid_rows: int = 3,
            placements: list[dict[str, Any]] | None = None,
            palette_name: str | None = None,
        ) -> str:
            try:
                self.session.create_scene(
                    width=width,
                    height=height,
                    grid_columns=grid_columns,
                    grid_rows=grid_rows,
                    placements=placements or [],
                    palette=palette_name,
                    commit_message="Создание новой сцены агентом",
                )
                summary = self.session.get_scene_summary()
                return f"Сцена успешно создана! Шаг: {summary['step_id']}. Объектов в сцене: {sum(len(l['items']) for l in summary['layers'])}."
            except SceneValidationError as e:
                return f"ОШИБКА ВАЛИДАЦИИ при создании сцены:\n{e}"
            except Exception as e:
                return f"ОШИБКА: {e}"

        return StructuredTool(
            name="create_scene_from_scratch",
            description="Создает новую сцену с нуля с заданным размером холста, сеткой и начальными объектами.",
            args_schema=CreateSceneArgs,
            func=create_scene,
        )

    def _make_apply_diff_tool(self) -> StructuredTool:
        def apply_diff(operations: list[dict[str, Any]], commit_message: str = "Применение диффа") -> str:
            try:
                entry = self.session.apply_diff(operations=operations, commit_message=commit_message)
                return f"✅ Дифф успешно применён! Шаг #{entry.step_id}."
            except SceneValidationError as e:
                return f"⛔ ДИФФ ОТКЛОНЁН (сцена осталась без изменений):\n{e}"
            except Exception as e:
                return f"⛔ ОШИБКА при выполнении операции: {e}"

        return StructuredTool(
            name="apply_scene_diff",
            description="Применяет транзакционный список операций к текущей сцене.",
            args_schema=ApplySceneDiffArgs,
            func=apply_diff,
        )

    def _make_inspect_scene_tool(self) -> StructuredTool:
        def inspect_and_validate_scene() -> str:
            summary = self.session.get_scene_summary()
            report = self.session.composition.validate(self.store.get)
            
            text_info = [
                "📊 ИНСПЕКЦИЯ СЦЕНЫ:",
                f"- Размер холста: {summary['canvas']['width']}x{summary['canvas']['height']}",
                f"- Сетка: {summary['grid']['columns']}x{summary['grid']['rows']}",
                f"- Статус валидации: {'✅ Валидна' if report.ok else '❌ Есть ошибки'}",
            ]
            
            if report.out_of_bounds:
                text_info.append(f"🚨 Выход за границы: {report.out_of_bounds}")
            if report.overlaps:
                text_info.append(f"🚨 Пересечения в слоях: {[ov.first for ov in report.overlaps]}")
                
            # ИСПРАВЛЕНИЕ: Убрали возврат картинки Base64 строкой, чтобы не убить агента!
            return "\n".join(text_info)

        return StructuredTool(
            name="inspect_and_validate_scene",
            description="Проверяет текущую сцену на пересечения и границы. Возвращает ТОЛЬКО текстовый отчет.",
            func=inspect_and_validate_scene,
        )

    def get_tools_documentation(self) -> str:
        lines = ["## Доступные инструменты:\n"]
        for tool in self.get_tools():
            lines.append(f"### {tool.name}")
            lines.append(f"{tool.description}")
            if hasattr(tool, "args_schema") and tool.args_schema:
                schema = tool.args_schema.model_json_schema()
                props = schema.get("properties", {})
                if props:
                    lines.append("Аргументы:")
                    for arg_name, arg_info in props.items():
                        arg_type = arg_info.get("type", "any")
                        arg_desc = arg_info.get("description", "Без описания")
                        lines.append(f"  - {arg_name} ({arg_type}): {arg_desc}")
            lines.append("")
        return "\n".join(lines)

    def _make_history_tools(self) -> list[StructuredTool]:
        def list_history() -> str:
            history = self.session.get_history()
            lines = ["📜 История изменений сцены:"]
            for h in history:
                curr = "👉" if h["is_current"] else "  "
                lines.append(f"{curr} Шаг #{h['step_id']}: {h['message']}")
            return "\n".join(lines)

        def revert_to_step(step_id: int) -> str:
            try:
                entry = self.session.revert_to_step(step_id)
                return f"✅ Откат к шагу #{entry.step_id} выполнен. Текущая сцена восстановлена."
            except Exception as e:
                return f"⛔ Ошибка отката: {e}"

        return [
            StructuredTool(name="list_history", description="Показывает историю.", func=list_history),
            StructuredTool(name="revert_to_step", description="Откатывает сцену.", args_schema=RevertToStepArgs, func=revert_to_step),
        ]