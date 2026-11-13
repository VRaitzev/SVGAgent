"""Реестр инструментов (Tools) для ReAct-агента на LangChain."""

from __future__ import annotations

from typing import Any, Literal

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from renderer import CompositionRenderer
from scene_session import SceneSession, SceneValidationError
from svg_store import SVGStore


class EmptyArgs(BaseModel):
    pass


class SearchComponentsArgs(BaseModel):
    query: str = Field(..., description="Поисковый запрос: метафора, название или описание нужных компонентов.")
    detail_level: Literal["L", "S"] | None = Field(
        None,
        description="Фильтр по уровню детализации ('L' — крупный/детализированный, 'S' — компактный/упрощённый).",
    )
    role: Literal["main", "background"] | None = Field(
        None,
        description="Фильтр по роли объекта: 'main' (главный цветной объект) или 'background' (фоновый монохромный объект).",
    )
    top_k: int | None = Field(None, description="Количество возвращаемых кандидатов (если не задано — используется настройка UI).")


class CreateSceneArgs(BaseModel):
    width: float = Field(800.0, description="Ширина холста в пикселях (фиксировано 800).")
    height: float = Field(800.0, description="Высота холста в пикселях (фиксировано 800).")
    grid_columns: int = Field(3, description="Количество столбцов сетки.")
    grid_rows: int = Field(3, description="Количество строк сетки.")
    detail_level: Literal["L", "S"] = Field("L", description="Уровень детализации композиции: 'L' или 'S'.")
    uniform_scale: float = Field(1.0, description="Единый относительный масштаб для всех объектов сцены. Подбирай так, чтобы группа главных объектов занимала ~60–80% ширины холста (считай по реальным размерам ассетов).")
    strict_bounds: bool = Field(False, description="False разрешает композиционную обрезку краями холста (>=30% внутри кадра).")
    layers: list[dict[str, Any]] | None = Field(
        default=None,
        description="Список слоёв от нижнего к верхнему ('Background' и 'Main').",
    )
    placements: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Список объектов. Задавай продуманные координаты верхнего левого угла {'image_id': '...', 'layer': 'Main', 'x': 120, 'y': 180} "
            "с учётом реальных размеров объекта (width * uniform_scale, height * uniform_scale), чтобы сразу избежать наложений."
        ),
    )
    palette_name: str | None = Field(
        "lavender_mist",
        description="Имя цветовой схемы ('lavender_mist', 'peach_sunset', 'mint_breeze').",
    )


class ApplySceneDiffArgs(BaseModel):
    operations: list[dict[str, Any]] = Field(
        ...,
        description=(
            "ПАКЕТНЫЙ (КОМПЛЕКСНЫЙ) список всех операций за один шаг! "
            "Передавай СРАЗУ ВСЕ нужные сдвиги, добавления, смены ролей и масштаба в одном массиве operations, "
            "не дроби правки на отдельные вызовы. "
            "Формат каждого элемента: {'op': '<имя_операции>', 'params': {...}}. "
            "Примеры комбо-пакета в одном вызове: [\n"
            "  {'op': 'move_item_delta', 'params': {'item_id': 'id1', 'dx': -40, 'dy': 20}},\n"
            "  {'op': 'update_item', 'params': {'item_id': 'id2', 'x': 480, 'y': 420}},\n"
            "  {'op': 'toggle_item_role', 'params': {'item_id': 'id3'}},\n"
            "  {'op': 'add_item', 'params': {'image_id': 'img4', 'layer': 'Background', 'x': 80, 'y': 520}}\n"
            "]."
        ),
    )
    commit_message: str = Field("Комплексная правка сцены", description="Краткое описание всех правок в пакете.")


class RevertToStepArgs(BaseModel):
    step_id: int = Field(..., description="ID шага истории, к которому нужно откатиться.")


def _calc_bbox_overlap(b1: list[float], b2: list[float]) -> tuple[float, float]:
    ix1 = max(b1[0], b2[0])
    iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2])
    iy2 = min(b1[3], b2[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0, 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area1 = max(1.0, (b1[2] - b1[0]) * (b1[3] - b1[1]))
    area2 = max(1.0, (b2[2] - b2[0]) * (b2[3] - b2[1]))
    ratio = inter / min(area1, area2)
    return inter, ratio


class ToolRegistry:
    """Фабрика инструментов для LangChain-агента."""

    def __init__(
        self,
        session: SceneSession,
        store: SVGStore,
        renderer: CompositionRenderer,
        *,
        library_mode: Literal["full_catalog", "rag"] = "full_catalog",
        rag_top_k: int = 10,
        multimodal_tools: bool = False,
    ) -> None:
        self.session = session
        self.store = store
        self.renderer = renderer
        self.library_mode: Literal["full_catalog", "rag"] = library_mode
        self.rag_top_k: int = rag_top_k
        self.multimodal_tools: bool = multimodal_tools

        # Бюджет одного запроса пользователя: сколько раз агент может ИЗМЕНИТЬ сцену
        # (create_scene_from_scratch + apply_scene_diff считаются вместе).
        self.max_edit_steps: int = 6
        self.edits_used: int = 0
        self.inspects_used: int = 0

    def reset_budget(self, max_edit_steps: int | None = None) -> None:
        """Вызывается в начале каждого запроса пользователя."""
        if max_edit_steps is not None:
            self.max_edit_steps = max(1, int(max_edit_steps))
        self.edits_used = 0
        self.inspects_used = 0

    @property
    def edits_left(self) -> int:
        return max(0, self.max_edit_steps - self.edits_used)

    def _budget_suffix(self) -> str:
        if self.edits_left > 0:
            return f"\n🧮 Использовано правок: {self.edits_used}/{self.max_edit_steps} (осталось {self.edits_left})."
        return (
            f"\n🛑 Лимит правок исчерпан ({self.edits_used}/{self.max_edit_steps}). "
            "Больше НЕ вызывай create_scene_from_scratch и apply_scene_diff — напиши краткий итог пользователю."
        )

    def _limit_reached_msg(self) -> str:
        return (
            f"🛑 Лимит правок исчерпан ({self.edits_used}/{self.max_edit_steps}). Сцена оставлена в текущем состоянии. "
            "НЕ вызывай больше инструменты изменения — сразу напиши пользователю краткий итог: что получилось и что не удалось довести."
        )

    def get_tools(self) -> list[StructuredTool]:
        return [
            self._make_search_tool(),
            self._make_create_scene_tool(),
            self._make_apply_diff_tool(),
            self._make_inspect_scene_tool(),
            *self._make_history_tools(),
        ]

    def _make_search_tool(self) -> StructuredTool:
        def search_components(
            query: str,
            detail_level: Literal["L", "S"] | None = None,
            role: Literal["main", "background"] | None = None,
            top_k: int | None = None,
        ) -> str:
            effective_level = detail_level or self.session.composition.detail_level
            if self.library_mode == "full_catalog":
                images = self.store.list(limit=200, detail_level=effective_level, role=role)
            else:
                k = top_k or self.rag_top_k
                images = self.store.search(query, limit=k, detail_level=effective_level, role=role)

            self.session.clear_candidates()
            self.session.update_candidates([img.id for img in images])
            if not images:
                return f"В библиотеке не найдено компонентов (detail_level={effective_level}, role={role})."

            lines = [f"Доступно {len(images)} компонентов (размер={effective_level}, режим={self.library_mode}):"]
            for img in images:
                lines.append(
                    f"- image_id: {img.id!r} | Название: {img.name or 'Без названия'} | "
                    f"Роль: {img.role} | Исходный размер (при scale=1.0): {img.width:.0f}x{img.height:.0f} px | "
                    f"Концепт: {img.concept_key} | Описание: {(img.description or '')[:140]}"
                )
            return "\n".join(lines)

        return StructuredTool(
            name="search_components",
            description="Возвращает SVG-компоненты из библиотеки с их точными габаритами в пикселях (и сжатые превью в VLM-режиме).",
            args_schema=SearchComponentsArgs,
            func=search_components,
        )

    def _make_create_scene_tool(self) -> StructuredTool:
        def create_scene(
            width: float = 800.0,
            height: float = 800.0,
            grid_columns: int = 3,
            grid_rows: int = 3,
            detail_level: Literal["L", "S"] = "L",
            uniform_scale: float = 1.0,
            strict_bounds: bool = False,
            layers: list[dict[str, Any]] | None = None,
            placements: list[dict[str, Any]] | None = None,
            palette_name: str | None = "lavender_mist",
        ) -> str:
            if self.edits_left <= 0:
                return self._limit_reached_msg()
            self.edits_used += 1  # учитываем каждую попытку, даже неудачную — так цикл не станет бесконечным
            try:
                norm_placements = [
                    p.model_dump() if hasattr(p, "model_dump") else dict(p)
                    for p in (placements or [])
                ]
                norm_layers = (
                    [l.model_dump() if hasattr(l, "model_dump") else dict(l) for l in layers]
                    if layers is not None
                    else None
                )
                self.session.create_scene(
                    width=800.0,
                    height=800.0,
                    grid_columns=grid_columns,
                    grid_rows=grid_rows,
                    detail_level=detail_level,
                    uniform_scale=uniform_scale,
                    strict_bounds=strict_bounds,
                    layers=norm_layers,
                    placements=norm_placements,
                    palette=palette_name,
                    commit_message=f"Создание сцены ({detail_level}, палитра {palette_name})",
                )
                summary = self.session.get_scene_summary()
                total_items = sum(len(l["items"]) for l in summary["layers"])
                warn_str = ("\n⚠️ Предупреждения:\n- " + "\n- ".join(summary["warnings"])) if summary.get("warnings") else ""
                return (
                    f"✅ Сцена создана (шаг #{summary['step_id']}, размер={summary['detail_level']}, объектов={total_items}).{warn_str}\n"
                    f"Теперь вызови `inspect_and_validate_scene`, чтобы проверить рендер."
                    f"{self._budget_suffix()}"
                )
            except SceneValidationError as e:
                return f"⛔ ОШИБКА ВАЛИДАЦИИ при создании сцены:\n{e}{self._budget_suffix()}"
            except Exception as e:
                return f"⛔ ОШИБКА: {e}{self._budget_suffix()}"

        return StructuredTool(
            name="create_scene_from_scratch",
            description="Создаёт новую сцену на холсте 800x800 с заданным уровнем детализации ('L' или 'S'), единым масштабом и всеми объектами сразу.",
            args_schema=CreateSceneArgs,
            func=create_scene,
        )

    def _make_apply_diff_tool(self) -> StructuredTool:
        def apply_diff(operations: list[dict[str, Any]], commit_message: str = "Комплексная правка сцены") -> str:
            try:
                normalized_ops: list[dict[str, Any]] = []
                for raw_op in operations:
                    op_dict = raw_op.model_dump() if hasattr(raw_op, "model_dump") else dict(raw_op)
                    if op_dict.get("op") == "set_canvas":
                        continue
                    if "params" not in op_dict or not isinstance(op_dict["params"], dict):
                        op_type = op_dict.pop("op", None)
                        op_dict = {"op": op_type, "params": op_dict}
                    normalized_ops.append(op_dict)

                if not normalized_ops:
                    return "⚠️ Изменение размера холста запрещено (холст фиксирован 800x800). Двигай сами объекты через move_item_delta или update_item."

                if self.edits_left <= 0:
                    return self._limit_reached_msg()
                self.edits_used += 1  # считаем попытку правки (в том числе отклонённую валидацией)

                entry = self.session.apply_diff(
                    operations=normalized_ops,
                    commit_message=commit_message,
                )
                # getattr: если в SceneSession нет last_warnings, дифф уже закоммичен — нельзя сообщать агенту «ошибка»
                warnings = list(getattr(self.session, "last_warnings", None) or [])
                warn_str = ("\n⚠️ Предупреждения:\n- " + "\n- ".join(warnings)) if warnings else ""
                return (
                    f"✅ Комплексный дифф из {len(normalized_ops)} операций применён (шаг #{entry.step_id}: '{commit_message}').{warn_str}\n"
                    f"Вызови `inspect_and_validate_scene` для проверки."
                    f"{self._budget_suffix()}"
                )
            except SceneValidationError as e:
                return f"⛔ ДИФФ ОТКЛОНЁН (сцена осталась без изменений):\n{e}{self._budget_suffix()}"
            except Exception as e:
                return f"⛔ ОШИБКА при выполнении операции: {e}{self._budget_suffix()}"

        return StructuredTool(
            name="apply_scene_diff",
            description=(
                "Применяет ПАКЕТ операций к сцене за один вызов. Всегда объединяй все нужные сдвиги, добавления и смены ролей в один список operations!"
            ),
            args_schema=ApplySceneDiffArgs,
            func=apply_diff,
        )

    def _make_inspect_scene_tool(self) -> StructuredTool:
        def inspect_and_validate_scene() -> str:
            if self.inspects_used >= self.max_edit_steps + 2:
                return "🛑 Лимит проверок исчерпан. НЕ вызывай больше инструменты — сразу напиши краткий итог пользователю."
            self.inspects_used += 1
            summary = self.session.get_scene_summary()
            report = self.session.composition.validate(self.store.get)
            color_errors = self.session._validate_color_rules(self.session.composition, self.session.active_palette)

            text_info = [
                "📊 ИНСПЕКЦИЯ СЦЕНЫ:",
                f"- Шаг истории: #{summary['step_id']} | Уровень детализации: {summary['detail_level']} | Единый масштаб: {summary['uniform_scale']}",
                f"- Размер холста: {summary['canvas']['width']:g}x{summary['canvas']['height']:g}",
                f"- Статус базовой валидации ТЗ: {'✅ Валидна' if (report.ok and not color_errors) else '❌ Есть ошибки'}",
                "- Слои и объекты:",
            ]

            overlap_notes: list[str] = []
            for layer in summary["layers"]:
                items = layer["items"]
                text_info.append(f"  * Слой {layer['name']!r} (объектов={len(items)}):")
                for it in items:
                    text_info.append(
                        f"    - item_id={it['item_id']!r} ({it['image_name']}, role={it['role']}): "
                        f"верхний_левый_угол=({it['x']}, {it['y']}), размер_на_холсте=({it['width']}x{it['height']} px), "
                        f"bbox=[x1={it['bbox'][0]}, y1={it['bbox'][1]}, x2={it['bbox'][2]}, y2={it['bbox'][3]}]"
                    )
                for i in range(len(items)):
                    for j in range(i + 1, len(items)):
                        a, b = items[i], items[j]
                        inter_area, ratio = _calc_bbox_overlap(a["bbox"], b["bbox"])
                        if ratio > 0.15:
                            overlap_notes.append(
                                f"  ℹ️ Пересечение рамок (bbox) в слое {layer['name']!r}: "
                                f"{a['image_name']!r} ({a['item_id']!r}) и {b['image_name']!r} ({b['item_id']!r}) "
                                f"— {ratio*100:.0f}%. Намеренное наложение (мелкий предмет на крупной базе, передний план на фоне) — норма для этого стиля. Исправляй только если на рендере фигуры или тени конфликтуют вопреки физике — одним пакетным apply_scene_diff."
                            )

            if overlap_notes:
                text_info.append("\n🔍 ГЕОМЕТРИЧЕСКАЯ СПРАВКА ПО BBOX:")
                text_info.extend(overlap_notes)

            if summary.get("warnings"):
                text_info.append("\n⚠️ ПРЕДУПРЕЖДЕНИЯ СЕССИИ:")
                for w in summary["warnings"]:
                    text_info.append(f"  - {w}")

            if not report.ok or color_errors:
                text_info.append("\n" + self.session.format_validation_errors(self.session.composition, report, color_errors))

            return "\n".join(text_info)

        return StructuredTool(
            name="inspect_and_validate_scene",
            description="Проверяет сцену по правилам ТЗ и в VLM-режиме прикладывает сжатый читаемый рендер сцены.",
            args_schema=EmptyArgs,
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
                return f"✅ Откат к шагу #{entry.step_id} выполнен."
            except Exception as e:
                return f"⛔ Ошибка отката: {e}"

        return [
            StructuredTool(
                name="list_history",
                description="Показывает историю изменений сцены.",
                args_schema=EmptyArgs,
                func=list_history,
            ),
            StructuredTool(
                name="revert_to_step",
                description="Откатывает сцену к шагу истории.",
                args_schema=RevertToStepArgs,
                func=revert_to_step,
            ),
        ]