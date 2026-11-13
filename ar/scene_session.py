"""Менеджер сессии сцены: история шагов (Undo / Redo / Revert) и транзакционное применение диффов.

Отвечает за:
- хранение текущей композиции (Composition), выбранной палитры и пула кандидатов (image_id);
- ведение истории изменений (HistoryEntry) с возможностью отката на любой шаг по его step_id;
- транзакционное применение диффов (apply_diff): все операции сначала применяются к копии
  композиции, затем вызывается validate(). Если обнаружены нарушения границ или пересечения
  внутри одного слоя, дифф отклоняется (рабочая сцена не ломается) и выбрасывается
  SceneValidationError с подробным описанием проблемы и координатами для самоисправления агента.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

from ar.composition import (
    Composition,
    CompositionError,
    GetImage,
    Grid,
    Layer,
    Placement,
    ValidationReport,
)
from ar.palette import PRESET_PALETTES, Palette, PaletteError


class SceneSessionError(Exception):
    """Базовая ошибка сессии сцены."""


class SceneValidationError(SceneSessionError):
    """Дифф привёл к невалидной композиции (пересечения в слое или выход за границы холста)."""

    def __init__(self, message: str, report: ValidationReport) -> None:
        super().__init__(message)
        self.report = report


@dataclass(frozen=True)
class HistoryEntry:
    """Один сохранённый шаг в истории изменений сцены."""

    step_id: int
    timestamp: str
    message: str
    snapshot: dict[str, Any]
    active_palette: dict[str, Any] | None = None
    candidates: tuple[str, ...] = ()


OpType = Literal[
    "add_item",
    "update_item",
    "move_item_cell",
    "move_item_delta",
    "remove_item",
    "duplicate_item",
    "reorder_item",
    "move_item_to_layer",
    "add_layer",
    "remove_layer",
    "update_layer",
    "reorder_layer",
    "set_canvas",
    "set_grid",
    "apply_palette",
]


@dataclass
class DiffOperation:
    """Одна атомарная операция внутри диффа сцены.

    Поддерживаемые типы op и их параметры в params:
    - 'add_item': image_id, x?, y?, cell? ([col, row]), scale?, rotation?, colors?, layer?
    - 'update_item': item_id, x?, y?, scale?, rotation?, colors?, layer?
    - 'move_item_cell': item_id, column, row
    - 'move_item_delta': item_id, dx?, dy?
    - 'remove_item': item_id
    - 'duplicate_item': item_id, dx?, dy?
    - 'reorder_item': item_id, action ('front' | 'back' | 'up' | 'down' | 'index'), index?
    - 'move_item_to_layer': item_id, layer, index?
    - 'add_layer': name, allow_internal_overlap?, strict_bounds?, index?
    - 'remove_layer': layer (id или имя)
    - 'update_layer': layer (id или имя), name?, allow_internal_overlap?, strict_bounds?, visible?
    - 'reorder_layer': layer (id или имя), index
    - 'set_canvas': width?, height?, strict_bounds?, name?
    - 'set_grid': columns, rows
    - 'apply_palette': palette_name? (из PRESET_PALETTES) ИЛИ colors? (список HEX-цветов), name?
    """

    op: OpType
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DiffOperation:
        if "op" not in data:
            raise SceneSessionError("В операции диффа отсутствует обязательное поле 'op'")
        op_type = data["op"]
        if "params" in data and isinstance(data["params"], dict):
            params = dict(data["params"])
        else:
            params = {k: v for k, v in data.items() if k != "op"}
        return cls(op=op_type, params=params)


class SceneSession:
    """Управляет жизненным циклом сцены, пулом кандидатов, транзакционными диффами и историей шагов."""

    def __init__(
        self,
        get_image: GetImage,
        composition: Composition | None = None,
        *,
        initial_message: str = "Инициализация пустой сцены",
    ) -> None:
        self.get_image = get_image
        self._comp: Composition = composition if composition is not None else Composition()
        self._active_palette: Palette | None = None
        self._candidates: list[str] = []

        self._history: list[HistoryEntry] = []
        self._cursor: int = -1  # индекс текущего активного шага в self._history
        self._next_step_id: int = 0

        self._record_snapshot(initial_message)

    # ---------- доступ к текущему состоянию ----------

    @property
    def composition(self) -> Composition:
        return self._comp

    @property
    def active_palette(self) -> Palette | None:
        return self._active_palette

    @property
    def candidates(self) -> list[str]:
        """Список image_id компонентов, отобранных для работы со сценой."""
        return list(self._candidates)

    @property
    def current_step_id(self) -> int:
        return self._history[self._cursor].step_id

    # ---------- управление пулом кандидатов ----------

    def set_candidates(self, image_ids: list[str]) -> list[str]:
        """Полностью заменяет рабочий пул кандидатов (проверяя, что такие иллюстрации существуют)."""
        unique: list[str] = []
        for img_id in image_ids:
            self.get_image(img_id)  # проверит существование в хранилище
            if img_id not in unique:
                unique.append(img_id)
        self._candidates = unique
        return list(self._candidates)

    def update_candidates(
        self,
        add_ids: list[str] | None = None,
        remove_ids: list[str] | None = None,
    ) -> list[str]:
        """Добавляет и/или удаляет элементы из рабочего пула кандидатов."""
        remove_set = set(remove_ids or [])
        self._candidates = [cid for cid in self._candidates if cid not in remove_set]
        for img_id in add_ids or []:
            self.get_image(img_id)
            if img_id not in self._candidates:
                self._candidates.append(img_id)
        return list(self._candidates)

    # ---------- история изменений (Undo / Redo / Revert) ----------

    def _record_snapshot(self, message: str) -> HistoryEntry:
        # Если мы откатывались назад и делаем новое изменение, отрезаем неактуальную ветку Redo
        if self._cursor < len(self._history) - 1:
            self._history = self._history[: self._cursor + 1]

        entry = HistoryEntry(
            step_id=self._next_step_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            message=message,
            snapshot=self._comp.to_dict(),
            active_palette=self._active_palette.to_dict() if self._active_palette else None,
            candidates=tuple(self._candidates),
        )
        self._next_step_id += 1
        self._history.append(entry)
        self._cursor = len(self._history) - 1
        return entry

    def _restore_entry(self, entry: HistoryEntry) -> None:
        self._comp = Composition.from_dict(entry.snapshot)
        self._active_palette = (
            Palette.from_dict(entry.active_palette) if entry.active_palette else None
        )
        self._candidates = list(entry.candidates)

    def get_history(self) -> list[dict[str, Any]]:
        """Возвращает список всех шагов истории с отметкой текущего активного шага."""
        result: list[dict[str, Any]] = []
        for idx, entry in enumerate(self._history):
            layers_data = entry.snapshot.get("layers", [])
            total_items = sum(len(l.get("items", [])) for l in layers_data)
            result.append(
                {
                    "step_id": entry.step_id,
                    "timestamp": entry.timestamp,
                    "message": entry.message,
                    "is_current": idx == self._cursor,
                    "layers_count": len(layers_data),
                    "items_count": total_items,
                }
            )
        return result

    def undo(self) -> HistoryEntry:
        """Откатывает сцену на один шаг назад."""
        if self._cursor <= 0:
            raise SceneSessionError("Невозможно выполнить undo: мы уже на самом первом шаге")
        self._cursor -= 1
        entry = self._history[self._cursor]
        self._restore_entry(entry)
        return entry

    def redo(self) -> HistoryEntry:
        """Возвращает сцену на один шаг вперёд (если ранее делался undo)."""
        if self._cursor >= len(self._history) - 1:
            raise SceneSessionError("Невозможно выполнить redo: нет отменённых шагов впереди")
        self._cursor += 1
        entry = self._history[self._cursor]
        self._restore_entry(entry)
        return entry

    def revert_to_step(self, step_id: int) -> HistoryEntry:
        """Переключает состояние сцены к конкретному шагу по его step_id."""
        for idx, entry in enumerate(self._history):
            if entry.step_id == step_id:
                self._cursor = idx
                self._restore_entry(entry)
                return entry
        available = [e.step_id for e in self._history]
        raise SceneSessionError(
            f"Шаг с step_id={step_id} не найден. Доступные шаги: {available}"
        )

    # ---------- форматирование детальных ошибок валидации для агента ----------

    def format_validation_errors(
        self,
        comp: Composition,
        report: ValidationReport,
    ) -> str:
        """Собирает понятное диагностическое сообщение с координатами и размерами проблемных объектов."""
        if report.ok:
            return "Композиция валидна (нет пересечений внутри слоёв и выхода за границы холста)."

        lines: list[str] = [
            f"Ошибка валидации композиции (холст {comp.width:g}x{comp.height:g}, "
            f"сетка {comp.grid.columns}x{comp.grid.rows}):"
        ]

        if report.out_of_bounds:
            lines.append("1. Выход объектов за границы холста (out_of_bounds):")
            for item_id in report.out_of_bounds:
                item = comp.get(item_id)
                layer = comp.get_item_layer(item_id)
                img = self.get_image(item.image_id)
                x0, y0, x1, y1 = comp.bbox(item_id, self.get_image)
                mode_str = "жёсткая рамка" if layer.strict_bounds else "гибкая рамка (центр внутри)"
                lines.append(
                    f"   - item_id={item_id!r} ({img.name or item.image_id}, слой={layer.name!r}, {mode_str}): "
                    f"bbox=[{x0:.1f}, {y0:.1f}, {x1:.1f}, {y1:.1f}] выходит за пределы "
                    f"[0, 0, {comp.width:g}, {comp.height:g}]. "
                    f"Подсказка: сдвинь объект ближе к центру или уменьши его scale (сейчас scale={item.scale:g})."
                )

        if report.overlaps:
            lines.append("2. Недопустимые пересечения объектов внутри одного слоя (overlaps):")
            for ov in report.overlaps:
                layer = comp.get_layer(ov.layer_id)
                item_a = comp.get(ov.first)
                item_b = comp.get(ov.second)
                img_a = self.get_image(item_a.image_id)
                img_b = self.get_image(item_b.image_id)
                box_a = comp.bbox(ov.first, self.get_image)
                box_b = comp.bbox(ov.second, self.get_image)
                lines.append(
                    f"   - В слое {layer.name!r} пересекаются: "
                    f"item_id={ov.first!r} ({img_a.name or item_a.image_id}, "
                    f"bbox=[{box_a[0]:.1f}, {box_a[1]:.1f}, {box_a[2]:.1f}, {box_a[3]:.1f}]) И "
                    f"item_id={ov.second!r} ({img_b.name or item_b.image_id}, "
                    f"bbox=[{box_b[0]:.1f}, {box_b[1]:.1f}, {box_b[2]:.1f}, {box_b[3]:.1f}]), "
                    f"площадь пересечения={ov.area:.1f} px². "
                    f"Подсказка: разведи их по разным клеткам сетки/координатам, уменьши scale "
                    f"или перенеси фоновый объект в другой слой (move_item_to_layer)."
                )

        return "\n".join(lines)

    # ---------- создание новой сцены с нуля ----------

    def create_scene(
        self,
        *,
        width: float = 1200.0,
        height: float = 800.0,
        grid_columns: int = 3,
        grid_rows: int = 3,
        strict_bounds: bool = True,
        name: str | None = None,
        layers: list[dict[str, Any]] | None = None,
        placements: list[dict[str, Any]] | None = None,
        palette: Palette | str | list[str] | None = None,
        commit_message: str = "Создание новой сцены с нуля",
        strict_validation: bool = True,
    ) -> HistoryEntry:
        """Транзакционно создаёт новую сцену с нуля.

        Если новая сцена не проходит validate() (при strict_validation=True),
        старая сцена остаётся нетронутой и выбрасывается SceneValidationError.
        """
        draft = Composition(
            width=width,
            height=height,
            grid=Grid(columns=grid_columns, rows=grid_rows),
            strict_bounds=strict_bounds,
            name=name,
        )

        # Если переданы пользовательские слои (например: Background, Main, Foreground)
        if layers:
            draft.layers.clear()
            for l_spec in layers:
                draft.add_layer(
                    name=l_spec.get("name", "Layer"),
                    allow_internal_overlap=bool(l_spec.get("allow_internal_overlap", False)),
                    strict_bounds=l_spec.get("strict_bounds", strict_bounds),
                )

        # Размещаем начальные объекты
        for p_spec in placements or []:
            self._apply_add_item(draft, p_spec)

        # Применяем палитру, если указана
        resolved_palette = self._resolve_palette(palette) if palette is not None else self._active_palette
        if resolved_palette is not None:
            resolved_palette.apply_to_composition(draft, self.get_image)

        report = draft.validate(self.get_image)
        if strict_validation and not report.ok:
            err_msg = self.format_validation_errors(draft, report)
            raise SceneValidationError(
                f"Новая сцена не создана! Изменения отклонены.\n{err_msg}",
                report=report,
            )

        self._comp = draft
        if resolved_palette is not None:
            self._active_palette = resolved_palette
        return self._record_snapshot(commit_message)

    # ---------- транзакционное применение диффа ----------

    def apply_diff(
        self,
        operations: list[DiffOperation | dict[str, Any]],
        *,
        commit_message: str = "Применение диффа сцены",
        strict_validation: bool = True,
    ) -> HistoryEntry:
        """Транзакционно применяет список операций к копии текущей сцены.

        1. Клонирует текущую композицию в песочницу (draft).
        2. Последовательно выполняет все операции.
        3. Запускает draft.validate(self.get_image).
        4. Если есть ошибки валидации (и strict_validation=True) — отклоняет дифф,
           оставляя текущую сцену без изменений, и бросает SceneValidationError.
        5. Если всё ок — сохраняет новую сцену и записывает шаг в историю.
        """
        if not operations:
            raise SceneSessionError("Список операций диффа не может быть пустым")

        parsed_ops = [
            op if isinstance(op, DiffOperation) else DiffOperation.from_dict(op)
            for op in operations
        ]

        draft = Composition.from_dict(self._comp.to_dict())
        draft_palette = self._active_palette

        try:
            for op_obj in parsed_ops:
                draft_palette = self._execute_single_op(draft, draft_palette, op_obj)
        except (CompositionError, PaletteError, KeyError, TypeError, ValueError) as exc:
            raise SceneSessionError(
                f"Ошибка при выполнении операции {op_obj.op!r} с параметрами {op_obj.params}: {exc}"
            ) from exc

        report = draft.validate(self.get_image)
        if strict_validation and not report.ok:
            err_msg = self.format_validation_errors(draft, report)
            raise SceneValidationError(
                f"Дифф отклонён! Сцена осталась на шаге #{self.current_step_id}.\n{err_msg}",
                report=report,
            )

        self._comp = draft
        self._active_palette = draft_palette
        return self._record_snapshot(commit_message)

    # ---------- выполнение отдельных операций диффа ----------

    def _resolve_palette(self, raw: Palette | str | list[str] | dict[str, Any]) -> Palette:
        if isinstance(raw, Palette):
            return raw
        if isinstance(raw, str):
            key = raw.strip().lower()
            if key in PRESET_PALETTES:
                return PRESET_PALETTES[key]
            raise PaletteError(
                f"Неизвестный пресет палитры {raw!r}. Доступные пресеты: {list(PRESET_PALETTES)}"
            )
        if isinstance(raw, list):
            return Palette(raw)
        if isinstance(raw, dict):
            if "palette_name" in raw and raw["palette_name"]:
                return self._resolve_palette(raw["palette_name"])
            return Palette.from_dict(raw)
        raise PaletteError(f"Некорректный формат палитры: {raw!r}")

    def _apply_add_item(self, draft: Composition, p: dict[str, Any]) -> Placement:
        image_id = p["image_id"]
        self.get_image(image_id)  # проверяем, что иллюстрация существует в базе

        scale = float(p.get("scale", 1.0))
        rotation = float(p.get("rotation", 0.0))
        colors = p.get("colors")
        layer = p.get("layer")

        item = draft.add(
            image_id=image_id,
            x=float(p.get("x", 0.0)),
            y=float(p.get("y", 0.0)),
            scale=scale,
            rotation=rotation,
            colors=colors,
            layer=layer,
        )

        # Если передана клетка сетки cell=[col, row] или column/row — центрируем объект в этой клетке
        if "cell" in p and p["cell"] is not None:
            col, row = int(p["cell"][0]), int(p["cell"][1])
            draft.move_to_cell(item.id, col, row, self.get_image)
        elif "column" in p and "row" in p and p["column"] is not None and p["row"] is not None:
            draft.move_to_cell(item.id, int(p["column"]), int(p["row"]), self.get_image)

        return item

    def _execute_single_op(
        self,
        draft: Composition,
        current_palette: Palette | None,
        op_obj: DiffOperation,
    ) -> Palette | None:
        op = op_obj.op
        p = op_obj.params

        if op == "add_item":
            item = self._apply_add_item(draft, p)
            if current_palette is not None and not p.get("colors"):
                img = self.get_image(item.image_id)
                draft.update(item.id, colors=current_palette.map_colors(img.colors))

        elif op == "update_item":
            item_id = p["item_id"]
            draft.update(
                item_id,
                x=float(p["x"]) if p.get("x") is not None else None,
                y=float(p["y"]) if p.get("y") is not None else None,
                scale=float(p["scale"]) if p.get("scale") is not None else None,
                rotation=float(p["rotation"]) if p.get("rotation") is not None else None,
                colors=p.get("colors"),
                layer=p.get("layer"),
            )
            if "cell" in p and p["cell"] is not None:
                col, row = int(p["cell"][0]), int(p["cell"][1])
                draft.move_to_cell(item_id, col, row, self.get_image)
            elif "column" in p and "row" in p and p["column"] is not None and p["row"] is not None:
                draft.move_to_cell(item_id, int(p["column"]), int(p["row"]), self.get_image)

        elif op == "move_item_cell":
            item_id = p["item_id"]
            col = int(p["column"] if "column" in p else p["col"])
            row = int(p["row"])
            draft.move_to_cell(item_id, col, row, self.get_image)

        elif op == "move_item_delta":
            item_id = p["item_id"]
            item = draft.get(item_id)
            dx = float(p.get("dx", 0.0))
            dy = float(p.get("dy", 0.0))
            draft.update(item_id, x=item.x + dx, y=item.y + dy)

        elif op == "remove_item":
            draft.remove(p["item_id"])

        elif op == "duplicate_item":
            draft.duplicate(
                p["item_id"],
                dx=float(p.get("dx", 20.0)),
                dy=float(p.get("dy", 20.0)),
            )

        elif op == "reorder_item":
            item_id = p["item_id"]
            action = str(p.get("action", "front")).lower()
            if action == "front":
                draft.bring_to_front(item_id)
            elif action == "back":
                draft.send_to_back(item_id)
            elif action == "up":
                draft.move_up(item_id)
            elif action == "down":
                draft.move_down(item_id)
            elif action == "index":
                draft.move_to_layer(item_id, int(p["index"]))
            else:
                raise CompositionError(f"Неизвестное действие reorder_item: {action!r}")

        elif op == "move_item_to_layer":
            draft.move_item_to_layer(
                p["item_id"],
                p["layer"],
                index=int(p["index"]) if p.get("index") is not None else None,
            )

        elif op == "add_layer":
            draft.add_layer(
                name=p["name"],
                allow_internal_overlap=bool(p.get("allow_internal_overlap", False)),
                strict_bounds=p.get("strict_bounds"),
                index=int(p["index"]) if p.get("index") is not None else None,
            )

        elif op == "remove_layer":
            draft.remove_layer(p["layer"])

        elif op == "update_layer":
            layer_obj: Layer = draft.get_layer(p["layer"])
            if p.get("name") is not None:
                layer_obj.name = str(p["name"])
            if p.get("allow_internal_overlap") is not None:
                layer_obj.allow_internal_overlap = bool(p["allow_internal_overlap"])
            if p.get("strict_bounds") is not None:
                layer_obj.strict_bounds = bool(p["strict_bounds"])
            if p.get("visible") is not None:
                layer_obj.visible = bool(p["visible"])

        elif op == "reorder_layer":
            draft.reorder_layer(p["layer"], int(p["index"]))

        elif op == "set_canvas":
            if p.get("width") is not None:
                w = float(p["width"])
                if w <= 0:
                    raise CompositionError("Ширина холста должна быть больше 0")
                draft.width = w
            if p.get("height") is not None:
                h = float(p["height"])
                if h <= 0:
                    raise CompositionError("Высота холста должна быть больше 0")
                draft.height = h
            if p.get("strict_bounds") is not None:
                sb = bool(p["strict_bounds"])
                draft.strict_bounds = sb
                if p.get("apply_to_all_layers", False):
                    for l_obj in draft.layers:
                        l_obj.strict_bounds = sb
            if p.get("name") is not None:
                draft.name = str(p["name"])

        elif op == "set_grid":
            draft.grid = Grid(columns=int(p["columns"]), rows=int(p["rows"]))

        elif op == "apply_palette":
            current_palette = self._resolve_palette(p)
            current_palette.apply_to_composition(draft, self.get_image)

        else:
            raise SceneSessionError(f"Неизвестный тип операции диффа: {op!r}")

        return current_palette

    # ---------- сводка состояния сцены для промпта агента ----------

    def get_scene_summary(self) -> dict[str, Any]:
        """Возвращает подробное описание текущей сцены (с вычисленными bbox и клетками сетки),
        которое удобно передавать в системный промпт агента.
        """
        comp = self._comp
        cell_w, cell_h = comp.cell_size()
        report = comp.validate(self.get_image)

        layers_info: list[dict[str, Any]] = []
        for l_idx, layer in enumerate(comp.layers):
            items_info: list[dict[str, Any]] = []
            for i_idx, item in enumerate(layer.items):
                img = self.get_image(item.image_id)
                x0, y0, x1, y1 = comp.bbox(item.id, self.get_image)
                cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                approx_col = min(max(int(cx // cell_w), 0), comp.grid.columns - 1)
                approx_row = min(max(int(cy // cell_h), 0), comp.grid.rows - 1)
                items_info.append(
                    {
                        "item_id": item.id,
                        "image_id": item.image_id,
                        "image_name": img.name,
                        "order_in_layer": i_idx,
                        "x": round(item.x, 2),
                        "y": round(item.y, 2),
                        "scale": round(item.scale, 3),
                        "rotation": round(item.rotation, 2),
                        "bbox": [round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)],
                        "center": [round(cx, 1), round(cy, 1)],
                        "approx_cell": [approx_col, approx_row],
                        "colors_map": dict(item.colors),
                    }
                )
            layers_info.append(
                {
                    "layer_id": layer.id,
                    "name": layer.name,
                    "layer_order": l_idx,
                    "allow_internal_overlap": layer.allow_internal_overlap,
                    "strict_bounds": layer.strict_bounds,
                    "visible": layer.visible,
                    "items": items_info,
                }
            )

        return {
            "step_id": self.current_step_id,
            "canvas": {
                "width": comp.width,
                "height": comp.height,
                "strict_bounds": comp.strict_bounds,
            },
            "grid": {
                "columns": comp.grid.columns,
                "rows": comp.grid.rows,
                "cell_width": round(cell_w, 1),
                "cell_height": round(cell_h, 1),
            },
            "active_palette": self._active_palette.to_dict() if self._active_palette else None,
            "candidate_pool": list(self._candidates),
            "layers": layers_info,
            "validation": {
                "ok": report.ok,
                "out_of_bounds": list(report.out_of_bounds),
                "overlaps": [
                    {
                        "layer_id": ov.layer_id,
                        "first": ov.first,
                        "second": ov.second,
                        "area": round(ov.area, 1),
                    }
                    for ov in report.overlaps
                ],
            },
            "recent_history": self.get_history()[-5:],
        }
