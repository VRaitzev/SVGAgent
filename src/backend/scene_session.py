"""Транзакционная сессия редактирования композиции:
- предупреждения при отсутствии парной версии L/S (без тихой рассинхронизации);
- автоматический пересчёт цветов в update_item при смене image_id;
- единый относительный масштаб всех объектов сцены (uniform_scale);
- переключение Главный ('main') <-> Фоновый ('background').
"""

from __future__ import annotations

import copy
import uuid
from dataclasses import dataclass
from typing import Any, Callable

from composition import (
    Composition,
    CompositionError,
    GetImage,
    Grid,
    Layer,
    Placement,
    ValidationReport,
)
from palette import Palette, resolve_palette
from svg_image import SVGImage, normalize_color


class SceneValidationError(Exception):
    """Ошибка валидации сцены с подробным пояснением для агента или пользователя."""


@dataclass
class HistoryEntry:
    step_id: int
    message: str
    composition: Composition
    active_palette: Palette | None = None


class SceneSession:
    """Управляет состоянием текущей композиции, историей изменений и пулом кандидатов."""

    def __init__(
        self,
        get_image: GetImage,
        find_variant: Callable[[str, str], SVGImage | None] | None = None,
        find_role_variant: Callable[[str, str], SVGImage | None] | None = None,
        composition: Composition | None = None,
    ) -> None:
        self.get_image = get_image
        self.find_variant = find_variant
        self.find_role_variant = find_role_variant
        self.composition: Composition = composition or Composition(width=800.0, height=800.0)
        self.active_palette: Palette | None = None
        self.candidates: list[str] = []
        self.last_warnings: list[str] = []

        self._history: list[HistoryEntry] = [
            HistoryEntry(
                step_id=0,
                message="Начальное состояние сцены",
                composition=self.composition.clone(),
                active_palette=copy.deepcopy(self.active_palette),
            )
        ]
        self._cursor: int = 0

    @property
    def current_step_id(self) -> int:
        return self._history[self._cursor].step_id

    def update_candidates(self, image_ids: list[str]) -> None:
        seen = set(self.candidates)
        for iid in image_ids:
            if iid not in seen:
                seen.add(iid)
                self.candidates.append(iid)

    def clear_candidates(self) -> None:
        self.candidates.clear()

    def _commit(self, comp: Composition, palette: Palette | None, message: str) -> HistoryEntry:
        self._history = self._history[: self._cursor + 1]
        next_id = self._history[-1].step_id + 1
        entry = HistoryEntry(
            step_id=next_id,
            message=message,
            composition=comp.clone(),
            active_palette=copy.deepcopy(palette),
        )
        self._history.append(entry)
        self._cursor = len(self._history) - 1
        self.composition = comp
        self.active_palette = palette
        return entry

    # ---------- Валидация цветов и сцены ----------

    def _validate_color_rules(self, comp: Composition, palette: Palette | None) -> list[str]:
        if not palette:
            return []
        errors: list[str] = []

        if comp.background_color and not palette.is_color_allowed(comp.background_color, role="background"):
            errors.append(
                f"Цвет холста {comp.background_color} не разрешён для фона в схеме {palette.title!r}."
            )

        for layer in comp.layers:
            is_bg_layer = layer.name.lower() in ("background", "фон", "bg")
            for item in layer.items:
                img = self.get_image(item.image_id)
                effective_role = "background" if (is_bg_layer or img.role == "background") else "main"
                for orig_c, mapped_c in item.colors.items():
                    if not palette.is_color_allowed(mapped_c, role=effective_role):
                        if effective_role == "background":
                            errors.append(
                                f"Фоновый объект {item.id} ({img.name}) использует запрещённый цвет {mapped_c}. "
                                f"Для фонов разрешены только: {palette.bg_colors}."
                            )
                        else:
                            errors.append(
                                f"Главный объект {item.id} ({img.name}) использует цвет {mapped_c} вне палитры {palette.title!r}."
                            )
        return errors

    def format_validation_errors(
        self,
        comp: Composition,
        report: ValidationReport,
        color_errors: list[str] | None = None,
    ) -> str:
        lines: list[str] = []
        if report.scale_violations:
            lines.append("Нарушение единого относительного масштаба объектов:")
            for sv in report.scale_violations:
                lines.append(f"  - {sv}")
        if report.role_violations:
            lines.append("Нарушение ролей объектов (Главные / Фоновые):")
            for rv in report.role_violations:
                lines.append(f"  - {rv}")
        if color_errors:
            lines.append("Нарушение цветовой схемы:")
            for ce in color_errors:
                lines.append(f"  - {ce}")
        if report.out_of_bounds:
            lines.append("Объект слишком сильно вышел за границы холста (должно оставаться >=30% внутри кадра):")
            for item_id in report.out_of_bounds:
                _, _, it = comp.find_item(item_id)
                x1, y1, x2, y2 = it.bbox(self.get_image)
                lines.append(
                    f"  - item_id={item_id!r}: bbox=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}), "
                    f"холст {comp.width:g}x{comp.height:g}."
                )
        if report.overlaps:
            lines.append("Полное перекрытие объектов (один полностью прячет другой):")
            for ov in report.overlaps:
                lines.append(
                    f"  - Слой {ov.layer_name!r}: объекты {ov.first!r} и {ov.second!r} "
                    f"перекрываются слишком сильно (площадь {ov.area:.1f} px²)."
                )
        return "\n".join(lines)

    # ---------- Единый масштаб сцены (uniform_scale) ----------

    def _set_uniform_scale(self, comp: Composition, new_scale: float) -> None:
        if new_scale <= 0:
            raise ValueError("Масштаб должен быть положительным")
        comp.uniform_scale = new_scale
        for layer in comp.layers:
            for item in layer.items:
                img = self.get_image(item.image_id)
                cx = item.x + (img.width * item.scale) / 2
                cy = item.y + (img.height * item.scale) / 2
                item.scale = new_scale
                item.x = round(cx - (img.width * new_scale) / 2, 2)
                item.y = round(cy - (img.height * new_scale) / 2, 2)

    # ---------- Перекраска сцены по ролям ----------

    def _recolor_composition_with_palette(self, comp: Composition, palette: Palette) -> None:
        comp.background_color = palette.canvas_bg
        for layer in comp.layers:
            is_bg_layer = layer.name.lower() in ("background", "фон", "bg")
            for item in layer.items:
                img = self.get_image(item.image_id)
                effective_role = "background" if (is_bg_layer or img.role == "background") else "main"
                item.colors = palette.map_colors_for_role(img.colors, role=effective_role)

    # ---------- Смена размера иллюстраций (L <-> S) с возвратом предупреждений ----------

    def _switch_detail_level(
        self,
        comp: Composition,
        target_level: str,
        palette: Palette | None = None,
    ) -> list[str]:
        level = target_level.upper()
        if level not in ("L", "S"):
            raise ValueError("Размер иллюстраций должен быть 'L' или 'S'")
        comp.detail_level = level
        warnings: list[str] = []
        active_pal = palette or self.active_palette

        if not self.find_variant:
            warnings.append("Поиск парных вариантов L/S не настроен в сессии.")
            return warnings

        for layer in comp.layers:
            for item in layer.items:
                cur_img = self.get_image(item.image_id)
                if cur_img.detail_level == level:
                    continue
                variant = self.find_variant(item.image_id, level)
                if variant is not None:
                    old_w = cur_img.width * item.scale
                    old_h = cur_img.height * item.scale
                    cx = item.x + old_w / 2
                    cy = item.y + old_h / 2
                    item.image_id = variant.id
                    new_w = variant.width * item.scale
                    new_h = variant.height * item.scale
                    item.x = round(cx - new_w / 2, 2)
                    item.y = round(cy - new_h / 2, 2)
                    if active_pal:
                        is_bg_layer = layer.name.lower() in ("background", "фон", "bg")
                        role = "background" if (is_bg_layer or variant.role == "background") else "main"
                        item.colors = active_pal.map_colors_for_role(variant.colors, role=role)
                else:
                    warnings.append(
                        f"Для объекта {cur_img.name!r} (concept_key={cur_img.concept_key!r}) "
                        f"не найден парный вариант размера {level!r} — оставлен вариант {cur_img.detail_level!r}."
                    )
        return warnings

    # ---------- Переключение роли объекта: Главный ('main') <-> Фоновый ('background') ----------

    def _switch_item_role(
        self,
        comp: Composition,
        item_id: str,
        target_role: str | None,
        palette: Palette | None,
    ) -> None:
        cur_layer, _, item = comp.find_item(item_id)
        cur_img = self.get_image(item.image_id)
        desired_role = (
            ("background" if target_role.lower() == "background" else "main")
            if target_role
            else ("background" if cur_img.role == "main" else "main")
        )
        if cur_img.role == desired_role:
            return

        if not self.find_role_variant:
            raise SceneValidationError("Функция поиска парной роли (main/background) не настроена.")

        variant = self.find_role_variant(item.image_id, desired_role)
        if variant is None:
            raise SceneValidationError(
                f"Для объекта {cur_img.name!r} в библиотеке не найдена парная версия в роли {desired_role!r}."
            )

        cx = item.x + (cur_img.width * item.scale) / 2
        cy = item.y + (cur_img.height * item.scale) / 2
        item.image_id = variant.id
        item.x = round(cx - (variant.width * item.scale) / 2, 2)
        item.y = round(cy - (variant.height * item.scale) / 2, 2)

        dst_layer_name = "Background" if desired_role == "background" else "Main"
        if cur_layer.name != dst_layer_name and any(l.name == dst_layer_name for l in comp.layers):
            comp.move_item_to_layer(item.id, dst_layer_name)

        if palette:
            item.colors = palette.map_colors_for_role(variant.colors, role=desired_role)
        else:
            item.colors = {}

    # ---------- Создание сцены с нуля ----------

    def create_scene(
        self,
        *,
        width: float = 800.0,
        height: float = 800.0,
        grid_columns: int = 3,
        grid_rows: int = 3,
        strict_bounds: bool = False,
        detail_level: str = "L",
        uniform_scale: float = 1.0,
        show_isometric_guides: bool = True,
        layers: list[dict[str, Any]] | None = None,
        placements: list[dict[str, Any]] | None = None,
        palette: str | dict[str, Any] | Palette | None = None,
        commit_message: str = "Создание новой сцены",
        strict_validation: bool = True,
    ) -> HistoryEntry:
        self.last_warnings.clear()
        layer_objs: list[Layer] = []
        if layers:
            for lspec in layers:
                layer_objs.append(
                    Layer(
                        name=lspec["name"],
                        allow_internal_overlap=bool(lspec.get("allow_internal_overlap", True)),
                        strict_bounds=lspec.get("strict_bounds", False),
                        visible=bool(lspec.get("visible", True)),
                    )
                )
        else:
            layer_objs = [
                Layer(name="Background", allow_internal_overlap=True, strict_bounds=False),
                Layer(name="Main", allow_internal_overlap=True, strict_bounds=False),
            ]

        comp = Composition(
            width=width,
            height=height,
            grid=Grid(columns=grid_columns, rows=grid_rows),
            layers=layer_objs,
            strict_bounds=strict_bounds,
            detail_level=detail_level.upper(),
            uniform_scale=float(uniform_scale),
            show_isometric_guides=show_isometric_guides,
        )

        resolved_palette = resolve_palette(palette) if palette else None
        if resolved_palette:
            comp.background_color = resolved_palette.canvas_bg

        for spec in placements or []:
            self._apply_add_item(comp, spec, palette=resolved_palette)

        if resolved_palette:
            self._recolor_composition_with_palette(comp, resolved_palette)

        report = comp.validate(self.get_image)
        color_errors = self._validate_color_rules(comp, resolved_palette)
        if strict_validation and (not report.ok or color_errors):
            raise SceneValidationError(self.format_validation_errors(comp, report, color_errors))

        return self._commit(comp, resolved_palette, commit_message)

    # ---------- Применение транзакционного диффа ----------

    def _apply_add_item(
        self,
        comp: Composition,
        params: dict[str, Any],
        palette: Palette | None = None,
    ) -> Placement:
        image_id = params["image_id"]
        if self.find_variant and comp.detail_level in ("L", "S"):
            cur = self.get_image(image_id)
            if cur.detail_level != comp.detail_level:
                variant = self.find_variant(image_id, comp.detail_level)
                if variant is not None:
                    image_id = variant.id
                else:
                    self.last_warnings.append(
                        f"Объект {cur.name!r} добавлен в размере {cur.detail_level!r}, так как пара {comp.detail_level!r} не найдена."
                    )

        img = self.get_image(image_id)
        target_layer_name = params.get("layer")
        if not target_layer_name:
            if img.role == "background":
                target_layer_name = "Background"
            else:
                target_layer_name = "Main" if any(l.name == "Main" for l in comp.layers) else comp.layers[-1].name

        scale = comp.uniform_scale
        rotation = float(params.get("rotation", 0.0))
        colors = dict(params.get("colors") or {})

        if not colors and palette:
            role = "background" if (target_layer_name.lower() == "background" or img.role == "background") else "main"
            colors = palette.map_colors_for_role(img.colors, role=role)

        if params.get("cell") is not None:
            col, row = params["cell"]
            return comp.place_in_cell(
                image_id=image_id,
                column=float(col),
                row=float(row),
                get_image=self.get_image,
                scale=scale,
                rotation=rotation,
                colors=colors,
                layer=target_layer_name,
            )

        target_layer = comp.get_layer(target_layer_name)
        pl = Placement(
            image_id=image_id,
            x=float(params.get("x", 0.0)),
            y=float(params.get("y", 0.0)),
            scale=scale,
            rotation=rotation,
            colors=colors,
        )
        target_layer.items.append(pl)
        return pl

    def apply_diff(
        self,
        operations: list[dict[str, Any]],
        *,
        commit_message: str = "Обновление сцены",
        strict_validation: bool = True,
    ) -> HistoryEntry:
        if not operations:
            raise ValueError("Список операций пуст")

        self.last_warnings.clear()
        draft = self.composition.clone()
        draft_palette = copy.deepcopy(self.active_palette)

        for raw_op in operations:
            op = raw_op.get("op")
            params = raw_op.get("params") or {}

            if op == "add_item":
                self._apply_add_item(draft, params, palette=draft_palette)

            elif op == "update_item":
                layer, _, item = draft.find_item(params["item_id"])
                image_changed = False
                if "image_id" in params and params["image_id"] != item.image_id:
                    item.image_id = params["image_id"]
                    image_changed = True
                if "x" in params:
                    item.x = float(params["x"])
                if "y" in params:
                    item.y = float(params["y"])
                if "scale" in params:
                    self._set_uniform_scale(draft, float(params["scale"]))
                if "rotation" in params:
                    item.rotation = float(params["rotation"])

                img = self.get_image(item.image_id)
                is_bg = layer.name.lower() in ("background", "фон", "bg") or img.role == "background"
                eff_role = "background" if is_bg else "main"

                if "colors" in params:
                    new_colors = dict(params["colors"])
                    if draft_palette:
                        for c_val in new_colors.values():
                            if not draft_palette.is_color_allowed(c_val, role=eff_role):
                                raise SceneValidationError(
                                    f"Цвет {c_val} запрещён для роли {eff_role!r} в схеме {draft_palette.title!r}!"
                                )
                    item.colors = new_colors
                elif image_changed:
                    # Автоматически пересчитываем цвета под новую иллюстрацию при смене image_id
                    if draft_palette:
                        item.colors = draft_palette.map_colors_for_role(img.colors, role=eff_role)
                    else:
                        item.colors = {}

            elif op == "set_uniform_scale":
                self._set_uniform_scale(draft, float(params["scale"]))

            elif op == "toggle_item_role":
                self._switch_item_role(
                    draft,
                    params["item_id"],
                    params.get("role"),
                    draft_palette,
                )

            elif op == "move_item_cell":
                draft.move_item_to_cell(
                    params["item_id"],
                    float(params["column"]),
                    float(params["row"]),
                    self.get_image,
                )

            elif op == "move_item_delta":
                _, _, item = draft.find_item(params["item_id"])
                item.x += float(params.get("dx", 0.0))
                item.y += float(params.get("dy", 0.0))

            elif op == "remove_item":
                draft.remove_item(params["item_id"])

            elif op == "duplicate_item":
                layer, _, item = draft.find_item(params["item_id"])
                clone_it = copy.deepcopy(item)
                clone_it.id = uuid.uuid4().hex
                clone_it.x += float(params.get("dx", 40.0))
                clone_it.y += float(params.get("dy", 40.0))
                layer.items.append(clone_it)

            elif op == "reorder_item":
                draft.reorder_item(params["item_id"], int(params["index"]))

            elif op == "move_item_to_layer":
                draft.move_item_to_layer(
                    params["item_id"],
                    params["layer"],
                    params.get("index"),
                )

            elif op == "add_layer":
                draft.add_layer(
                    name=params["name"],
                    allow_internal_overlap=bool(params.get("allow_internal_overlap", True)),
                    strict_bounds=params.get("strict_bounds", False),
                    visible=bool(params.get("visible", True)),
                    index=params.get("index"),
                )

            elif op == "remove_layer":
                draft.remove_layer(params["layer"])

            elif op == "update_layer":
                layer = draft.get_layer(params["layer"])
                if "name" in params:
                    layer.name = str(params["name"])
                if "allow_internal_overlap" in params:
                    layer.allow_internal_overlap = bool(params["allow_internal_overlap"])
                if "strict_bounds" in params:
                    layer.strict_bounds = params["strict_bounds"]
                if "visible" in params:
                    layer.visible = bool(params["visible"])

            elif op == "reorder_layer":
                draft.reorder_layer(params["layer"], int(params["index"]))

            elif op == "set_canvas":
                if "width" in params:
                    draft.width = float(params["width"])
                if "height" in params:
                    draft.height = float(params["height"])
                if "strict_bounds" in params:
                    draft.strict_bounds = bool(params["strict_bounds"])
                if "show_isometric_guides" in params:
                    draft.show_isometric_guides = bool(params["show_isometric_guides"])
                if "background_color" in params:
                    bg_c = params["background_color"]
                    if bg_c and draft_palette and not draft_palette.is_color_allowed(bg_c, role="background"):
                        raise SceneValidationError(
                            f"Цвет {bg_c} запрещён для фона в схеме {draft_palette.title!r}."
                        )
                    draft.background_color = normalize_color(bg_c) if bg_c else None

            elif op == "set_grid":
                draft.grid = Grid(
                    columns=int(params.get("columns", draft.grid.columns)),
                    rows=int(params.get("rows", draft.grid.rows)),
                )

            elif op == "set_detail_level":
                warns = self._switch_detail_level(draft, str(params["detail_level"]), palette=draft_palette)
                self.last_warnings.extend(warns)

            elif op == "apply_palette":
                spec = params.get("palette") or params.get("palette_name")
                draft_palette = resolve_palette(spec)
                if draft_palette:
                    self._recolor_composition_with_palette(draft, draft_palette)

            else:
                raise ValueError(f"Неизвестная операция диффа: {op!r}")

        report = draft.validate(self.get_image)
        color_errors = self._validate_color_rules(draft, draft_palette)
        if strict_validation and (not report.ok or color_errors):
            raise SceneValidationError(self.format_validation_errors(draft, report, color_errors))

        return self._commit(draft, draft_palette, commit_message)

    # ---------- История ----------

    def undo(self) -> HistoryEntry:
        if self._cursor <= 0:
            raise CompositionError("Нечего отменять: вы на самом первом шаге")
        self._cursor -= 1
        entry = self._history[self._cursor]
        self.composition = entry.composition.clone()
        self.active_palette = copy.deepcopy(entry.active_palette)
        return entry

    def redo(self) -> HistoryEntry:
        if self._cursor >= len(self._history) - 1:
            raise CompositionError("Нечего повторять: вы на последнем шаге")
        self._cursor += 1
        entry = self._history[self._cursor]
        self.composition = entry.composition.clone()
        self.active_palette = copy.deepcopy(entry.active_palette)
        return entry

    def revert_to_step(self, step_id: int) -> HistoryEntry:
        for idx, entry in enumerate(self._history):
            if entry.step_id == step_id:
                self._cursor = idx
                self.composition = entry.composition.clone()
                self.active_palette = copy.deepcopy(entry.active_palette)
                return entry
        raise CompositionError(f"Шаг истории #{step_id} не найден")

    def get_history(self) -> list[dict[str, Any]]:
        return [
            {
                "step_id": entry.step_id,
                "message": entry.message,
                "is_current": idx == self._cursor,
                "layers_count": len(entry.composition.layers),
                "items_count": len(entry.composition.items),
            }
            for idx, entry in enumerate(self._history)
        ]

    # ---------- Сводка для LLM-агента ----------

    def get_scene_summary(self) -> dict[str, Any]:
        cw, ch = self.composition.grid.cell_size(self.composition.width, self.composition.height)
        layers_info: list[dict[str, Any]] = []
        for l_idx, layer in enumerate(self.composition.layers):
            items_info: list[dict[str, Any]] = []
            for it in layer.items:
                img = self.get_image(it.image_id)
                x1, y1, x2, y2 = it.bbox(self.get_image)
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                approx_cell = self.composition.grid.cell_at(cx, cy, self.composition.width, self.composition.height)
                items_info.append(
                    {
                        "item_id": it.id,
                        "image_id": it.image_id,
                        "image_name": img.name,
                        "concept_key": img.concept_key,
                        "role": img.role,
                        "detail_level": img.detail_level,
                        "x": round(it.x, 1),
                        "y": round(it.y, 1),
                        "width": round(img.width * it.scale, 1),
                        "height": round(img.height * it.scale, 1),
                        "scale": it.scale,
                        "rotation": it.rotation,
                        "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                        "approx_cell": list(approx_cell),
                    }
                )
            layers_info.append(
                {
                    "layer_id": layer.id,
                    "name": layer.name,
                    "index": l_idx,
                    "strict_bounds": layer.strict_bounds,
                    "allow_internal_overlap": layer.allow_internal_overlap,
                    "visible": layer.visible,
                    "items": items_info,
                }
            )

        return {
            "step_id": self.current_step_id,
            "detail_level": self.composition.detail_level,
            "uniform_scale": self.composition.uniform_scale,
            "warnings": list(self.last_warnings),
            "canvas": {
                "width": self.composition.width,
                "height": self.composition.height,
                "background_color": self.composition.background_color,
                "strict_bounds": self.composition.strict_bounds,
                "show_isometric_guides": self.composition.show_isometric_guides,
            },
            "grid": {
                "columns": self.composition.grid.columns,
                "rows": self.composition.grid.rows,
                "cell_width": round(cw, 1),
                "cell_height": round(ch, 1),
            },
            "active_palette": self.active_palette.to_dict() if self.active_palette else None,
            "layers": layers_info,
        }