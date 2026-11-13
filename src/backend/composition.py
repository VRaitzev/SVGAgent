"""Модель композиции: холст, дробная сетка, смысловые слои (Background / Main),
единый относительный масштаб (uniform_scale) и мягкая валидация изометрических композиций.
"""

from __future__ import annotations

import copy
import math
import uuid
from dataclasses import dataclass, field
from typing import Callable

from svg_image import SVGImage

GetImage = Callable[[str], SVGImage]


class CompositionError(Exception):
    """Базовая ошибка композиции."""


class ItemNotFoundError(CompositionError):
    """Объект с таким id не найден."""


class LayerNotFoundError(CompositionError):
    """Слой с таким id или именем не найден."""


@dataclass
class Grid:
    columns: int = 4
    rows: int = 3

    def __post_init__(self) -> None:
        if self.columns < 1 or self.rows < 1:
            raise ValueError("Сетка должна иметь минимум 1 столбец и 1 строку")

    def cell_size(self, canvas_width: float, canvas_height: float) -> tuple[float, float]:
        return canvas_width / self.columns, canvas_height / self.rows

    def cell_rect(
        self,
        column: float,
        row: float,
        canvas_width: float,
        canvas_height: float,
    ) -> tuple[float, float, float, float]:
        if not (-0.5 <= column < self.columns and -0.5 <= row < self.rows):
            raise IndexError(f"Клетка ({column}, {row}) вне сетки {self.columns}x{self.rows}")
        cw, ch = self.cell_size(canvas_width, canvas_height)
        return column * cw, row * ch, (column + 1) * cw, (row + 1) * ch

    def cell_center(
        self,
        column: float,
        row: float,
        canvas_width: float,
        canvas_height: float,
    ) -> tuple[float, float]:
        """Поддерживает как целые (1, 1), так и дробные координаты сетки (1.5, 1.33)."""
        x1, y1, x2, y2 = self.cell_rect(column, row, canvas_width, canvas_height)
        return (x1 + x2) / 2, (y1 + y2) / 2

    def cell_at(
        self,
        x: float,
        y: float,
        canvas_width: float,
        canvas_height: float,
    ) -> tuple[float, float]:
        cw, ch = self.cell_size(canvas_width, canvas_height)
        col = round(min(max(x / cw - 0.5, 0.0), self.columns - 1.0), 2)
        row = round(min(max(y / ch - 0.5, 0.0), self.rows - 1.0), 2)
        return col, row


@dataclass
class Placement:
    image_id: str
    x: float = 0.0
    y: float = 0.0
    scale: float = 1.0
    rotation: float = 0.0
    colors: dict[str, str] = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def __post_init__(self) -> None:
        if self.scale <= 0:
            raise ValueError("scale должен быть положительным")

    def bbox(self, get_image: GetImage) -> tuple[float, float, float, float]:
        img = get_image(self.image_id)
        w = img.width * self.scale
        h = img.height * self.scale
        angle = math.radians(self.rotation)
        cos_a = abs(math.cos(angle))
        sin_a = abs(math.sin(angle))
        rw = w * cos_a + h * sin_a
        rh = w * sin_a + h * cos_a
        cx = self.x + w / 2
        cy = self.y + h / 2
        return cx - rw / 2, cy - rh / 2, cx + rw / 2, cy + rh / 2


def _rects_intersect(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
    eps: float = 1e-3,
) -> tuple[bool, float]:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    w = ix2 - ix1
    h = iy2 - iy1
    if w > eps and h > eps:
        return True, w * h
    return False, 0.0


def _rect_area(r: tuple[float, float, float, float]) -> float:
    return max(0.0, r[2] - r[0]) * max(0.0, r[3] - r[1])


@dataclass
class Layer:
    name: str
    items: list[Placement] = field(default_factory=list)
    allow_internal_overlap: bool = True
    strict_bounds: bool | None = False
    visible: bool = True
    id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("Имя слоя не может быть пустым")


@dataclass
class OverlapIssue:
    layer_id: str
    layer_name: str
    first: str
    second: str
    area: float


@dataclass
class ValidationReport:
    out_of_bounds: list[str] = field(default_factory=list)
    overlaps: list[OverlapIssue] = field(default_factory=list)
    role_violations: list[str] = field(default_factory=list)
    scale_violations: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return (
            not self.out_of_bounds
            and not self.overlaps
            and not self.role_violations
            and not self.scale_violations
        )


@dataclass
class Composition:
    width: float
    height: float
    grid: Grid = field(default_factory=lambda: Grid(columns=4, rows=3))
    layers: list[Layer] = field(default_factory=list)
    strict_bounds: bool = False
    detail_level: str = "L"
    uniform_scale: float = 1.0
    show_isometric_guides: bool = True
    background_color: str | None = None
    name: str | None = None
    id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Размеры холста должны быть положительными")
        if self.uniform_scale <= 0:
            raise ValueError("uniform_scale должен быть положительным")
        if not self.layers:
            self.layers = [
                Layer(name="Background", allow_internal_overlap=True, strict_bounds=False),
                Layer(name="Main", allow_internal_overlap=True, strict_bounds=False),
            ]

    @property
    def items(self) -> list[Placement]:
        return [it for layer in self.layers for it in layer.items]

    def get_layer(self, id_or_name: str) -> Layer:
        for layer in self.layers:
            if layer.id == id_or_name or layer.name == id_or_name:
                return layer
        raise LayerNotFoundError(f"Слой не найден: {id_or_name!r}")

    def add_layer(
        self,
        name: str,
        *,
        allow_internal_overlap: bool = True,
        strict_bounds: bool | None = False,
        visible: bool = True,
        index: int | None = None,
    ) -> Layer:
        if any(l.name == name for l in self.layers):
            raise ValueError(f"Слой с именем {name!r} уже существует")
        layer = Layer(
            name=name,
            allow_internal_overlap=allow_internal_overlap,
            strict_bounds=strict_bounds,
            visible=visible,
        )
        if index is None or index >= len(self.layers):
            self.layers.append(layer)
        else:
            self.layers.insert(max(0, index), layer)
        return layer

    def remove_layer(self, id_or_name: str) -> None:
        if len(self.layers) <= 1:
            raise CompositionError("Нельзя удалить последний оставшийся слой")
        layer = self.get_layer(id_or_name)
        self.layers.remove(layer)

    def reorder_layer(self, id_or_name: str, new_index: int) -> None:
        layer = self.get_layer(id_or_name)
        self.layers.remove(layer)
        self.layers.insert(min(max(0, new_index), len(self.layers)), layer)

    def find_item(self, item_id: str) -> tuple[Layer, int, Placement]:
        for layer in self.layers:
            for idx, it in enumerate(layer.items):
                if it.id == item_id:
                    return layer, idx, it
        raise ItemNotFoundError(f"Объект с id={item_id!r} не найден")

    def place_in_cell(
        self,
        image_id: str,
        column: float,
        row: float,
        get_image: GetImage,
        *,
        scale: float | None = None,
        rotation: float = 0.0,
        colors: dict[str, str] | None = None,
        layer: str | None = None,
    ) -> Placement:
        img = get_image(image_id)
        eff_scale = self.uniform_scale if scale is None else scale
        cx, cy = self.grid.cell_center(column, row, self.width, self.height)
        x = cx - (img.width * eff_scale) / 2
        y = cy - (img.height * eff_scale) / 2
        target_layer = self.get_layer(layer) if layer else self.layers[-1]
        pl = Placement(
            image_id=image_id,
            x=round(x, 2),
            y=round(y, 2),
            scale=eff_scale,
            rotation=rotation,
            colors=dict(colors or {}),
        )
        target_layer.items.append(pl)
        return pl

    def move_item_to_cell(
        self,
        item_id: str,
        column: float,
        row: float,
        get_image: GetImage,
    ) -> Placement:
        _, _, item = self.find_item(item_id)
        img = get_image(item.image_id)
        cx, cy = self.grid.cell_center(column, row, self.width, self.height)
        item.x = round(cx - (img.width * item.scale) / 2, 2)
        item.y = round(cy - (img.height * item.scale) / 2, 2)
        return item

    def move_item_to_layer(
        self,
        item_id: str,
        target_layer: str,
        index: int | None = None,
    ) -> Placement:
        src_layer, _, item = self.find_item(item_id)
        dst_layer = self.get_layer(target_layer)
        src_layer.items.remove(item)
        if index is None or index >= len(dst_layer.items):
            dst_layer.items.append(item)
        else:
            dst_layer.items.insert(max(0, index), item)
        return item

    def reorder_item(self, item_id: str, new_index: int) -> Placement:
        layer, _, item = self.find_item(item_id)
        layer.items.remove(item)
        layer.items.insert(min(max(0, new_index), len(layer.items)), item)
        return item

    def remove_item(self, item_id: str) -> None:
        layer, _, item = self.find_item(item_id)
        layer.items.remove(item)

    def clone(self) -> Composition:
        return copy.deepcopy(self)

    def validate(self, get_image: GetImage) -> ValidationReport:
        report = ValidationReport()
        eps = 1e-2
        canvas_box = (0.0, 0.0, self.width, self.height)

        for layer in self.layers:
            if not layer.visible:
                continue
            effective_strict = self.strict_bounds if layer.strict_bounds is None else layer.strict_bounds
            is_bg_layer = layer.name.lower() in ("background", "фон", "bg")
            bboxes: list[tuple[Placement, tuple[float, float, float, float]]] = []

            for item in layer.items:
                # 1. Правило ТЗ: нельзя менять пропорции объектов относительно друг друга
                if abs(item.scale - self.uniform_scale) > 1e-3:
                    report.scale_violations.append(
                        f"Объект {item.id} имеет масштаб {item.scale}, отличный от единого масштаба сцены ({self.uniform_scale})."
                    )

                img = get_image(item.image_id)
                # 2. Проверка ролей: главные объекты — в Main, фоновые — в Background
                if is_bg_layer and img.role == "main":
                    report.role_violations.append(
                        f"Главный объект {item.id} ({img.name}) нельзя размещать в фоновом слое {layer.name!r}."
                    )
                if not is_bg_layer and img.role == "background":
                    report.role_violations.append(
                        f"Фоновый монохромный объект {item.id} ({img.name}) должен находиться в слое 'Background', а не {layer.name!r}."
                    )

                # 3. Проверка границ холста:
                # разрешаем композиционную обрезку краями холста (Deliberate Crop / Bleed),
                # но минимум 30% площади объекта должно оставаться внутри кадра.
                box = item.bbox(get_image)
                x1, y1, x2, y2 = box
                if effective_strict:
                    if x1 < -eps or y1 < -eps or x2 > self.width + eps or y2 > self.height + eps:
                        report.out_of_bounds.append(item.id)
                else:
                    inter, inter_area = _rects_intersect(box, canvas_box)
                    total_area = _rect_area(box)
                    if not inter or (total_area > 0 and (inter_area / total_area) < 0.30):
                        report.out_of_bounds.append(item.id)

                bboxes.append((item, box))

            # 4. Проверка перекрытий:
            # Свободно разрешаем наслоение мелкого объекта на крупный (ручка, телефон, пазл, монета
            # могут полностью лежать поверх книги или папки!).
            # Блокируем только случай, когда два сопоставимых по размеру объекта (>70% площади)
            # поставлены в одну и ту же точку с почти совпадающими центрами.
            for i in range(len(bboxes)):
                for j in range(i + 1, len(bboxes)):
                    it_a, box_a = bboxes[i]
                    it_b, box_b = bboxes[j]
                    hit, area = _rects_intersect(box_a, box_b)
                    if not hit:
                        continue
                    if not layer.allow_internal_overlap:
                        report.overlaps.append(
                            OverlapIssue(
                                layer_id=layer.id,
                                layer_name=layer.name,
                                first=it_a.id,
                                second=it_b.id,
                                area=area,
                            )
                        )
                    else:
                        area_a = _rect_area(box_a)
                        area_b = _rect_area(box_b)
                        max_area = max(area_a, area_b)
                        min_area = min(area_a, area_b)
                        size_ratio = (min_area / max_area) if max_area > 0 else 1.0
                        cax, cay = (box_a[0] + box_a[2]) / 2, (box_a[1] + box_a[3]) / 2
                        cbx, cby = (box_b[0] + box_b[2]) / 2, (box_b[1] + box_b[3]) / 2
                        center_dist = math.hypot(cax - cbx, cay - cby)
                        if size_ratio > 0.75 and (area / max_area) > 0.85 and center_dist < 25.0:
                            report.overlaps.append(
                                OverlapIssue(
                                    layer_id=layer.id,
                                    layer_name=layer.name,
                                    first=it_a.id,
                                    second=it_b.id,
                                    area=area,
                                )
                            )
        return report