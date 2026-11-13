
"""Композиция: холст, сетка, смысловые слои (Layer) и размещённые на них иллюстрации (Placement).

Композиция хранит только ссылки на иллюстрации (image_id) и параметры их
размещения внутри слоёв. Сами SVG не копируются и не меняются: перекраска хранится как
подмена цветов и применяется только при сборке итогового SVG.

Соглашения о координатах (как в Figma и SVG):
- начало координат в левом верхнем углу холста, ось Y направлена вниз;
- (x, y) у объекта — положение левого верхнего угла его рамки до поворота;
- размер рамки = размер иллюстрации * scale;
- rotation — градусы по часовой стрелке вокруг центра рамки;
- порядок слоёв в списке layers: от нижнего (фон) к верхнему (передний план);
- порядок объектов внутри layer.items: последний рисуется поверх остальных в рамках своего слоя;
- пересечения объектов запрещены ВНУТРИ одного слоя (если allow_internal_overlap=False),
  но разрешены МЕЖДУ разными слоями.
"""

from __future__ import annotations

import json
import math
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field

from ar.svg_image import SVGImage, normalize_color

GetImage = Callable[[str], SVGImage]
Rect = tuple[float, float, float, float]  # x_min, y_min, x_max, y_max

EPS = 1e-6


class CompositionError(Exception):
    """Базовая ошибка композиции."""


class ItemNotFoundError(CompositionError):
    """Объекта с таким id нет в композиции."""


class LayerNotFoundError(CompositionError):
    """Слоя с таким id или именем нет в композиции."""


@dataclass
class Grid:
    """Параметры сетки. Клетки не хранятся, их размер считается от холста."""

    columns: int = 1
    rows: int = 1

    def __post_init__(self) -> None:
        if self.columns < 1 or self.rows < 1:
            raise CompositionError("В сетке должно быть минимум 1 столбец и 1 строка")


@dataclass
class Placement:
    """Одна иллюстрация, размещённая в слое холста."""

    image_id: str
    x: float = 0.0
    y: float = 0.0
    scale: float = 1.0
    rotation: float = 0.0
    colors: dict[str, str] = field(default_factory=dict)  # было -> стало
    id: str = field(default_factory=lambda: uuid.uuid4().hex)


@dataclass
class Layer:
    """Смысловой слой композиции, хранящий внутри себя список объектов (Placement).

    - allow_internal_overlap: если False (по умолчанию), объекты внутри этого слоя
      не могут пересекаться друг с другом. Объекты из РАЗНЫХ слоёв пересекаться могут.
    - strict_bounds: если True (по умолчанию — жёсткая рамка), объекты этого слоя
      обязаны целиком помещаться внутри холста. Если False (гибкая рамка), объект может
      частично выходить за края холста (пока его центр остаётся внутри холста).
    """

    name: str = "Layer"
    items: list[Placement] = field(default_factory=list)
    allow_internal_overlap: bool = False
    strict_bounds: bool = True
    visible: bool = True
    id: str = field(default_factory=lambda: uuid.uuid4().hex)


@dataclass(frozen=True)
class Overlap:
    layer_id: str  # id слоя, внутри которого произошло пересечение
    first: str  # id первого объекта
    second: str  # id второго объекта
    area: float  # площадь пересечения рамок


@dataclass
class ValidationReport:
    overlaps: list[Overlap] = field(default_factory=list)
    out_of_bounds: list[str] = field(default_factory=list)  # id объектов, нарушивших границы холста

    @property
    def ok(self) -> bool:
        return not self.overlaps and not self.out_of_bounds


def _normalize_colors(colors: dict[str, str] | None) -> dict[str, str]:
    result: dict[str, str] = {}
    for old, new in (colors or {}).items():
        old_n, new_n = normalize_color(old), normalize_color(new)
        if old_n is None or new_n is None:
            raise CompositionError(f"Некорректная подмена цвета: {old!r} -> {new!r}")
        result[old_n] = new_n
    return result


def _check_scale(scale: float) -> None:
    if not scale > 0:
        raise CompositionError("scale должен быть больше 0")


@dataclass
class Composition:
    """Холст width x height с сеткой, параметризуемой рамкой и списком слоёв (Layer)."""

    width: float = 1200.0
    height: float = 800.0
    grid: Grid = field(default_factory=Grid)
    strict_bounds: bool = True  # глобальный режим рамки холста по умолчанию для новых слоёв
    name: str | None = None
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    layers: list[Layer] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not (self.width > 0 and self.height > 0):
            raise CompositionError("Размеры холста должны быть больше 0")
        if not self.layers:
            # По умолчанию создаём базовый слой, чтобы холст сразу был готов к добавлению объектов
            self.layers.append(Layer(name="Main", strict_bounds=self.strict_bounds))

    # ---------- сквозной доступ к объектам (обратная совместимость) ----------

    @property
    def items(self) -> list[Placement]:
        """Плоский список всех объектов со всех слоёв снизу вверх (в порядке отрисовки)."""
        result: list[Placement] = []
        for layer in self.layers:
            result.extend(layer.items)
        return result

    def get(self, item_id: str) -> Placement:
        """Находит объект по его id в любом слое."""
        for layer in self.layers:
            for item in layer.items:
                if item.id == item_id:
                    return item
        raise ItemNotFoundError(f"Нет объекта с id={item_id}")

    def get_item_layer(self, item_id: str) -> Layer:
        """Возвращает слой, в котором лежит объект с заданным item_id."""
        for layer in self.layers:
            for item in layer.items:
                if item.id == item_id:
                    return layer
        raise ItemNotFoundError(f"Нет объекта с id={item_id}")

    # ---------- управление слоями (Layer) ----------

    def get_layer(self, layer_id_or_name: str) -> Layer:
        """Ищет слой сначала по точному id, затем по имени (без учёта регистра)."""
        for layer in self.layers:
            if layer.id == layer_id_or_name:
                return layer
        target_name = layer_id_or_name.strip().lower()
        for layer in self.layers:
            if layer.name.strip().lower() == target_name:
                return layer
        raise LayerNotFoundError(f"Нет слоя с id или именем {layer_id_or_name!r}")

    def add_layer(
        self,
        name: str,
        *,
        allow_internal_overlap: bool = False,
        strict_bounds: bool | None = None,
        index: int | None = None,
    ) -> Layer:
        """Создаёт новый слой. По умолчанию добавляет его поверх остальных."""
        layer = Layer(
            name=name,
            allow_internal_overlap=allow_internal_overlap,
            strict_bounds=self.strict_bounds if strict_bounds is None else strict_bounds,
        )
        if index is None:
            self.layers.append(layer)
        else:
            self.layers.insert(max(0, min(index, len(self.layers))), layer)
        return layer

    def remove_layer(self, layer_id_or_name: str) -> None:
        """Удаляет слой вместе с его объектами. Нельзя удалить последний оставшийся слой."""
        if len(self.layers) <= 1:
            raise CompositionError("В композиции должен оставаться хотя бы один слой")
        layer = self.get_layer(layer_id_or_name)
        self.layers.remove(layer)

    def layer_order_index(self, layer_id_or_name: str) -> int:
        """Индекс самого слоя в списке слоёв (0 — самый нижний/фоновый слой)."""
        layer = self.get_layer(layer_id_or_name)
        return self.layers.index(layer)

    def reorder_layer(self, layer_id_or_name: str, new_index: int) -> Layer:
        """Перемещает целый слой на новую позицию по глубине."""
        layer = self.get_layer(layer_id_or_name)
        self.layers.remove(layer)
        self.layers.insert(max(0, min(new_index, len(self.layers))), layer)
        return layer

    # ---------- управление объектами (Placement) ----------

    def add(
        self,
        image_id: str,
        x: float = 0.0,
        y: float = 0.0,
        *,
        scale: float = 1.0,
        rotation: float = 0.0,
        colors: dict[str, str] | None = None,
        layer: str | None = None,
    ) -> Placement:
        """Добавляет иллюстрацию в указанный слой (по id или имени) или в самый верхний слой."""
        if not image_id:
            raise CompositionError("image_id не может быть пустым")
        _check_scale(scale)

        if layer is not None:
            target_layer = self.get_layer(layer)
        else:
            if not self.layers:
                self.layers.append(Layer(name="Main", strict_bounds=self.strict_bounds))
            target_layer = self.layers[-1]

        item = Placement(
            image_id=image_id,
            x=x,
            y=y,
            scale=scale,
            rotation=rotation,
            colors=_normalize_colors(colors),
        )
        target_layer.items.append(item)
        return item

    def update(
        self,
        item_id: str,
        *,
        x: float | None = None,
        y: float | None = None,
        scale: float | None = None,
        rotation: float | None = None,
        colors: dict[str, str] | None = None,
        layer: str | None = None,
    ) -> Placement:
        """Меняет переданные параметры объекта (включая перенос в другой слой при необходимости)."""
        item = self.get(item_id)
        if scale is not None:
            _check_scale(scale)
            item.scale = scale
        if x is not None:
            item.x = x
        if y is not None:
            item.y = y
        if rotation is not None:
            item.rotation = rotation
        if colors is not None:
            item.colors = _normalize_colors(colors)
        if layer is not None:
            self.move_item_to_layer(item_id, layer)
        return item

    def remove(self, item_id: str) -> None:
        layer = self.get_item_layer(item_id)
        item = self.get(item_id)
        layer.items.remove(item)

    def duplicate(self, item_id: str, dx: float = 20.0, dy: float = 20.0) -> Placement:
        """Копия объекта (новый id) со смещением; кладётся в тот же слой прямо над оригиналом."""
        layer = self.get_item_layer(item_id)
        src = self.get(item_id)
        copy_item = Placement(
            image_id=src.image_id,
            x=src.x + dx,
            y=src.y + dy,
            scale=src.scale,
            rotation=src.rotation,
            colors=dict(src.colors),
        )
        idx = layer.items.index(src)
        layer.items.insert(idx + 1, copy_item)
        return copy_item

    # ---------- перемещение объектов между слоями и внутри слоя ----------

    def move_item_to_layer(
        self,
        item_id: str,
        target_layer_id_or_name: str,
        index: int | None = None,
    ) -> Placement:
        """Перекладывает объект из его текущего слоя в другой слой."""
        src_layer = self.get_item_layer(item_id)
        dst_layer = self.get_layer(target_layer_id_or_name)
        item = self.get(item_id)
        src_layer.items.remove(item)
        if index is None:
            dst_layer.items.append(item)
        else:
            dst_layer.items.insert(max(0, min(index, len(dst_layer.items))), item)
        return item

    def layer_index(self, item_id: str) -> int:
        """Индекс объекта внутри его текущего слоя (0 — самый нижний в этом слое)."""
        layer = self.get_item_layer(item_id)
        item = self.get(item_id)
        return layer.items.index(item)

    def move_to_layer(self, item_id: str, index: int) -> None:
        """Меняет порядок объекта внутри его текущего слоя."""
        layer = self.get_item_layer(item_id)
        item = self.get(item_id)
        layer.items.remove(item)
        layer.items.insert(max(0, min(index, len(layer.items))), item)

    def bring_to_front(self, item_id: str) -> None:
        """Поднимает объект на самый верх его текущего слоя."""
        layer = self.get_item_layer(item_id)
        self.move_to_layer(item_id, len(layer.items))

    def send_to_back(self, item_id: str) -> None:
        """Опускает объект в самый низ его текущего слоя."""
        self.move_to_layer(item_id, 0)

    def move_up(self, item_id: str) -> None:
        self.move_to_layer(item_id, self.layer_index(item_id) + 1)

    def move_down(self, item_id: str) -> None:
        self.move_to_layer(item_id, self.layer_index(item_id) - 1)

    # ---------- сетка ----------

    def cell_size(self) -> tuple[float, float]:
        return self.width / self.grid.columns, self.height / self.grid.rows

    def cell_rect(self, column: int, row: int) -> Rect:
        """Границы клетки (индексы с нуля): x_min, y_min, x_max, y_max."""
        if not (0 <= column < self.grid.columns and 0 <= row < self.grid.rows):
            raise CompositionError(
                f"Клетки ({column}, {row}) нет в сетке "
                f"{self.grid.columns}x{self.grid.rows} (индексы с нуля)"
            )
        w, h = self.cell_size()
        return column * w, row * h, (column + 1) * w, (row + 1) * h

    def move_to_cell(self, item_id: str, column: int, row: int, get_image: GetImage) -> Placement:
        """Ставит объект в центр клетки (учитывая масштаб и поворот)."""
        x0, y0, x1, y1 = self.cell_rect(column, row)
        item = self.get(item_id)
        bx0, by0, bx1, by1 = self.bbox(item_id, get_image)
        item.x += (x0 + x1) / 2 - (bx0 + bx1) / 2
        item.y += (y0 + y1) / 2 - (by0 + by1) / 2
        return item

    # ---------- геометрия и проверка ----------

    def item_size(self, item_id: str, get_image: GetImage) -> tuple[float, float]:
        item = self.get(item_id)
        image = get_image(item.image_id)
        return image.width * item.scale, image.height * item.scale

    def bbox(self, item_id: str, get_image: GetImage) -> Rect:
        """Рамка объекта на холсте с учётом масштаба и поворота."""
        item = self.get(item_id)
        w, h = self.item_size(item_id, get_image)
        cx, cy = item.x + w / 2, item.y + h / 2
        angle = math.radians(item.rotation)
        cos, sin = math.cos(angle), math.sin(angle)
        xs, ys = [], []
        for dx, dy in ((-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)):
            xs.append(cx + dx * cos - dy * sin)
            ys.append(cy + dx * sin + dy * cos)
        return min(xs), min(ys), max(xs), max(ys)

    def validate(
        self,
        get_image: GetImage,
        tolerance: float = EPS,
        *,
        strict_bounds: bool | None = None,
    ) -> ValidationReport:
        """Проверяет композицию по правилам слоёв и границ холста:

        1. Границы холста (out_of_bounds):
           - При жёсткой рамке (strict=True): весь bbox объекта должен лежать внутри [0..width, 0..height].
           - При гибкой рамке (strict=False): объект может частично выступать за края холста,
             но его центр обязан находиться внутри холста (чтобы объект не улетел за пределы сцены).
           Если параметр strict_bounds не передан явно в validate(), используется настройка слоя (layer.strict_bounds).

        2. Пересечения (overlaps):
           - Проверяются ТОЛЬКО между объектами ОДНОГО И ТОГО ЖЕ слоя (если у слоя allow_internal_overlap=False).
           - Объекты, находящиеся в РАЗНЫХ слоях, могут свободно перекрывать друг друга.
        """
        report = ValidationReport()

        for layer in self.layers:
            if not layer.visible:
                continue

            boxes = {item.id: self.bbox(item.id, get_image) for item in layer.items}
            use_strict = layer.strict_bounds if strict_bounds is None else strict_bounds

            # 1. Проверка границ холста
            for item_id, (x0, y0, x1, y1) in boxes.items():
                if use_strict:
                    if (
                        x0 < -tolerance
                        or y0 < -tolerance
                        or x1 > self.width + tolerance
                        or y1 > self.height + tolerance
                    ):
                        report.out_of_bounds.append(item_id)
                else:
                    # Гибкая рамка: проверяем, что центр объекта остаётся в пределах холста
                    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                    if (
                        cx < -tolerance
                        or cy < -tolerance
                        or cx > self.width + tolerance
                        or cy > self.height + tolerance
                    ):
                        report.out_of_bounds.append(item_id)

            # 2. Проверка пересечений внутри одного слоя
            if not layer.allow_internal_overlap:
                ids = list(boxes)
                for i, a_id in enumerate(ids):
                    ax0, ay0, ax1, ay1 = boxes[a_id]
                    for b_id in ids[i + 1 :]:
                        bx0, by0, bx1, by1 = boxes[b_id]
                        iw = min(ax1, bx1) - max(ax0, bx0)
                        ih = min(ay1, by1) - max(ay0, by0)
                        if iw > tolerance and ih > tolerance:
                            report.overlaps.append(Overlap(layer.id, a_id, b_id, iw * ih))

        return report

    # ---------- сериализация ----------

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "canvas": {
                "width": self.width,
                "height": self.height,
                "strict_bounds": self.strict_bounds,
            },
            "grid": {"columns": self.grid.columns, "rows": self.grid.rows},
            "layers": [
                {
                    "id": layer.id,
                    "name": layer.name,
                    "allow_internal_overlap": layer.allow_internal_overlap,
                    "strict_bounds": layer.strict_bounds,
                    "visible": layer.visible,
                    "items": [
                        {
                            "id": it.id,
                            "image_id": it.image_id,
                            "x": it.x,
                            "y": it.y,
                            "scale": it.scale,
                            "rotation": it.rotation,
                            "colors": dict(it.colors),
                        }
                        for it in layer.items
                    ],
                }
                for layer in self.layers
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> Composition:
        canvas_data = data["canvas"]
        strict_bounds = canvas_data.get("strict_bounds", True)

        layers: list[Layer] = []
        if "layers" in data:
            for raw_layer in data["layers"]:
                layer_items: list[Placement] = []
                for raw in raw_layer.get("items", []):
                    _check_scale(raw["scale"])
                    layer_items.append(
                        Placement(
                            id=raw["id"],
                            image_id=raw["image_id"],
                            x=raw["x"],
                            y=raw["y"],
                            scale=raw["scale"],
                            rotation=raw["rotation"],
                            colors=_normalize_colors(raw.get("colors")),
                        )
                    )
                layers.append(
                    Layer(
                        id=raw_layer["id"],
                        name=raw_layer.get("name", "Layer"),
                        items=layer_items,
                        allow_internal_overlap=raw_layer.get("allow_internal_overlap", False),
                        strict_bounds=raw_layer.get("strict_bounds", strict_bounds),
                        visible=raw_layer.get("visible", True),
                    )
                )
        else:
            # Обратная совместимость со старым форматом JSON (где был только плоский список items)
            layer_items = []
            for raw in data.get("items", []):
                _check_scale(raw["scale"])
                layer_items.append(
                    Placement(
                        id=raw["id"],
                        image_id=raw["image_id"],
                        x=raw["x"],
                        y=raw["y"],
                        scale=raw["scale"],
                        rotation=raw["rotation"],
                        colors=_normalize_colors(raw.get("colors")),
                    )
                )
            layers.append(Layer(name="Main", items=layer_items, strict_bounds=strict_bounds))

        return cls(
            width=canvas_data["width"],
            height=canvas_data["height"],
            grid=Grid(data["grid"]["columns"], data["grid"]["rows"]),
            strict_bounds=strict_bounds,
            name=data.get("name"),
            id=data["id"],
            layers=layers,
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_json(cls, text: str) -> Composition:
        return cls.from_dict(json.loads(text))


