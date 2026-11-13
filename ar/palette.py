"""Цветовая палитра и автоматический маппинг цветов для иллюстраций и композиций.

Палитра (Palette) хранит упорядоченный набор целевых HEX-цветов и умеет
сопоставлять исходные цвета SVG-элементов с цветами палитры по светлоте
(относительной яркости), чтобы сохранять контраст между контурами, заливками и бликами.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass

from ar.composition import Composition, GetImage, Placement
from ar.svg_image import SVGImage, normalize_color

# Базовая таблица именованных CSS-цветов на случай, если в SVG встретится не HEX
_NAMED_COLORS: dict[str, tuple[int, int, int]] = {
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "red": (255, 0, 0),
    "green": (0, 128, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "cyan": (0, 255, 255),
    "magenta": (255, 0, 255),
    "gray": (128, 128, 128),
    "grey": (128, 128, 128),
    "orange": (255, 165, 0),
    "purple": (128, 0, 128),
    "pink": (255, 192, 203),
    "brown": (165, 42, 42),
    "navy": (0, 0, 128),
    "teal": (0, 128, 128),
    "silver": (192, 192, 192),
}


class PaletteError(Exception):
    """Ошибка работы с цветовой палитрой."""


def parse_rgb(color: str) -> tuple[int, int, int]:
    """Переводит нормализованный цвет (#rrggbb, rgb(...) или имя) в (R, G, B)."""
    c = normalize_color(color)
    if not c:
        raise PaletteError(f"Некорректный цвет: {color!r}")

    if re.fullmatch(r"#[0-9a-f]{6}", c):
        return int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)

    if re.fullmatch(r"#[0-9a-f]{8}", c):
        return int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)

    m = re.match(r"rgba?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)", c)
    if m:
        return (
            min(255, int(m.group(1))),
            min(255, int(m.group(2))),
            min(255, int(m.group(3))),
        )

    if c in _NAMED_COLORS:
        return _NAMED_COLORS[c]

    raise PaletteError(f"Не удалось распознать RGB-значения цвета: {color!r}")


def luminance(color: str) -> float:
    """Воспринимаемая светлота цвета (0.0 — чёрный, 255.0 — белый)."""
    try:
        r, g, b = parse_rgb(color)
    except PaletteError:
        return 128.0
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


@dataclass(frozen=True)
class Palette:
    """Цветовая палитра для автоматической перекраски элементов и композиций."""

    colors: tuple[str, ...]
    name: str | None = None

    def __init__(self, colors: Iterable[str], name: str | None = None) -> None:
        normalized: list[str] = []
        for raw in colors:
            c = normalize_color(raw)
            if not c:
                raise PaletteError(f"Некорректный цвет в палитре: {raw!r}")
            parse_rgb(c)  # проверяем, что цвет валидный и конвертируется в RGB
            if c not in normalized:
                normalized.append(c)

        if not normalized:
            raise PaletteError("Палитра должна содержать хотя бы один валидный цвет")

        object.__setattr__(self, "colors", tuple(normalized))
        object.__setattr__(self, "name", name)

    def map_colors(self, source_colors: Iterable[str]) -> dict[str, str]:
        """Строит словарь замены {'#старый': '#новый'} по рангу светлоты.

        Самые тёмные исходные цвета получают самые тёмные цвета палитры,
        а самые светлые — самые светлые. Это сохраняет читаемость контуров и деталей.
        """
        unique_src: list[str] = []
        for raw in source_colors:
            c = normalize_color(raw)
            if c and c not in unique_src:
                unique_src.append(c)

        if not unique_src:
            return {}

        sorted_src = sorted(unique_src, key=luminance)
        sorted_pal = sorted(self.colors, key=luminance)

        n_src = len(sorted_src)
        n_pal = len(sorted_pal)

        if n_src == 1:
            # Единственный цвет перекрашиваем в средний/основной цвет палитры
            return {sorted_src[0]: sorted_pal[n_pal // 2]}

        mapping: dict[str, str] = {}
        for i, src_color in enumerate(sorted_src):
            pal_idx = round(i * (n_pal - 1) / (n_src - 1))
            mapping[src_color] = sorted_pal[pal_idx]
        return mapping

    def apply_to_item(self, item: Placement, image: SVGImage) -> dict[str, str]:
        """Вычисляет и записывает словарь перекраски в конкретный Placement."""
        mapping = self.map_colors(image.colors)
        item.colors = mapping
        return mapping

    def apply_to_composition(self, comp: Composition, get_image: GetImage) -> None:
        """Автоматически перекрашивает всю композицию в цвета палитры."""
        for item in comp.items:
            image = get_image(item.image_id)
            comp.update(item.id, colors=self.map_colors(image.colors))

    # ---------- сериализация ----------

    def to_dict(self) -> dict:
        return {"name": self.name, "colors": list(self.colors)}

    @classmethod
    def from_dict(cls, data: dict | list[str]) -> Palette:
        if isinstance(data, list):
            return cls(colors=data)
        return cls(colors=data["colors"], name=data.get("name"))

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_json(cls, text: str) -> Palette:
        return cls.from_dict(json.loads(text))


# Готовые пресеты палитр для быстрого выбора в UI и агенте
PRESET_PALETTES: dict[str, Palette] = {
    "corporate_blue": Palette(
        ["#0f172a", "#1e40af", "#3b82f6", "#93c5fd", "#eff6ff"],
        name="Corporate Blue",
    ),
    "warm_sunset": Palette(
        ["#431407", "#9a3412", "#ea580c", "#fdba74", "#ffedd5"],
        name="Warm Sunset",
    ),
    "emerald_mint": Palette(
        ["#064e3b", "#047857", "#10b981", "#6ee7b7", "#ecfdf5"],
        name="Emerald Mint",
    ),
    "cyber_violet": Palette(
        ["#1e1b4b", "#4c1d95", "#8b5cf6", "#c4b5fd", "#f5f3ff"],
        name="Cyber Violet",
    ),
    "monochrome": Palette(
        ["#111827", "#374151", "#6b7280", "#d1d5db", "#f9fafb"],
        name="Monochrome",
    ),
}
