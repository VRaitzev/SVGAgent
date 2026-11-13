"""Модель и парсер отдельного SVG-компонента:
- автоматический парсинг реальных width, height, viewbox и colors в __post_init__ (при любом способе создания);
- свойство _root (ET.Element) для полной совместимости с renderer.py;
- нормализация всех CSS-цветов (#rgb, #rrggbb, #rrggbbaa, rgb(), именованные цвета);
- единая генерация concept_key с очисткой (L)/(S).
"""

from __future__ import annotations

import hashlib
import re
import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field, replace
from typing import Literal

SVG_NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG_NS)

DetailLevel = Literal["L", "S"]
ObjectRole = Literal["main", "background"]

CSS_NAMED_COLORS: dict[str, str] = {
    "white": "#ffffff",
    "black": "#000000",
    "red": "#ff0000",
    "green": "#008000",
    "blue": "#0000ff",
    "yellow": "#ffff00",
    "cyan": "#00ffff",
    "magenta": "#ff00ff",
    "gray": "#808080",
    "grey": "#808080",
    "silver": "#c0c0c0",
    "maroon": "#800000",
    "olive": "#808000",
    "lime": "#00ff00",
    "aqua": "#00ffff",
    "teal": "#008080",
    "navy": "#000080",
    "fuchsia": "#ff00ff",
    "purple": "#800080",
    "orange": "#ffa500",
    "pink": "#ffc0cb",
    "violet": "#ee82ee",
    "indigo": "#4b0082",
    "gold": "#ffd700",
    "coral": "#ff7f50",
    "salmon": "#fa8072",
    "khaki": "#f0e68c",
    "plum": "#dda0dd",
    "lavender": "#e6e6fa",
    "beige": "#f5f5dc",
    "ivory": "#fffff0",
    "snow": "#fffafa",
    "whitesmoke": "#f5f5f5",
    "lightgray": "#d3d3d3",
    "lightgrey": "#d3d3d3",
    "darkgray": "#a9a9a9",
    "darkgrey": "#a9a9a9",
    "dimgray": "#696969",
    "dimgrey": "#696969",
    "slategray": "#708090",
    "slategrey": "#708090",
    "darkslategray": "#2f4f4f",
    "darkslategrey": "#2f4f4f",
}

NON_COLOR_VALUES = {"none", "transparent", "currentcolor", "inherit", "initial", "unset"}

_RGB_RE = re.compile(
    r"^rgba?\(\s*(\d{1,3})\s*,\s*(\d{1,3})\s*,\s*(\d{1,3})(?:\s*,\s*[\d\.]+)?\s*\)$",
    re.IGNORECASE,
)
_COLOR_ATTRS = ("fill", "stroke", "stop-color", "flood-color", "lighting-color")
_STYLE_COLOR_RE = re.compile(
    r"(fill|stroke|stop-color|flood-color|lighting-color)\s*:\s*([^;\"']+)",
    re.IGNORECASE,
)


def compute_hash(svg_text: str) -> str:
    """Вычисляет детерминированный SHA-256 хэш SVG-разметки для дедупликации в хранилище."""
    normalized = re.sub(r"\s+", " ", svg_text.strip())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def normalize_color(value: str | None) -> str | None:
    """Приводит любой CSS-цвет (#rgb, #rrggbb, #rrggbbaa, rgb(), именованный) к нижнему регистру #rrggbb.
    Возвращает None для 'none', 'transparent', 'url(...)' и некорректных значений.
    """
    if not value:
        return None
    raw = value.strip().lower()
    if not raw or raw in NON_COLOR_VALUES or raw.startswith("url("):
        return None

    if raw in CSS_NAMED_COLORS:
        return CSS_NAMED_COLORS[raw]

    if raw.startswith("#"):
        hex_part = raw[1:]
        if len(hex_part) == 3 and all(c in "0123456789abcdef" for c in hex_part):
            return "#" + "".join(c * 2 for c in hex_part)
        if len(hex_part) == 4 and all(c in "0123456789abcdef" for c in hex_part):
            if hex_part[3] == "0":
                return None
            return "#" + "".join(c * 2 for c in hex_part[:3])
        if len(hex_part) == 6 and all(c in "0123456789abcdef" for c in hex_part):
            return "#" + hex_part
        if len(hex_part) == 8 and all(c in "0123456789abcdef" for c in hex_part):
            if hex_part[6:8] == "00":
                return None
            return "#" + hex_part[:6]
        return None

    m = _RGB_RE.match(raw)
    if m:
        r = max(0, min(255, int(m.group(1))))
        g = max(0, min(255, int(m.group(2))))
        b = max(0, min(255, int(m.group(3))))
        return f"#{r:02x}{g:02x}{b:02x}"

    return None


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    norm = normalize_color(hex_color) or "#000000"
    h = norm.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def color_luminance(color: str) -> float:
    """Относительная яркость цвета от 0.0 (чёрный) до 1.0 (белый)."""
    r, g, b = hex_to_rgb(color)
    return (0.299 * r + 0.587 * g + 0.114 * b) / 255.0


def is_structural_color(color: str | None) -> bool:
    """Проверяет, является ли цвет конструктивным (чёрный/тёмный контур, падающая тень или чисто белый блик/страница),
    который НЕ должен перекрашиваться в акцентные цвета палитры.
    """
    norm = normalize_color(color)
    if not norm:
        return True
    r, g, b = hex_to_rgb(norm)
    lum = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
    max_c, min_c = max(r, g, b), min(r, g, b)
    saturation = (max_c - min_c) / max_c if max_c > 0 else 0.0

    # Чёрные и тёмно-сланцевые контуры/тени (#000000, #1e293b, #231f20 и т.д.)
    if lum < 0.22 and saturation < 0.45:
        return True
    # Чисто белые или почти белые нейтральные страницы/блики (#ffffff, #f8fafc)
    if lum > 0.95 and saturation < 0.12:
        return True
    return False


def derive_concept_key(name: str | None, explicit_key: str | None = None) -> str:
    """Генерирует единый concept_key для пар L и S, корректно очищая '(L)', '(S)', '_L', '-S', 'main', 'bg'."""
    if explicit_key and explicit_key.strip():
        base = explicit_key.strip().lower()
    else:
        base = (name or "item").strip().lower()

    # Убираем скобки и суффиксы размера/роли
    base = re.sub(r"[\(\[\{]\s*(?:размер\s*|size\s*)?(?:l|s|large|small|main|bg|background|фон|главный)\s*[\)\]\}]", " ", base, flags=re.IGNORECASE)
    base = re.sub(
        r"(?:^|[_\-\s])(size[_\-\s]*|размер[_\-\s]*)?(l|s|large|small|main|bg|background|фон|фоновое|главный|главные)(?=$|[_\-\s])",
        " ",
        base,
        flags=re.IGNORECASE,
    )
    slug = re.sub(r"[^a-zа-яё0-9]+", "_", base, flags=re.IGNORECASE).strip("_")
    return slug or "item"


def ensure_svg_xmlns(svg_text: str) -> str:
    """Гарантирует наличие xmlns='http://www.w3.org/2000/svg' у корневого тега <svg>."""
    cleaned = svg_text.strip()
    if "<svg" in cleaned and "xmlns=" not in cleaned[:300]:
        cleaned = re.sub(
            r"<svg\b",
            f'<svg xmlns="{SVG_NS}"',
            cleaned,
            count=1,
            flags=re.IGNORECASE,
        )
    return cleaned


def _parse_svg_metadata(
    svg_text: str,
    *,
    include_structural: bool = False,
) -> tuple[ET.Element, float, float, tuple[float, float, float, float], tuple[str, ...]]:
    clean_svg = ensure_svg_xmlns(svg_text)
    root = ET.fromstring(clean_svg)
    min_x, min_y = 0.0, 0.0
    width: float | None = None
    height: float | None = None

    w_raw = re.sub(r"[^\d\.]+", "", root.attrib.get("width", ""))
    h_raw = re.sub(r"[^\d\.]+", "", root.attrib.get("height", ""))
    if w_raw:
        try:
            width = float(w_raw)
        except ValueError:
            pass
    if h_raw:
        try:
            height = float(h_raw)
        except ValueError:
            pass

    vb = root.attrib.get("viewBox") or root.attrib.get("viewbox")
    vb_tuple: tuple[float, float, float, float] | None = None
    if vb:
        parts = [float(p) for p in re.split(r"[\s,]+", vb.strip()) if p]
        if len(parts) == 4 and parts[2] > 0 and parts[3] > 0:
            min_x, min_y, vb_w, vb_h = parts[0], parts[1], parts[2], parts[3]
            vb_tuple = (min_x, min_y, vb_w, vb_h)
            if width is None or width <= 0:
                width = vb_w
            if height is None or height <= 0:
                height = vb_h

    final_w = width if (width is not None and width > 0) else 200.0
    final_h = height if (height is not None and height > 0) else 200.0
    final_vb = vb_tuple if vb_tuple is not None else (0.0, 0.0, final_w, final_h)

    found_colors: list[str] = []
    seen: set[str] = set()

    def _register(raw_val: str | None) -> None:
        norm = normalize_color(raw_val)
        if not norm or norm in seen:
            return
        if not include_structural and is_structural_color(norm):
            return
        seen.add(norm)
        found_colors.append(norm)

    for elem in root.iter():
        for attr in _COLOR_ATTRS:
            if attr in elem.attrib:
                _register(elem.attrib[attr])
        style_str = elem.attrib.get("style")
        if style_str:
            for _, val in _STYLE_COLOR_RE.findall(style_str):
                _register(val)
        if elem.tag.endswith("style") and elem.text:
            for _, val in _STYLE_COLOR_RE.findall(elem.text):
                _register(val)

    return root, final_w, final_h, final_vb, tuple(found_colors)
    def _register(raw_val: str | None) -> None:
        norm = normalize_color(raw_val)
        if norm and not is_structural_color(norm) and norm not in seen:
            seen.add(norm)
            found_colors.append(norm)

    for elem in root.iter():
        for attr in _COLOR_ATTRS:
            if attr in elem.attrib:
                _register(elem.attrib[attr])
        style_str = elem.attrib.get("style")
        if style_str:
            for _, val in _STYLE_COLOR_RE.findall(style_str):
                _register(val)
        if elem.tag.endswith("style") and elem.text:
            for _, val in _STYLE_COLOR_RE.findall(elem.text):
                _register(val)

    return root, final_w, final_h, final_vb, tuple(found_colors)


def recolor_svg_markup(svg_text: str, color_map: dict[str, str]) -> str:
    """Перекрашивает SVG-разметку, заменяя любые представления исходных цветов
    (включая именованные, #rgb и #rrggbbaa с сохранением альфа-канала) на целевые цвета палитры.
    """
    if not color_map:
        return ensure_svg_xmlns(svg_text)

    normalized_map: dict[str, str] = {}
    for k, v in color_map.items():
        nk = normalize_color(k)
        nv = normalize_color(v)
        if nk and nv:
            normalized_map[nk] = nv

    if not normalized_map:
        return ensure_svg_xmlns(svg_text)

    def _replace_color_token(raw_val: str) -> str:
        val_clean = raw_val.strip()
        norm = normalize_color(val_clean)
        if not norm or norm not in normalized_map:
            return raw_val
        target_hex = normalized_map[norm]
        if val_clean.startswith("#") and len(val_clean) == 9:
            return target_hex + val_clean[7:9]
        return target_hex

    def _attr_repl(match: re.Match) -> str:
        attr_name = match.group(1)
        quote = match.group(2)
        val = match.group(3)
        return f"{attr_name}={quote}{_replace_color_token(val)}{quote}"

    result = re.sub(
        r"\b(fill|stroke|stop-color|flood-color|lighting-color)\s*=\s*([\"'])([^\"']+)\2",
        _attr_repl,
        ensure_svg_xmlns(svg_text),
        flags=re.IGNORECASE,
    )

    def _style_repl(match: re.Match) -> str:
        prop = match.group(1)
        val = match.group(2)
        return f"{prop}:{_replace_color_token(val)}"

    result = re.sub(_STYLE_COLOR_RE, _style_repl, result)
    return result


@dataclass(frozen=True)
class SVGImage:
    svg: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    name: str | None = "Без названия"
    description: str | None = ""
    detail_level: DetailLevel = "L"
    role: ObjectRole = "main"
    concept_key: str | None = ""
    width: float = 0.0
    height: float = 0.0
    colors: tuple[str, ...] = field(default_factory=tuple)
    hash: str = ""
    source_path: str | None = None
    png_path: str | None = None
    created_at: str | None = None
    viewbox: tuple[float, float, float, float] | None = None
    _root: ET.Element | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        clean_svg = ensure_svg_xmlns(self.svg) if self.svg else ""
        if clean_svg != self.svg:
            object.__setattr__(self, "svg", clean_svg)

        if not self.hash and clean_svg:
            object.__setattr__(self, "hash", compute_hash(clean_svg))

        if not self.concept_key:
            object.__setattr__(self, "concept_key", derive_concept_key(self.name))

        # Всегда парсим реальные габариты, viewbox, цвета и XML-корень _root из SVG!
        if clean_svg:
            try:
                root, parsed_w, parsed_h, parsed_vb, parsed_cols = _parse_svg_metadata(
                    clean_svg,
                    include_structural=(self.role == "background"),
                )
                object.__setattr__(self, "_root", root)
                if self.width <= 0:
                    object.__setattr__(self, "width", parsed_w)
                if self.height <= 0:
                    object.__setattr__(self, "height", parsed_h)
                if self.viewbox is None:
                    object.__setattr__(self, "viewbox", parsed_vb)
                if not self.colors:
                    object.__setattr__(self, "colors", parsed_cols)
                return
            except Exception:
                pass

        if self.width <= 0:
            object.__setattr__(self, "width", 200.0)
        if self.height <= 0:
            object.__setattr__(self, "height", 200.0)
        if self.viewbox is None:
            object.__setattr__(self, "viewbox", (0.0, 0.0, float(self.width), float(self.height)))

    def with_colors(self, color_map: dict[str, str]) -> "SVGImage":
        """Возвращает копию SVGImage с перекрашенной SVG-разметкой."""
        if not color_map:
            return self
        new_svg = recolor_svg_markup(self.svg, color_map)
        root, w, h, vb, new_cols = _parse_svg_metadata(new_svg)
        return replace(
            self,
            svg=new_svg,
            width=w,
            height=h,
            viewbox=vb,
            colors=new_cols,
            hash=compute_hash(new_svg),
            _root=root,
        )

    @classmethod
    def from_svg(
        cls,
        svg: str,
        *,
        id: str | None = None,
        name: str = "Без названия",
        description: str | None = None,
        detail_level: DetailLevel = "L",
        role: ObjectRole = "main",
        concept_key: str | None = None,
        source_path: str | None = None,
        png_path: str | None = None,
        created_at: str | None = None,
    ) -> "SVGImage":
        clean_svg = ensure_svg_xmlns(svg)
        lvl: DetailLevel = "S" if str(detail_level).upper() == "S" else "L"
        rl: ObjectRole = "background" if str(role).lower() == "background" else "main"
        ckey = derive_concept_key(name, explicit_key=concept_key)
        return cls(
            svg=clean_svg,
            id=id or uuid.uuid4().hex,
            name=name,
            description=description or "",
            detail_level=lvl,
            role=rl,
            concept_key=ckey,
            hash=compute_hash(clean_svg),
            source_path=source_path,
            png_path=png_path,
            created_at=created_at,
        )