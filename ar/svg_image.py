"""Объект иллюстрации: то, с чем работают хранилище, композиции и агент."""

from __future__ import annotations

import hashlib
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import cached_property

SVG_NS = "http://www.w3.org/2000/svg"

_NON_COLORS = {"none", "currentcolor", "inherit", "transparent", "context-fill", "context-stroke", ""}
_COLOR_ATTRS = ("fill", "stroke", "stop-color", "flood-color", "lighting-color")
_STYLE_DECL = re.compile(r"(fill|stroke|stop-color|flood-color|lighting-color)\s*:\s*([^;]+)")
_NUMBER = re.compile(r"^\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")


def compute_hash(svg: str) -> str:
    return hashlib.sha256(svg.encode("utf-8")).hexdigest()


def normalize_color(value: str) -> str | None:
    v = value.strip().lower()
    if v in _NON_COLORS or v.startswith("url("):
        return None
    if re.fullmatch(r"#[0-9a-f]{3}", v):  # #abc -> #aabbcc
        v = "#" + "".join(ch * 2 for ch in v[1:])
    return v


def _number(value: str | None) -> float | None:
    if not value:
        return None
    m = _NUMBER.match(value)
    return float(m.group(1)) if m else None


@dataclass(frozen=True)
class SVGImage:
    """Иллюстрация из библиотеки.

    Хранимые поля: id, svg, hash, name, source_path, png_path, description.
    Всё остальное (размеры, цвета) считается из самого SVG.
    """

    id: str
    svg: str
    hash: str
    name: str | None = None
    source_path: str | None = None  # откуда загружен исходный .svg (если из файла)
    png_path: str | None = None  # растровое превью для vision/мультимодальной модели
    description: str | None = None  # текстовое описание (необязательное)

    @cached_property
    def _root(self) -> ET.Element:
        return ET.fromstring(self.svg)

    @cached_property
    def viewbox(self) -> tuple[float, float, float, float]:
        """(min_x, min_y, width, height) в координатах самой иллюстрации."""
        vb = self._root.get("viewBox")
        if vb:
            parts = [float(p) for p in re.split(r"[\s,]+", vb.strip()) if p]
            if len(parts) == 4 and parts[2] > 0 and parts[3] > 0:
                return tuple(parts)  # type: ignore[return-value]
        w, h = _number(self._root.get("width")), _number(self._root.get("height"))
        if w and h:
            return (0.0, 0.0, w, h)
        raise ValueError("У SVG нет ни viewBox, ни width/height")

    @property
    def width(self) -> float:
        return self.viewbox[2]

    @property
    def height(self) -> float:
        return self.viewbox[3]

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height

    @cached_property
    def colors(self) -> tuple[str, ...]:
        """Уникальные цвета (fill, stroke, stop-color…) в порядке появления."""
        found: dict[str, None] = {}
        for el in self._root.iter():
            candidates = [el.get(a) for a in _COLOR_ATTRS]
            candidates += [m.group(2) for m in _STYLE_DECL.finditer(el.get("style", ""))]
            if el.tag.endswith("style") and el.text:
                candidates += [m.group(2) for m in _STYLE_DECL.finditer(el.text)]
            for raw in candidates:
                if raw and (c := normalize_color(raw)):
                    found[c] = None
        return tuple(found)
