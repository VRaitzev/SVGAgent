"""Рендеринг отдельных объектов, слоёв и всей композиции в SVG, PNG и структуру для фронтенда.

Отвечает за:
- применение словаря перекраски Placement.colors к XML-дереву иллюстрации без мутации SVGImage;
- изоляцию внутренних id (градиентов, clipPath, масок), чтобы разные SVG не конфликтовали на одном холсте;
- сборку единого векторного SVG с группировкой по слоям (<g id="layer_...">) и трансформациями объектов;
- растеризацию отдельного объекта, слоя или всей композиции в PNG через resvg_py.
"""

from __future__ import annotations

import copy
import re
import xml.etree.ElementTree as ET

import resvg_py

from ar.composition import Composition, GetImage, Layer, Placement
from ar.svg_image import SVG_NS, SVGImage, normalize_color

ET.register_namespace("", SVG_NS)
ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")

_COLOR_ATTRS = ("fill", "stroke", "stop-color", "flood-color", "lighting-color")
_STYLE_DECL = re.compile(r"(fill|stroke|stop-color|flood-color|lighting-color)(\s*:\s*)([^;]+)")
_URL_REF = re.compile(r"url\(\s*['\"]?#([^'\"\)\s]+)['\"]?\s*\)")


class RenderError(Exception):
    """Ошибка при рендеринге SVG или PNG."""


def recolor_element(root: ET.Element, color_map: dict[str, str]) -> ET.Element:
    """Возвращает копию XML-дерева с подменёнными цветами согласно color_map."""
    if not color_map:
        return copy.deepcopy(root)

    norm_map: dict[str, str] = {}
    for old_c, new_c in color_map.items():
        k, v = normalize_color(old_c), normalize_color(new_c)
        if k and v:
            norm_map[k] = v

    if not norm_map:
        return copy.deepcopy(root)

    tree = copy.deepcopy(root)

    def _replace_style(style_text: str) -> str:
        def _sub(match: re.Match) -> str:
            prop, sep, val = match.group(1), match.group(2), match.group(3)
            norm_val = normalize_color(val)
            if norm_val and norm_val in norm_map:
                return f"{prop}{sep}{norm_map[norm_val]}"
            return match.group(0)

        return _STYLE_DECL.sub(_sub, style_text)

    for el in tree.iter():
        for attr in _COLOR_ATTRS:
            raw = el.get(attr)
            if raw:
                norm_val = normalize_color(raw)
                if norm_val and norm_val in norm_map:
                    el.set(attr, norm_map[norm_val])

        style = el.get("style")
        if style:
            el.set("style", _replace_style(style))

        if el.tag.endswith("style") and el.text:
            el.text = _replace_style(el.text)

    return tree


def namespace_ids(root: ET.Element, prefix: str) -> ET.Element:
    """Добавляет префикс ко всем внутренним id и ссылкам url(#id) / href='#id',
    чтобы градиенты и маски разных объектов не перезаписывали друг друга на общем холсте.
    """
    tree = copy.deepcopy(root)
    ids: set[str] = set()
    for el in tree.iter():
        el_id = el.get("id")
        if el_id:
            ids.add(el_id)
            el.set("id", f"{prefix}{el_id}")

    if not ids:
        return tree

    def _sub_url(match: re.Match) -> str:
        ref = match.group(1)
        return f"url(#{prefix}{ref})" if ref in ids else match.group(0)

    for el in tree.iter():
        for attr, val in list(el.attrib.items()):
            if "url(" in val:
                el.set(attr, _URL_REF.sub(_sub_url, val))
            elif attr in ("href", "{http://www.w3.org/1999/xlink}href") and val.startswith("#"):
                ref = val[1:]
                if ref in ids:
                    el.set(attr, f"#{prefix}{ref}")

        if el.tag.endswith("style") and el.text and "url(" in el.text:
            el.text = _URL_REF.sub(_sub_url, el.text)

    return tree


class CompositionRenderer:
    """Рендерит отдельные объекты (Placement), слои (Layer) и всю композицию (Composition) в SVG и PNG."""

    def __init__(self, get_image: GetImage) -> None:
        self.get_image = get_image

    # ---------- рендер одного объекта (Placement) ----------

    def render_item_svg(self, item: Placement) -> str:
        """Возвращает перекрашенный SVG-код одного объекта (без внешних координат холста)."""
        image: SVGImage = self.get_image(item.image_id)
        if not item.colors:
            return image.svg
        recolored_root = recolor_element(image._root, item.colors)
        return ET.tostring(recolored_root, encoding="unicode")

    def render_item_png(self, item: Placement, width: int | None = None) -> bytes:
        """Рендерит один объект (с учётом его перекраски и scale) в PNG-байты."""
        image = self.get_image(item.image_id)
        svg_str = self.render_item_svg(item)
        target_width = width if width is not None else max(1, round(image.width * item.scale))
        try:
            return bytes(resvg_py.svg_to_bytes(svg_string=svg_str, width=target_width))
        except ValueError as e:
            raise RenderError(f"Не удалось отрисовать объект {item.id} в PNG: {e}") from e

    # ---------- вспомогательная сборка узла объекта на холсте ----------

    def _build_item_node(self, item: Placement) -> ET.Element:
        image = self.get_image(item.image_id)
        w = image.width * item.scale
        h = image.height * item.scale
        vb_x, vb_y, vb_w, vb_h = image.viewbox

        item_root = recolor_element(image._root, item.colors)
        item_root = namespace_ids(item_root, prefix=f"item_{item.id[:8]}_")

        transforms = [f"translate({item.x:g}, {item.y:g})"]
        if abs(item.rotation) > 1e-6:
            transforms.append(f"rotate({item.rotation:g}, {w / 2:g}, {h / 2:g})")

        group = ET.Element(
            f"{{{SVG_NS}}}g",
            {
                "id": f"item_{item.id}",
                "transform": " ".join(transforms),
            },
        )

        nested_attrib = dict(item_root.attrib)
        nested_attrib["x"] = "0"
        nested_attrib["y"] = "0"
        nested_attrib["width"] = f"{w:g}"
        nested_attrib["height"] = f"{h:g}"
        nested_attrib["viewBox"] = f"{vb_x:g} {vb_y:g} {vb_w:g} {vb_h:g}"

        nested_svg = ET.SubElement(group, f"{{{SVG_NS}}}svg", nested_attrib)
        for child in list(item_root):
            nested_svg.append(child)

        return group

    # ---------- рендер одного смыслового слоя (Layer) на холсте ----------

    def render_layer_svg(
        self,
        comp: Composition,
        layer_id_or_name: str,
        *,
        background: str | None = None,
    ) -> str:
        """Рендерит один конкретный слой (Layer) в размерах холста в SVG."""
        layer: Layer = comp.get_layer(layer_id_or_name)
        canvas_svg = ET.Element(
            f"{{{SVG_NS}}}svg",
            {
                "width": f"{comp.width:g}",
                "height": f"{comp.height:g}",
                "viewBox": f"0 0 {comp.width:g} {comp.height:g}",
                "fill": "none",
            },
        )
        if background:
            bg_color = normalize_color(background) or background
            ET.SubElement(
                canvas_svg,
                f"{{{SVG_NS}}}rect",
                {"width": f"{comp.width:g}", "height": f"{comp.height:g}", "fill": bg_color},
            )

        layer_group = ET.SubElement(
            canvas_svg,
            f"{{{SVG_NS}}}g",
            {"id": f"layer_{layer.id}", "data-layer-name": layer.name},
        )
        for item in layer.items:
            layer_group.append(self._build_item_node(item))

        return ET.tostring(canvas_svg, encoding="unicode")

    def render_layer_png(
        self,
        comp: Composition,
        layer_id_or_name: str,
        *,
        width: int | None = None,
        background: str | None = None,
    ) -> bytes:
        """Рендерит один конкретный слой (Layer) в PNG-байты."""
        svg_str = self.render_layer_svg(comp, layer_id_or_name, background=background)
        target_width = width if width is not None else max(1, round(comp.width))
        try:
            return bytes(resvg_py.svg_to_bytes(svg_string=svg_str, width=target_width))
        except ValueError as e:
            raise RenderError(f"Не удалось отрисовать слой {layer_id_or_name!r} в PNG: {e}") from e

    # ---------- рендер для фронтенда ----------

    def to_frontend_payload(self, comp: Composition) -> dict:
        """Собирает полную структуру композиции со слоями и готовыми SVG каждого объекта для фронтенда."""
        rendered_layers: list[dict] = []
        flat_items: list[dict] = []

        for l_idx, layer in enumerate(comp.layers):
            layer_items: list[dict] = []
            for i_idx, item in enumerate(layer.items):
                image = self.get_image(item.image_id)
                w = image.width * item.scale
                h = image.height * item.scale
                item_dict = {
                    "id": item.id,
                    "image_id": item.image_id,
                    "name": image.name,
                    "layer_id": layer.id,
                    "layer_name": layer.name,
                    "layer_index": l_idx,
                    "item_index": i_idx,
                    "x": item.x,
                    "y": item.y,
                    "width": w,
                    "height": h,
                    "original_width": image.width,
                    "original_height": image.height,
                    "scale": item.scale,
                    "rotation": item.rotation,
                    "colors": dict(item.colors),
                    "original_colors": list(image.colors),
                    "svg": self.render_item_svg(item),
                }
                layer_items.append(item_dict)
                if layer.visible:
                    flat_items.append(item_dict)

            rendered_layers.append(
                {
                    "id": layer.id,
                    "name": layer.name,
                    "index": l_idx,
                    "allow_internal_overlap": layer.allow_internal_overlap,
                    "strict_bounds": layer.strict_bounds,
                    "visible": layer.visible,
                    "items": layer_items,
                }
            )

        return {
            "id": comp.id,
            "name": comp.name,
            "canvas": {
                "width": comp.width,
                "height": comp.height,
                "strict_bounds": comp.strict_bounds,
            },
            "grid": {"columns": comp.grid.columns, "rows": comp.grid.rows},
            "layers": rendered_layers,
            "items": flat_items,
        }

    # ---------- рендер всей композиции ----------

    def render_composition_svg(
        self,
        comp: Composition,
        *,
        background: str | None = None,
    ) -> str:
        """Собирает всю композицию в единый векторный SVG-файл с группировкой по слоям."""
        canvas_svg = ET.Element(
            f"{{{SVG_NS}}}svg",
            {
                "width": f"{comp.width:g}",
                "height": f"{comp.height:g}",
                "viewBox": f"0 0 {comp.width:g} {comp.height:g}",
            },
        )

        if background:
            bg_color = normalize_color(background) or background
            ET.SubElement(
                canvas_svg,
                f"{{{SVG_NS}}}rect",
                {
                    "width": f"{comp.width:g}",
                    "height": f"{comp.height:g}",
                    "fill": bg_color,
                },
            )

        for layer in comp.layers:
            if not layer.visible:
                continue
            layer_group = ET.SubElement(
                canvas_svg,
                f"{{{SVG_NS}}}g",
                {
                    "id": f"layer_{layer.id}",
                    "data-layer-name": layer.name,
                },
            )
            for item in layer.items:
                layer_group.append(self._build_item_node(item))

        return ET.tostring(canvas_svg, encoding="unicode")

    def render_composition_png(
        self,
        comp: Composition,
        *,
        width: int | None = None,
        background: str | None = None,
    ) -> bytes:
        """Рендерит всю композицию в PNG-байты (для Vision-модели агента или превью)."""
        svg_str = self.render_composition_svg(comp, background=background)
        target_width = width if width is not None else max(1, round(comp.width))
        try:
            return bytes(resvg_py.svg_to_bytes(svg_string=svg_str, width=target_width))
        except ValueError as e:
            raise RenderError(f"Не удалось отрисовать композицию {comp.id} в PNG: {e}") from e