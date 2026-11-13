"""Цветовые схемы (палитры) с сохранением 3D-объёма, контуров и мягких теней у фоновых монохромных объектов."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal

from svg_image import color_luminance, is_structural_color, normalize_color


@dataclass
class Palette:
    name: str
    title: str
    bg_colors: list[str] = field(default_factory=list)
    accent_colors: list[str] = field(default_factory=list)
    canvas_bg: str = "#ffffff"

    def __post_init__(self) -> None:
        self.bg_colors = [normalize_color(c) or c.lower() for c in self.bg_colors if normalize_color(c)]
        self.accent_colors = [normalize_color(c) or c.lower() for c in self.accent_colors if normalize_color(c)]
        self.canvas_bg = normalize_color(self.canvas_bg) or "#ffffff"

    def allowed_for_role(self, role: Literal["main", "background"]) -> set[str]:
        bg_set = set(self.bg_colors) | {self.canvas_bg}
        if role == "background":
            return bg_set
        return bg_set | set(self.accent_colors)

    def is_color_allowed(self, color: str | None, *, role: Literal["main", "background"] = "main") -> bool:
        if not color:
            return True
        if role == "main" and is_structural_color(color):
            return True
        norm = normalize_color(color)
        if not norm:
            return True
        return norm in self.allowed_for_role(role)

    def map_colors_for_role(
        self,
        original_colors: Iterable[str],
        *,
        role: Literal["main", "background"] = "main",
    ) -> dict[str, str]:
        """Сопоставляет цвета исходного SVG цветам палитры с учётом яркости (тёмный -> тёмный, светлый -> светлый):
        - Для role='background': распределяет ВСЕ цвета объекта (включая контуры, тени и светлые грани)
          по 4 монохромным оттенкам bg_colors от самого тёмного (контур) до самого светлого (верхняя грань),
          чтобы фоновый объект выглядел объёмным, как в референсах Figma, а не плоским пятном!
        - Для role='main': сохраняет чёрные контуры/тени и белые страницы, а цветные детали маппит на accent_colors.
        """
        unique_orig: list[str] = []
        for raw_c in original_colors:
            norm = normalize_color(raw_c)
            if not norm:
                continue
            # Для главных объектов не перекрашиваем конструктивный чёрный/белый
            if role == "main" and is_structural_color(norm):
                continue
            if norm not in unique_orig:
                unique_orig.append(norm)

        if not unique_orig:
            return {}

        # Сортируем исходные цвета от самого тёмного (контур/тень) к самому светлому (верхняя грань/блик)
        sorted_orig = sorted(unique_orig, key=color_luminance)

        if role == "background":
            if not self.bg_colors:
                return {c: self.canvas_bg for c in sorted_orig}

            # Сортируем фоновые оттенки палитры от самого тёмного к самому светлому
            sorted_bg = sorted(self.bg_colors, key=color_luminance)
            n_orig = len(sorted_orig)
            n_bg = len(sorted_bg)
            mapping_bg: dict[str, str] = {}

            for idx, orig_c in enumerate(sorted_orig):
                if n_orig == 1:
                    target_idx = 0
                else:
                    # Равномерно растягиваем все тона объекта по градиенту bg_colors (сохраняя контраст между контуром, тенью и гранями)
                    target_idx = round(idx * (n_bg - 1) / (n_orig - 1))
                mapping_bg[orig_c] = sorted_bg[target_idx]
            return mapping_bg

        # Для главных объектов ('main'): используем accent_colors, отсортированные по яркости
        pool = list(self.accent_colors) if self.accent_colors else list(self.bg_colors)
        if not pool:
            return {}

        sorted_pool = sorted(pool, key=color_luminance)
        mapping: dict[str, str] = {}
        n_orig = len(sorted_orig)
        n_pool = len(sorted_pool)

        for idx, orig_c in enumerate(sorted_orig):
            if n_orig == 1:
                target_idx = min(n_pool - 1, max(0, n_pool // 2))
            elif idx < n_pool:
                target_idx = round(idx * (n_pool - 1) / (n_orig - 1)) if n_orig > 1 else 0
            else:
                target_idx = idx % n_pool
            mapping[orig_c] = sorted_pool[target_idx]

        return mapping

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "title": self.title,
            "bg_colors": list(self.bg_colors),
            "accent_colors": list(self.accent_colors),
            "canvas_bg": self.canvas_bg,
        }


# В каждой схеме bg_colors содержит 4 монохромные ступени одного тона:
# [1: тёмный контур, 2: падающая тень, 3: боковая грань, 4: светлая верхняя грань]
PRESET_PALETTES: dict[str, Palette] = {
    "lavender_mist": Palette(
        name="lavender_mist",
        title="Лавандовая схема",
        bg_colors=["#6d28d9", "#9370db", "#c4b5fd", "#ede9fe"],
        accent_colors=["#7c3aed", "#f59e0b", "#fde68a", "#f472b6"],
        canvas_bg="#dcd0ff",
    ),
    "peach_sunset": Palette(
        name="peach_sunset",
        title="Персиковая схема",
        bg_colors=["#c2410c", "#fb923c", "#fdba74", "#ffedd5"],
        accent_colors=["#ea580c", "#db2777", "#fde047", "#fb7185"],
        canvas_bg="#fed7aa",
    ),
    "mint_breeze": Palette(
        name="mint_breeze",
        title="Мятная схема",
        bg_colors=["#047857", "#34d399", "#6ee7b7", "#ecfdf5"],
        accent_colors=["#059669", "#0284c7", "#facc15", "#38bdf8"],
        canvas_bg="#d1fae5",
    ),
}


def register_palette(palette: Palette, save_path: str | Path = "palettes.json") -> None:
    PRESET_PALETTES[palette.name] = palette
    try:
        p = Path(save_path)
        existing: list[dict[str, Any]] = []
        if p.exists():
            existing = json.loads(p.read_text(encoding="utf-8"))
        existing = [item for item in existing if item.get("name") != palette.name]
        existing.append(palette.to_dict())
        p.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def load_custom_palettes(save_path: str | Path = "palettes.json") -> None:
    p = Path(save_path)
    if not p.exists():
        return
    try:
        items = json.loads(p.read_text(encoding="utf-8"))
        for it in items:
            pal = Palette(
                name=it["name"],
                title=it.get("title", it["name"]),
                bg_colors=it.get("bg_colors", []),
                accent_colors=it.get("accent_colors", []),
                canvas_bg=it.get("canvas_bg", "#ffffff"),
            )
            PRESET_PALETTES[pal.name] = pal
    except Exception:
        pass


def resolve_palette(spec: str | dict[str, Any] | Palette | None) -> Palette | None:
    if spec is None:
        return None
    if isinstance(spec, Palette):
        return spec
    if isinstance(spec, str):
        key = spec.strip().lower()
        if key in PRESET_PALETTES:
            return PRESET_PALETTES[key]
        for pal in PRESET_PALETTES.values():
            if pal.title.lower() == key:
                return pal
        raise ValueError(f"Неизвестная цветовая схема: {spec!r}. Доступны: {list(PRESET_PALETTES.keys())}")
    if isinstance(spec, dict):
        return Palette(
            name=spec.get("name", "custom"),
            title=spec.get("title", "Пользовательская схема"),
            bg_colors=spec.get("bg_colors", []),
            accent_colors=spec.get("accent_colors", []),
            canvas_bg=spec.get("canvas_bg", "#ffffff"),
        )
    raise ValueError(f"Некорректный формат палитры: {spec!r}")