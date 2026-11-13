"""Ядро AI-агента с чётким JPEG-сжатием рендеров (detail='high'), координатной линейкой для VLM,
жёстким бюджетом шагов (укладывается в 3-4 вызова инструментов) и пакетными комплексными диффами.
"""

from __future__ import annotations

import base64
import io
import json
import os
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.errors import GraphRecursionError
from langgraph.prebuilt import create_react_agent

from agent_tools import ToolRegistry
from renderer import CompositionRenderer
from scene_session import SceneSession
from svg_store import SVGStore


# Максимум правок сцены за один запрос пользователя (create_scene_from_scratch + apply_scene_diff)
DEFAULT_MAX_EDIT_STEPS = 6


class ContextMode(str, Enum):
    TEXT_ONLY = "text_only"
    MULTIMODAL = "multimodal"


def _compress_image_to_b64_jpeg(
    png_bytes: bytes,
    *,
    max_side: int = 768,
    quality: int = 86,
    bg_rgb: tuple[int, int, int] = (245, 243, 255),
    draw_ruler: bool = False,
) -> tuple[str, str]:
    """Сжимает PNG в чёткий JPEG (размером до max_side x max_side) для LLM.
    Если draw_ruler=True, наносит тонкую координатную сетку Холста 800x800 для точного глазомера VLM.
    Возвращает (mime_type, base64_str).
    """
    try:
        from PIL import Image, ImageDraw

        with Image.open(io.BytesIO(png_bytes)) as im:
            im = im.convert("RGBA")
            if max(im.width, im.height) > max_side:
                im.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
            background = Image.new("RGB", im.size, bg_rgb)
            background.paste(im, mask=im.split()[3])

            if draw_ruler:
                draw = ImageDraw.Draw(background)
                w, h = background.size
                for coord_800 in (200, 400, 600):
                    px = int(coord_800 / 800.0 * w)
                    py = int(coord_800 / 800.0 * h)
                    draw.line([(px, 0), (px, 12)], fill=(100, 116, 139), width=2)
                    draw.line([(0, py), (12, py)], fill=(100, 116, 139), width=2)
                    draw.text((px + 4, 2), str(coord_800), fill=(71, 85, 105))
                    draw.text((4, py + 2), str(coord_800), fill=(71, 85, 105))

            out = io.BytesIO()
            background.save(out, format="JPEG", quality=quality, optimize=True)
            return "image/jpeg", base64.b64encode(out.getvalue()).decode("utf-8")
    except Exception:
        return "image/png", base64.b64encode(png_bytes).decode("utf-8")


class AgentCore:
    """Оркестратор ReAct-агента для работы с SVG-композициями."""

    def __init__(
        self,
        store: SVGStore,
        session: SceneSession,
        renderer: CompositionRenderer,
        *,
        context_mode: ContextMode = ContextMode.TEXT_ONLY,
        library_mode: Literal["full_catalog", "rag"] = "full_catalog",
        rag_top_k: int = 10,
        visual_review: bool = True,
        model_name: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.0,
        max_edit_steps: int = DEFAULT_MAX_EDIT_STEPS,
    ) -> None:
        self.max_edit_steps = max(1, int(max_edit_steps))
        self.store = store
        self.session = session
        self.renderer = renderer
        self._context_mode = context_mode
        self.visual_review = visual_review

        self.model_name = model_name or os.getenv("LLM_MODEL") or "openai/gpt-4o-mini"
        self.api_key = api_key or os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("LLM_BASE_URL") or "https://api.proxyapi.ru/v1"

        if not self.api_key:
            raise ValueError("Не задан API-ключ для LLM (LLM_API_KEY или OPENAI_API_KEY).")

        self.llm = ChatOpenAI(
            model=self.model_name,
            api_key=self.api_key,
            base_url=self.base_url,
            max_retries=2,
        )

        self.tool_registry = ToolRegistry(
            session,
            store,
            renderer,
            library_mode=library_mode,
            rag_top_k=rag_top_k,
            multimodal_tools=(context_mode == ContextMode.MULTIMODAL),
        )
        self.tool_registry.reset_budget(self.max_edit_steps)
        self.tools = self.tool_registry.get_tools()
        self.agent = create_react_agent(self.llm, self.tools, prompt=self._build_step_messages)

    @property
    def context_mode(self) -> ContextMode:
        return self._context_mode

    @context_mode.setter
    def context_mode(self, value: ContextMode) -> None:
        self._context_mode = value
        self.tool_registry.multimodal_tools = value == ContextMode.MULTIMODAL

    def set_search_config(self, library_mode: Literal["full_catalog", "rag"], rag_top_k: int = 10) -> None:
        self.tool_registry.library_mode = library_mode
        self.tool_registry.rag_top_k = max(1, min(100, int(rag_top_k)))

    def _get_component_compressed_b64(self, img_obj: Any) -> tuple[str, str] | None:
        """Возвращает компактное превью компонента (256x256 JPEG)."""
        raw_bytes: bytes | None = None
        png_path = getattr(img_obj, "png_path", None) or Path("data/png") / f"{img_obj.id}.png"
        if Path(png_path).exists():
            try:
                raw_bytes = Path(png_path).read_bytes()
            except Exception:
                pass

        if raw_bytes is None:
            try:
                import resvg_py

                raw_bytes = bytes(resvg_py.svg_to_bytes(svg_string=img_obj.svg, width=256))
            except Exception:
                return None

        return _compress_image_to_b64_jpeg(raw_bytes, max_side=256, quality=80)

    def _build_step_messages(self, state: dict[str, Any]) -> list[BaseMessage]:
        """Формирует список сообщений только для ТЕКУЩЕГО шага LLM.
        Контролирует бюджет шагов (максимум 1 комплексная правка после сборки) и подкладывает рендер.
        """
        msgs: list[BaseMessage] = list(state.get("messages", []))
        if not msgs:
            return msgs

        last_msg = msgs[-1]
        if not isinstance(last_msg, ToolMessage):
            return msgs

        tool_name = getattr(last_msg, "name", "")

        # Считаем потраченные правки сцены (создание + диффы) и проверки в рамках этого запроса
        inspect_count = sum(
            1 for m in msgs if isinstance(m, ToolMessage) and getattr(m, "name", "") == "inspect_and_validate_scene"
        )
        edits_used = sum(
            1
            for m in msgs
            if isinstance(m, ToolMessage) and getattr(m, "name", "") in ("create_scene_from_scratch", "apply_scene_diff")
        )
        edits_left = max(0, self.max_edit_steps - edits_used)

        # Текстовый режим: картинок нет, но бюджет контролируем так же
        if self.context_mode != ContextMode.MULTIMODAL:
            if tool_name in ("inspect_and_validate_scene", "apply_scene_diff", "create_scene_from_scratch"):
                if edits_left <= 0:
                    msgs.append(
                        HumanMessage(
                            content=(
                                f"🛑 Лимит правок исчерпан ({edits_used}/{self.max_edit_steps}). НЕ вызывай больше инструменты — "
                                "сразу напиши краткий итог: что получилось и что осталось недоработанным."
                            )
                        )
                    )
                elif tool_name == "inspect_and_validate_scene":
                    msgs.append(
                        HumanMessage(
                            content=(
                                f"Осталось правок: {edits_left} из {self.max_edit_steps}. Если сцена в порядке — сразу ответь пользователю. "
                                "Если нужны исправления — отправь их ОДНИМ пакетом apply_scene_diff."
                            )
                        )
                    )
            return msgs

        # 1. После search_components показываем компактные превью (256px JPEG, detail='low') найденных компонентов
        if tool_name == "search_components" and self.session.candidates:
            content_blocks: list[dict[str, Any]] = [
                {
                    "type": "text",
                    "text": (
                        "Визуальные превью найденных компонентов. "
                        "Заранее продумай в уме всю композицию целиком (и для фонового слоя 'Background', и для главного 'Main'), "
                        "умножь исходные размеры на выбранный uniform_scale (группа главных объектов ~60–80% ширины холста), выбери 3–5 предметов с наложением и 1–2 фоновыми с обрезкой краем, и собери сцену ОДНИМ вызовом:"
                    ),
                }
            ]
            attached = 0
            for iid in self.session.candidates[:8]:
                try:
                    img_obj = self.store.get(iid)
                    comp_res = self._get_component_compressed_b64(img_obj)
                    if not comp_res:
                        continue
                    mime, b64_data = comp_res
                    content_blocks.append(
                        {
                            "type": "text",
                            "text": (
                                f"\n• image_id={img_obj.id!r} | {img_obj.name} "
                                f"| роль={img_obj.role} | реальный_размер={img_obj.width:.0f}x{img_obj.height:.0f} px"
                            ),
                        }
                    )
                    content_blocks.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime};base64,{b64_data}",
                                "detail": "low",
                            },
                        }
                    )
                    attached += 1
                except Exception:
                    continue

            if attached > 0:
                print(f"👁️ [VLM] Прикреплено {attached} превью компонентов из каталога в HumanMessage (detail='low')")
                msgs.append(HumanMessage(content=content_blocks))

        # 2. После inspect_and_validate_scene показываем рендер сцены в высоком разрешении (detail='high')
        elif tool_name == "inspect_and_validate_scene":
            try:
                png_bytes = self.renderer.render_composition_png(self.session.composition)
                mime, b64_scene = _compress_image_to_b64_jpeg(
                    png_bytes,
                    max_side=768,
                    quality=86,
                    draw_ruler=True,
                )
                kb_size = len(b64_scene) * 3 / 4 / 1024
                print(
                    f"👁️ [VLM] Рендер сцены прикреплён ({kb_size:.1f} KB, detail='high', проверка #{inspect_count}, правок={edits_used}/{self.max_edit_steps})"
                )

                if edits_left <= 0:
                    instruction_text = (
                        f"🛑 ФИНАЛЬНЫЙ РЕНДЕР СЦЕНЫ. Лимит правок исчерпан ({edits_used}/{self.max_edit_steps}). "
                        "НЕ вызывай больше инструменты! Сразу напиши краткий итог по этой картинке: что получилось, "
                        "и честно отметь, что осталось недоработанным."
                    )
                else:
                    instruction_text = (
                        "На изображении ниже — актуальный рендер твоей сцены (с засечками 200, 400, 600 px).\n"
                        f"Осталось правок: {edits_left} из {self.max_edit_steps}.\n"
                        "1) Если сцена выглядит гармонично, стоячие предметы не стоят поверх книги, тени не слипаются, "
                        "холст не пустой — НЕ вызывай инструменты, сразу отвечай пользователю.\n"
                        "2) Если есть реальные дефекты — заранее просчитай новые координаты для ВСЕХ проблемных объектов "
                        "и отправь ОДИН пакет `apply_scene_diff` (все сдвиги, добавления и смены ролей сразу)."
                    )

                msgs.append(
                    HumanMessage(
                        content=[
                            {"type": "text", "text": instruction_text},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime};base64,{b64_scene}",
                                    "detail": "high",
                                },
                            },
                        ]
                    )
                )
            except Exception as exc:
                print(f"⚠️ Не удалось отрендерить PNG для VLM-проверки: {exc}")

        return msgs

    def _get_system_prompt(self) -> str:
        static_role = (
            "Ты — арт-директор и специалист по изометрическим векторным композициям в системе Antigravity.\n\n"
            "1. ЖЁСТКИЕ ПРАВИЛА ТЗ (НЕЛЬЗЯ НАРУШАТЬ):\n"
            "- Размер холста фиксирован (800×800): НЕ пытайся менять размер холста. Размещай и двигай сами объекты внутри 800×800.\n"
            "- Уровень детализации (L и S): используй только уровень детализации текущей сцены ('L' или 'S').\n"
            "- Единые пропорции объектов (True Isometric Scale): НЕЛЬЗЯ менять масштаб отдельных объектов относительно друг друга "
            "(в изометрии нет перспективного уменьшения вдали). Все объекты масштабируются синхронно через единый `uniform_scale`.\n"
            "- Роли объектов: Главные ('main' — цветные, в слое 'Main') и Фоновые ('background' — монохромные, в слое 'Background'). "
            "Любой объект на холсте может переключать состояние `main` <-> `background` через `toggle_item_role`.\n"
            "- Цветовые схемы: 'lavender_mist' (Лавандовая), 'peach_sunset' (Персиковая), 'mint_breeze' (Мятная). "
            "Цвета фонов (bg_colors) МОГУТ использоваться в главных объектах, но акцентные цвета главных объектов "
            "(accent_colors) СТРОГО ЗАПРЕЩЕНЫ в фонах и фоновых объектах.\n\n"
            "2. СТИЛЬ И КОМПОЗИЦИЯ (ориентир — референсы из ТЗ). Референс — это плотный крупный кадр с историей, а НЕ каталог иконок в ячейках:\n"
            "- Мало объектов: 3–5 в сумме (1 главный герой + 1–2 поддерживающих + 1–2 фоновых). Не выкладывай всё, что есть в каталоге: каждый предмет должен работать на запрос. Не дублируй предметы (две одинаковые ручки и т.п.) без причины.\n"
            "- Крупный масштаб: подбирай `uniform_scale` по реальным размерам ассетов так, чтобы группа главных объектов занимала примерно 60–80% ширины холста. Лучше чуть крупнее и с обрезкой краем, чем мелко и с воздухом по всему периметру.\n"
            "- Одна компактная группа вокруг смыслового центра (центр или нижняя/боковая часть кадра): предметы стоят БЛИЗКО друг к другу. Пустой угол — нормально, а разбросанные по всему холсту предметы — нет. Сетку и параметр `cell` НЕ используй (по одному предмету в ячейку выходит «лист иконок»): задавай координаты `x, y`.\n"
            "- Наложение — это норма и признак стиля: мелкие плоские предметы (телефон, кошелёк, пазл, ручка, монета) лежат ПОВЕРХ крупных плоских (книга, папка) и частично их перекрывают; такие предметы идут ПОЗЖЕ базы в списке `placements` слоя `Main`.\n"
            "- Вертикальные предметы (лампа, ваза, цветы) стоят на столе РЯДОМ с книгой/папкой, вплотную, но никогда не на раскрытой книге и не друг на друге.\n"
            "- Фоновые объекты (`role='background'`, монохромные): 1–2 крупных, в слое `Background`, позади главных. Им положено частично выходить за край холста (должно оставаться хотя бы ~35% площади в кадре) — так появляется глубина. Если по теме запроса есть фоновые версии предметов, используй их (ищи без фильтра `role`).\n"
            "- Обрезка краями: приветствуется для фоновых и крупных базовых предметов (книга, папка, лампа-основание могут уходить за край). Главный герой остаётся целиком читаемым.\n"
            "- Единый свет: чёрные тени у всех предметов падают в одну сторону; лампы и направленные предметы «смотрят» внутрь сцены.\n"
            "- Баланс: если тяжёлая яркая группа с одной стороны, уравновешивай её лёгким фоновым силуэтом с другой.\n"
            "- Ловушка касания (Tangent Trap): избегай «почти касания» рамка в рамку у стоящих рядом предметов и их теней — либо уверенное наложение, либо явный зазор.\n""\n"
            "Старайся делать сцены логичными допустим ставь лампу выше обьекта который она должна освещать и тд\n""\n"
            "Старайся делать сцены плотными что бы сцена не выглдяла как набор разразненных обьектов\n""\n"
            f"3. РЕГЛАМЕНТ РАБОТЫ (БЮДЖЕТ: максимум {self.max_edit_steps} правок сцены за запрос):\n"
            f"Правкой считается каждый вызов `create_scene_from_scratch` или `apply_scene_diff`. Обычно хватает 3–4 шагов, "
            f"но если рендер показывает реальный брак, можно исправлять до {self.max_edit_steps} правок суммарно. "
            "Когда лимит исчерпан или сцена уже хороша — НЕ вызывай инструменты и напиши итог по текущему результату. Алгоритм:\n"
            "- ШАГ 1 (`search_components`): Найди нужные компоненты (если нужны и главные, и фоновые — не указывай фильтр `role`, чтобы получить весь список за 1 вызов!).\n"
            "- ШАГ 2 (`create_scene_from_scratch` или комплексный `apply_scene_diff`): В уме тщательно рассчитай `uniform_scale` и координаты `(x, y)` сразу для ВСЕХ объектов сцены (и фоновых в слое `Background`, и главных в слое `Main`), чтобы они с первой попытки встали без наложений.\n"
            "- ШАГ 3 (`inspect_and_validate_scene`): проверь визуальный рендер сцены.\n"
            "- ШАГ 4 (ТОЛЬКО если на рендере есть реальный брак): собери ВСЕ нужные исправления (сдвиги нескольких предметов, смену ролей, удаление или добавление) в ОДИН пакет `operations` одного вызова `apply_scene_diff`, затем снова `inspect_and_validate_scene`.\n"
            "- Повторяй ШАГ 3–4, только пока остаются правки и ты точно знаешь, как улучшить сцену. Стремись к хорошему результату за минимум правок: вноси в каждую правку все нужные изменения сразу. Поводы для правки: сцена выглядит как разреженный каталог иконок; объекты мелкие или далеко друг от друга; нет глубины (нет фоновых объектов/обрезки); наложение вопреки физике; главный герой обрезан. ВСЕГДА заканчивай ответом пользователю: даже если не всё идеально, опиши итоговую сцену и что не удалось довести.\n\n"
        )
        return static_role + self.tool_registry.get_tools_documentation()

    def _prepare_multimodal_content(self, query: str) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = [{"type": "text", "text": query}]

        if self.visual_review and self.session.composition.items:
            try:
                png_bytes = self.renderer.render_composition_png(self.session.composition)
                mime, b64_scene = _compress_image_to_b64_jpeg(
                    png_bytes,
                    max_side=768,
                    quality=86,
                    draw_ruler=True,
                )
                content.append({"type": "text", "text": "\nИсходное состояние сцены (рендер):"})
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime};base64,{b64_scene}",
                            "detail": "high",
                        },
                    }
                )
            except Exception as exc:
                print(f"⚠️ Не удалось отрендерить стартовый PNG: {exc}")

        return content

    def generate_composition(self, user_request: str) -> str:
        return self._run_agent(user_request, is_editing=False)

    def edit_composition(self, user_request: str) -> str:
        return self._run_agent(user_request, is_editing=True)

    # ---------- Извлечение текста и гарантированный финальный ответ ----------

    @staticmethod
    def _extract_text(content: Any) -> str:
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict) and block.get("type") == "text":
                    parts.append(str(block.get("text", "")))
            return "\n".join(p for p in parts if p).strip()
        return ""

    def _scene_brief(self) -> tuple[str, bool, int]:
        """Короткая текстовая сводка о том, что реально лежит на холсте. Возвращает (текст, валидна_ли, число_объектов)."""
        summary = self.session.get_scene_summary()
        names: list[str] = []
        for layer in summary["layers"]:
            for it in layer["items"]:
                names.append(f"{it['image_name']} [{layer['name']}]")
        report = self.session.composition.validate(self.session.get_image)
        color_errors = self.session._validate_color_rules(self.session.composition, self.session.active_palette)
        ok = report.ok and not color_errors
        brief = (
            f"Шаг истории #{summary['step_id']}, размер {summary['detail_level']}, "
            f"единый масштаб {summary['uniform_scale']}, объектов: {len(names)}"
            + (f" ({'; '.join(names)})" if names else "")
            + f". Валидация правил ТЗ: {'пройдена' if ok else 'есть замечания'}."
        )
        if not ok:
            brief += "\n" + self.session.format_validation_errors(self.session.composition, report, color_errors)
        return brief, ok, len(names)

    def _final_answer_fallback(self, user_request: str, reason: str) -> str:
        """Всегда возвращает НЕпустой ответ по тому состоянию сцены, которое уже есть в сессии.
        Сначала пытается получить короткий итог у LLM (без инструментов), при неудаче собирает ответ сам.
        """
        used = self.tool_registry.edits_used
        limit = self.max_edit_steps
        try:
            brief, ok, n_items = self._scene_brief()
        except Exception as exc:  # сводка не должна ронять ответ
            return f"Работа остановлена ({reason}). Сцена сохранена в последнем состоянии. (Сводка недоступна: {exc})"

        if n_items == 0:
            return (
                f"Не удалось собрать сцену: {reason}. Правок использовано {used}/{limit}. "
                "Попробуй переформулировать запрос или уточнить, какие объекты нужны."
            )

        try:
            resp = self.llm.invoke(
                [
                    SystemMessage(
                        content=(
                            "Ты — арт-директор. Напиши пользователю 2–4 предложения по-русски: что собрано на сцене и что "
                            "не удалось довести. Не вызывай инструменты, не придумывай объекты, которых нет в сводке."
                        )
                    ),
                    HumanMessage(
                        content=(
                            f"Запрос пользователя: {user_request}\n"
                            f"Причина остановки: {reason}. Правок использовано {used}/{limit}.\n"
                            f"Сводка сцены: {brief}"
                        )
                    ),
                ]
            )
            text = self._extract_text(resp.content)
            if text:
                return text
        except Exception as exc:
            print(f"⚠️ Итоговый ответ от LLM не получен: {exc}")

        tail = "" if ok else " Остались замечания по правилам ТЗ — их можно поправить следующим запросом."
        return f"Готово в пределах лимита ({used}/{limit} правок). {brief}{tail}"

    def _run_agent(self, user_request: str, *, is_editing: bool = False) -> str:
        self.tool_registry.reset_budget(self.max_edit_steps)

        scene_summary = self.session.get_scene_summary()
        scene_json = json.dumps(scene_summary, ensure_ascii=False, indent=2)

        messages: list[Any] = [SystemMessage(content=self._get_system_prompt())]
        prompt_text = (
            f"Текущее состояние сцены (JSON):\n{scene_json}\n\n"
            f"Запрос пользователя: {user_request}\n"
            f"(Напоминание: продумай все координаты заранее; бюджет — не более {self.max_edit_steps} правок сцены; "
            f"объединяй исправления в один пакет apply_scene_diff, а в конце обязательно дай текстовый итог.)"
        )

        if self.context_mode == ContextMode.MULTIMODAL:
            messages.append(HumanMessage(content=self._prepare_multimodal_content(prompt_text)))
        else:
            messages.append(HumanMessage(content=prompt_text))

        # Каждый вызов инструмента = 2 шага графа (модель + инструмент). Лимит даём с запасом под бюджет правок
        # (правки + проверки + поиск + финальный ответ), а реальное ограничение держит бюджет в самих инструментах.
        recursion_limit = 4 * self.max_edit_steps + 16

        def _invoke(msgs: list[Any]) -> dict[str, Any]:
            return self.agent.invoke({"messages": msgs}, config={"recursion_limit": recursion_limit})

        def _final_text(res: dict[str, Any]) -> tuple[str, Any]:
            """Возвращает (текст_ответа, последнее_сообщение). Пустая строка — ответа нет."""
            last = res["messages"][-1] if res.get("messages") else None
            # Ответом считаем только финальное сообщение модели без вызова инструментов; текст ToolMessage ответом не является
            is_final_ai = isinstance(last, AIMessage) and not getattr(last, "tool_calls", None)
            text = self._extract_text(last.content) if is_final_ai else ""
            # LangGraph при нехватке шагов сам подставляет служебную фразу вместо ответа — это тоже не результат
            if text.lower().startswith("sorry, need more steps"):
                return "", last
            return text, last

        def _debug_empty(last: Any) -> None:
            """Печатает в консоль, ПОЧЕМУ модель ничего не ответила (finish_reason, формат вызова инструмента и т.д.)."""
            try:
                print(
                    "🔎 [DEBUG] Пустой ответ модели: "
                    f"model={self.model_name!r}, mode={self.context_mode.value}, "
                    f"msg_type={type(last).__name__}, content={getattr(last, 'content', None)!r:.200}, "
                    f"tool_calls={getattr(last, 'tool_calls', None)}, "
                    f"invalid_tool_calls={getattr(last, 'invalid_tool_calls', None)}, "
                    f"additional_kwargs_keys={list((getattr(last, 'additional_kwargs', None) or {}).keys())}, "
                    f"response_metadata={getattr(last, 'response_metadata', None)}"
                )
            except Exception as exc:
                print(f"🔎 [DEBUG] Не удалось вывести диагностику: {exc}")

        try:
            result = _invoke(messages)
            text, last = _final_text(result)

            # Один автоматический повтор: модель иногда молчит после сообщения с картинками — подталкиваем её
            if not text:
                _debug_empty(last)
                if self.tool_registry.edits_used == 0:
                    nudge = (
                        "Ты не вызвал инструмент и не дал ответа. Сейчас же вызови `create_scene_from_scratch` "
                        "со всеми нужными объектами сцены (используй image_id из списка компонентов выше), без пояснений."
                    )
                else:
                    nudge = "Ты не дал ответа. Напиши краткий итог пользователю: что собрано на сцене и что не удалось довести."
                print("🔁 [RETRY] Повторный запрос к модели с подсказкой")
                result = _invoke(list(result["messages"]) + [HumanMessage(content=nudge)])
                text, last = _final_text(result)
                if not text:
                    _debug_empty(last)
        except GraphRecursionError:
            # Дошли до технического лимита графа: сцена в сессии уже содержит всё, что успели сделать — отдаём её итог
            return self._final_answer_fallback(user_request, "достигнут технический лимит шагов агента")
        except Exception as e:
            detail = getattr(e, "body", None) or getattr(e, "response", None)
            msg = f"Ошибка во время выполнения агента: {e}" + (f" | Детали: {detail}" if detail else "")
            raise RuntimeError(msg) from e

        if not text:
            return self._final_answer_fallback(user_request, "модель завершила работу без текстового ответа")
        return text