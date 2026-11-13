"""Скрипт пакетного наполнения базы LanceDB вашими SVG-иллюстрациями из Figma.

Автоматически определяет по структуре папок или именам файлов:
- уровень детализации: 'L' или 'S' (например, папки 'Размер L', 'Size_S', суффиксы '_L', '_S');
- роль объекта: 'main' (Главные объекты) или 'background' (Фоновые монохромные объекты);
- общий concept_key (например, 'book', 'headphones', 'lamp'), связывающий пару L и S для мгновенной автоподмены в конструкторе.

Примеры запуска (из папки backend/):
  uv run python ingest_svgs.py ./assets/figma_export --reset --describe
  uv run python ingest_svgs.py ./assets/size_l --detail-level L --role main
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
from pathlib import Path

import resvg_py
from dotenv import load_dotenv

from svg_image import DetailLevel, ObjectRole
from svg_store import DuplicateSVGError, SVGStore, normalize_concept_key

load_dotenv()


def infer_metadata_from_path(svg_path: Path, default_level: str | None, default_role: str | None) -> tuple[DetailLevel, ObjectRole]:
    """Определяет размер (L/S) и роль (main/background) по пути папок и имени файла."""
    full_str = " ".join([p.lower() for p in svg_path.parts])

    if default_level:
        level: DetailLevel = "S" if default_level.upper() == "S" else "L"
    elif any(tok in full_str for tok in ("размер s", "size s", "size_s", "_s.", "-s.")):
        level = "S"
    else:
        level = "L"

    if default_role:
        role: ObjectRole = "background" if default_role.lower() == "background" else "main"
    elif any(tok in full_str for tok in ("фоновые", "фон", "background", "_bg", "-bg")):
        role = "background"
    else:
        role = "main"

    return level, role


def describe_svg_with_vlm(svg_text: str, filename: str, role: ObjectRole, level: DetailLevel) -> tuple[str, str]:
    """Генерирует название и метафоры по PNG-рендеру через Vision-LLM."""
    api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("LLM_BASE_URL") or "https://api.proxyapi.ru/v1"
    model = os.getenv("LLM_MODEL") or "deepseek/deepseek-v4.1-flash-1017"

    if not api_key:
        return filename, filename

    try:
        import openai

        png_bytes = bytes(resvg_py.svg_to_bytes(svg_string=svg_text, width=384))
        b64 = base64.b64encode(png_bytes).decode("utf-8")

        client = openai.OpenAI(api_key=api_key, base_url=base_url)
        prompt = (
            f"Ты каталогер векторных иллюстраций (роль={role}, размер={level}, файл={filename!r}). "
            "Посмотри на изображение и верни строго JSON с двумя полями:\n"
            '{"name": "Короткое понятное название на русском (2-3 слова)", '
            '"description": "Описание объекта и 6-8 ключевых слов/метафор через запятую на русском и английском"}\n'
            "Без markdown-обёртки, только чистый JSON."
        )

        resp = client.chat.completions.create(
            model=model,
            temperature=0.2,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    ],
                }
            ],
        )
        raw = (resp.choices[0].message.content or "").strip()
        if raw.startswith("```"):
            raw = raw.strip("`").removeprefix("json").strip()
        data = json.loads(raw)
        name = str(data.get("name") or filename).strip()
        desc = str(data.get("description") or name).strip()
        return name, desc
    except Exception as exc:
        print(f"   ⚠️ Не удалось сгенерировать VLM-описание для {filename}: {exc}")
        return filename, filename


def main() -> None:
    parser = argparse.ArgumentParser(description="Пакетная загрузка SVG из Figma в базу LanceDB.")
    parser.add_argument("folder", type=str, help="Путь к папке с .svg файлами")
    parser.add_argument("--db", type=str, default="data/lancedb", help="Путь к базе LanceDB")
    parser.add_argument("--png-dir", type=str, default="data/png", help="Папка для PNG-превью")
    parser.add_argument("--detail-level", choices=["L", "S"], default=None, help="Принудительно задать размер (L или S)")
    parser.add_argument("--role", choices=["main", "background"], default=None, help="Принудительно задать роль")
    parser.add_argument("--describe", action="store_true", help="Генерировать названия и метафоры через Vision-LLM")
    parser.add_argument("--reset", action="store_true", help="Очистить текущую базу перед загрузкой")
    args = parser.parse_args()

    src_dir = Path(args.folder)
    if not src_dir.is_dir():
        raise SystemExit(f"❌ Папка не найдена: {src_dir}")

    if args.reset:
        print(f"🧹 Очистка базы {args.db} и превью {args.png_dir}...")
        shutil.rmtree(args.db, ignore_errors=True)
        shutil.rmtree(args.png_dir, ignore_errors=True)

    store = SVGStore(db_path=args.db, png_dir=args.png_dir)
    print(
        f"📦 Подключено хранилище (модель={store.model_name}, dim={store.dim}, multimodal={store.is_multimodal})"
    )

    svg_files = sorted(src_dir.rglob("*.svg"))
    if not svg_files:
        print(f"⚠️ В папке {src_dir} не найдено ни одного .svg файла.")
        return

    added, skipped, failed = 0, 0, 0
    for idx, svg_path in enumerate(svg_files, start=1):
        level, role = infer_metadata_from_path(svg_path, args.detail_level, args.role)
        concept = normalize_concept_key(svg_path.stem)
        stem_name = svg_path.stem.replace("_", " ").replace("-", " ").strip()

        try:
            svg_text = svg_path.read_text(encoding="utf-8-sig")
            if args.describe:
                name, desc = describe_svg_with_vlm(svg_text, stem_name, role, level)
            else:
                name, desc = stem_name, stem_name

            img = store.add(
                svg_path,
                name=name,
                description=desc,
                detail_level=level,
                role=role,
                concept_key=concept,
            )
            added += 1
            print(
                f"[{idx}/{len(svg_files)}] ✅ {img.name!r} | size={img.detail_level} | role={img.role} | concept={img.concept_key}"
            )
        except DuplicateSVGError as dup:
            skipped += 1
            print(f"[{idx}/{len(svg_files)}] ⏭️ Дубликат: {svg_path.name} (id={dup.existing_id[:8]}...)")
        except Exception as exc:
            failed += 1
            print(f"[{idx}/{len(svg_files)}] ❌ Ошибка ({svg_path.name}): {exc}")

    print(f"\n🎉 Готово! Добавлено: {added} | Дубликатов: {skipped} | Ошибок: {failed}")
    print(f"📊 Всего иллюстраций в базе: {store.count()}")


if __name__ == "__main__":
    main()
