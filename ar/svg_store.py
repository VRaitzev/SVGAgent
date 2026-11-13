"""Хранилище SVG-иллюстраций поверх LanceDB с поддержкой эмбеддингов и семантического поиска."""

from __future__ import annotations

import hashlib
import os
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import lancedb
import pyarrow as pa
import resvg_py

from ar.svg_image import SVG_NS, SVGImage, compute_hash

def get_schema(dim: int | None = None) -> pa.Schema:
    fields = [
        pa.field("id", pa.string(), nullable=False),
        pa.field("svg", pa.string(), nullable=False),
        pa.field("hash", pa.string(), nullable=False),
        pa.field("name", pa.string()),
        pa.field("source_path", pa.string()),
        pa.field("png_path", pa.string()),
        pa.field("description", pa.string()),
    ]
    if dim is not None and dim > 0:
        fields.append(pa.field("vector", pa.list_(pa.float32(), dim)))
    return pa.schema(fields)

class SVGStoreError(Exception):
    pass
class InvalidSVGError(SVGStoreError):
    pass
class DuplicateSVGError(SVGStoreError):
    def __init__(self, existing_id: str):
        super().__init__(f"SVG уже существует, id={existing_id}")
        self.existing_id = existing_id
class SVGNotFoundError(SVGStoreError):
    pass

def _quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"

def _validate_svg(svg: str) -> None:
    try:
        root = ET.fromstring(svg)
    except ET.ParseError as e:
        raise InvalidSVGError(f"Некорректный XML: {e}") from e
    if root.tag not in ("svg", f"{{{SVG_NS}}}svg"):
        raise InvalidSVGError(f"Корневой элемент должен быть <svg>, а не <{root.tag}>")


# ИСПРАВЛЕНИЕ: Вынесли дефолтный маркер на уровень модуля, чтобы избежать бага с object()
_UNSET = object()


class SVGStore:
    def __init__(
        self,
        db_path: str | Path = "data/lancedb",
        table_name: str = "svgs",
        png_dir: str | Path = "data/png",
        png_width: int = 512,
        *,
        embedding_api_key: str | None = None,
        embedding_base_url: str | None = None,
        embedding_model: str | None = None,
        embedding_dim: int | None = None,
    ):
        self.db_path = Path(db_path)
        self.table_name = table_name
        self.png_dir = Path(png_dir)
        self.png_width = png_width

        self.api_key = embedding_api_key or os.getenv("EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.base_url = embedding_base_url or os.getenv("EMBEDDING_BASE_URL") or "https://api.openai.com/v1"
        self.model_name = embedding_model or os.getenv("EMBEDDING_MODEL") or "text-embedding-3-small"
        
        if embedding_dim is not None:
            self.dim = embedding_dim
        elif "text-embedding-3-small" in self.model_name:
            self.dim = 1536
        elif "text-embedding-3-large" in self.model_name:
            self.dim = 3072
        elif "jina-clip" in self.model_name:
            self.dim = 768
        else:
            self.dim = None 

        self._db = lancedb.connect(str(self.db_path))
        schema = get_schema(self.dim)
        
        if self.table_name in self._db.table_names():
            self._table = self._db.open_table(self.table_name)
        else:
            self._table = self._db.create_table(self.table_name, schema=schema, mode="create")

    def _get_embedding(self, text: str) -> list[float] | None:
        if not self.api_key or not self.dim:
            return None
        try:
            import openai
            client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)
            response = client.embeddings.create(input=[text], model=self.model_name)
            return list(response.data[0].embedding)
        except Exception:
            return None

    @staticmethod
    def _read_source(source: str | Path) -> tuple[str, Path | None]:
        if isinstance(source, Path):
            path = source
        elif isinstance(source, str):
            if source.lstrip(" \t\r\n").startswith("<"):
                return source, None
            path = Path(source)
        else:
            raise TypeError("Ожидается str (текст SVG или путь) либо pathlib.Path")

        if not path.is_file():
            raise FileNotFoundError(f"Файл не найден: {path}")
        return path.read_text(encoding="utf-8-sig"), path

    def _auto_png_path(self, svg_id: str) -> Path:
        return self.png_dir / f"{svg_id}.png"

    def _render_png(self, svg: str, svg_id: str) -> str:
        try:
            data = resvg_py.svg_to_bytes(svg_string=svg, width=self.png_width)
        except ValueError as e:
            raise InvalidSVGError(f"Не удалось отрисовать SVG: {e}") from e
        target = self._auto_png_path(svg_id)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        return str(target)

    def _is_auto_png(self, png_path: str | None, svg_id: str) -> bool:
        return png_path is not None and Path(png_path) == self._auto_png_path(svg_id)

    def add(
        self,
        source: str | Path,
        *,
        name: str | None = None,
        description: str | None = None,
        png_path: str | Path | None = None,
    ) -> SVGImage:
        svg, path = self._read_source(source)
        _validate_svg(svg)

        digest = compute_hash(svg)
        existing = self.get_by_hash(digest)
        if existing:
            raise DuplicateSVGError(existing.id)

        if png_path is not None and not Path(png_path).is_file():
            raise FileNotFoundError(f"PNG не найден: {png_path}")

        svg_id = uuid.uuid4().hex
        png = str(png_path) if png_path is not None else self._render_png(svg, svg_id)

        resolved_name = name if name is not None else (path.stem if path else None)
        resolved_desc = description or resolved_name or "SVG illustration"
        
        vector = self._get_embedding(f"Name: {resolved_name}. Description: {resolved_desc}")

        image = SVGImage(
            id=svg_id, svg=svg, hash=digest, name=resolved_name,
            source_path=str(path) if path else None, png_path=png, description=description,
        )

        row = self._to_row(image, vector=vector)
        self._table.add([row])
        return image

    def get(self, svg_id: str) -> SVGImage:
        rows = self._table.search().where(f"id = {_quote(svg_id)}").limit(1).to_list()
        if not rows:
            raise SVGNotFoundError(f"Нет записи с id={svg_id}")
        return self._to_image(rows[0])

    def get_by_hash(self, digest: str) -> SVGImage | None:
        rows = self._table.search().where(f"hash = {_quote(digest)}").limit(1).to_list()
        return self._to_image(rows[0]) if rows else None

    def list(self, limit: int | None = None, offset: int = 0) -> list[SVGImage]:
        query = self._table.search()
        if offset:
            query = query.offset(offset)
        query = query.limit(limit if limit is not None else max(self.count(), 1))
        return [self._to_image(r) for r in query.to_list()]

    def exists(self, svg_id: str) -> bool:
        return self._table.count_rows(filter=f"id = {_quote(svg_id)}") > 0

    def count(self) -> int:
        return self._table.count_rows()

    def search(self, query: str, limit: int = 6) -> list[SVGImage]:
        if self.dim and self.api_key:
            vec = self._get_embedding(query)
            if vec is not None:
                try:
                    rows = self._table.search(vec).limit(limit).to_list()
                    return [self._to_image(r) for r in rows]
                except Exception:
                    pass

        q_lower = query.lower()
        all_rows = self._table.search().limit(1000).to_list()
        matched = []
        for r in all_rows:
            name = (r.get("name") or "").lower()
            desc = (r.get("description") or "").lower()
            if q_lower in name or q_lower in desc or any(w in name or w in desc for w in q_lower.split()):
                matched.append(self._to_image(r))
        
        if matched:
            return matched[:limit]
        
        return [self._to_image(r) for r in all_rows[:limit]]

    def update(self, svg_id: str, source: str | Path) -> SVGImage:
        old = self.get(svg_id)
        svg, path = self._read_source(source)
        _validate_svg(svg)

        digest = compute_hash(svg)
        other = self.get_by_hash(digest)
        if other and other.id != svg_id:
            raise DuplicateSVGError(other.id)

        png = old.png_path
        if png is None or self._is_auto_png(png, svg_id):
            png = self._render_png(svg, svg_id)

        vector = self._get_embedding(f"Name: {old.name}. Description: {old.description or ''}")

        values: dict[str, Any] = {
            "svg": svg, "hash": digest, "source_path": str(path) if path else None, "png_path": png,
        }
        if vector is not None and self.dim:
            values["vector"] = vector

        self._apply(svg_id, values)
        return self.get(svg_id)

    def update_metadata(
        self,
        svg_id: str,
        *,
        name: Any = _UNSET,
        description: Any = _UNSET,
        png_path: Any = _UNSET,
    ) -> SVGImage:
        image = self.get(svg_id)
        values: dict[str, Any] = {}
        
        new_name = name if name is not _UNSET else image.name
        new_desc = description if description is not _UNSET else image.description

        if name is not _UNSET:
            values["name"] = name
        if description is not _UNSET:
            values["description"] = description
        if png_path is not _UNSET:
            if png_path is not None and not Path(png_path).is_file():
                raise FileNotFoundError(f"PNG не найден: {png_path}")
            values["png_path"] = None if png_path is None else str(png_path)

        if (name is not _UNSET or description is not _UNSET) and self.dim:
            vector = self._get_embedding(f"Name: {new_name}. Description: {new_desc or ''}")
            if vector is not None:
                values["vector"] = vector

        if values:
            self._apply(svg_id, values)
        return self.get(svg_id)

    def delete(self, svg_id: str) -> None:
        image = self.get(svg_id)
        self._table.delete(f"id = {_quote(svg_id)}")
        if self._is_auto_png(image.png_path, svg_id):
            Path(image.png_path).unlink(missing_ok=True)

    def _apply(self, svg_id: str, values: dict[str, Any]) -> None:
        # ИСПРАВЛЕНИЕ: LanceDB корректно съедает словарь values напрямую
        if values:
            self._table.update(where=f"id = {_quote(svg_id)}", values=values)

    @staticmethod
    def _to_row(image: SVGImage, vector: list[float] | None = None) -> dict[str, Any]:
        row = {
            "id": image.id, "svg": image.svg, "hash": image.hash, "name": image.name,
            "source_path": image.source_path, "png_path": image.png_path, "description": image.description,
        }
        if vector is not None:
            row["vector"] = vector
        return row

    @staticmethod
    def _to_image(row: dict[str, Any]) -> SVGImage:
        return SVGImage(
            id=row["id"], svg=row["svg"], hash=row["hash"], name=row.get("name"),
            source_path=row.get("source_path"), png_path=row.get("png_path"), description=row.get("description"),
        )