"""Хранилище SVG-иллюстраций поверх LanceDB с поддержкой:
- уровней детализации L и S (detail_level: 'L' | 'S') и связи парных вариантов (concept_key);
- ролей объектов ('main' — главные цветные, 'background' — фоновые монохромные);
- мультимодальных эмбеддингов (по PNG-превью) и гибридного поиска.
"""

from __future__ import annotations

import base64
import os
import re
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import lancedb
import pyarrow as pa
import resvg_py


from svg_image import SVG_NS, DetailLevel, ObjectRole, SVGImage, compute_hash, derive_concept_key

def get_schema(dim: int | None = None) -> pa.Schema:
    fields = [
        pa.field("id", pa.string(), nullable=False),
        pa.field("svg", pa.string(), nullable=False),
        pa.field("hash", pa.string(), nullable=False),
        pa.field("name", pa.string()),
        pa.field("source_path", pa.string()),
        pa.field("png_path", pa.string()),
        pa.field("description", pa.string()),
        pa.field("detail_level", pa.string()),
        pa.field("role", pa.string()),
        pa.field("concept_key", pa.string()),
    ]
    if dim is not None and dim > 0:
        fields.append(pa.field("vector", pa.list_(pa.float32(), dim)))
    return pa.schema(fields)


class SVGStoreError(Exception):
    """Базовая ошибка хранилища."""


class InvalidSVGError(SVGStoreError):
    """Переданный текст не является корректным SVG."""


class DuplicateSVGError(SVGStoreError):
    """SVG с таким содержимым уже есть в хранилище."""

    def __init__(self, existing_id: str):
        super().__init__(f"SVG уже существует, id={existing_id}")
        self.existing_id = existing_id


class SVGNotFoundError(SVGStoreError):
    """Записи с таким id нет."""


def _quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _validate_svg(svg: str) -> None:
    try:
        root = ET.fromstring(svg)
    except ET.ParseError as e:
        raise InvalidSVGError(f"Некорректный XML: {e}") from e
    if root.tag not in ("svg", f"{{{SVG_NS}}}svg"):
        raise InvalidSVGError(f"Корневой элемент должен быть <svg>, а не <{root.tag}>")


def normalize_concept_key(raw: str | None) -> str:
    return derive_concept_key(raw)


_UNSET = object()


class SVGStore:
    """CRUD-обёртка над таблицей LanceDB с поддержкой размеров L/S, ролей и мультимодальных эмбеддингов."""

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
        multimodal_embeddings: bool | None = None,
    ):
        self.db_path = Path(db_path)
        self.table_name = table_name
        self.png_dir = Path(png_dir)
        self.png_width = png_width

        self.api_key = embedding_api_key or os.getenv("EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.base_url = embedding_base_url or os.getenv("EMBEDDING_BASE_URL") or "https://api.proxyapi.ru/v1"
        self.model_name = embedding_model or os.getenv("EMBEDDING_MODEL") or "cohere/embed-5-fast"

        env_dim = os.getenv("EMBEDDING_DIM")
        if embedding_dim is not None:
            self.dim = embedding_dim
        elif env_dim and env_dim.isdigit():
            self.dim = int(env_dim)
        elif not self.api_key:
            self.dim = None
        elif "embed-5" in self.model_name or "text-embedding-3-small" in self.model_name:
            self.dim = 1536
        elif "titan-embed-image" in self.model_name:
            self.dim = 1024
        elif "text-embedding-3-large" in self.model_name:
            self.dim = 3072
        elif "jina-clip" in self.model_name:
            self.dim = 768
        else:
            self.dim = None

        if multimodal_embeddings is not None:
            self.is_multimodal = multimodal_embeddings
        else:
            env_mm = os.getenv("EMBEDDING_MULTIMODAL", "").strip().lower()
            if env_mm in ("1", "true", "yes"):
                self.is_multimodal = True
            elif env_mm in ("0", "false", "no"):
                self.is_multimodal = False
            else:
                self.is_multimodal = any(
                    tag in self.model_name.lower()
                    for tag in ("embed-5", "titan-embed-image", "clip", "vision", "vl-embedding")
                )

        self.db_path.mkdir(parents=True, exist_ok=True)
        self._db = lancedb.connect(str(self.db_path))
        schema = get_schema(self.dim)

        if self.table_name in self._db.table_names():
            self._table = self._db.open_table(self.table_name)
        else:
            self._table = self._db.create_table(self.table_name, schema=schema, mode="create")

    # ---------- эмбеддинги ----------

    def _get_text_embedding(self, text: str) -> list[float] | None:
        if not self.api_key or not self.dim:
            return None
        try:
            import openai

            client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)
            response = client.embeddings.create(input=[text], model=self.model_name)
            vec = list(response.data[0].embedding)
            return vec if len(vec) == self.dim else None
        except Exception:
            return None

    def _get_image_embedding(self, png_path: str | Path, fallback_text: str) -> list[float] | None:
        if not self.api_key or not self.dim:
            return None

        if self.is_multimodal and png_path and Path(png_path).is_file():
            try:
                import openai

                raw_bytes = Path(png_path).read_bytes()
                b64_img = base64.b64encode(raw_bytes).decode("utf-8")
                data_uri = f"data:image/png;base64,{b64_img}"

                client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)
                for payload in (
                    [data_uri],
                    [{"image": data_uri, "text": fallback_text}],
                    [{"type": "image_url", "image_url": {"url": data_uri}}],
                ):
                    try:
                        response = client.embeddings.create(input=payload, model=self.model_name)
                        vec = list(response.data[0].embedding)
                        if len(vec) == self.dim:
                            return vec
                    except Exception:
                        continue
            except Exception:
                pass

        return self._get_text_embedding(fallback_text)

    @staticmethod
    def _read_source(source: str | Path) -> tuple[str, Path | None]:
        if isinstance(source, Path):
            path = source
        elif isinstance(source, str):
            if source.lstrip("\ufeff \t\r\n").startswith("<"):
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
        target.write_bytes(bytes(data))
        return str(target)

    def _is_auto_png(self, png_path: str | None, svg_id: str) -> bool:
        return png_path is not None and Path(png_path) == self._auto_png_path(svg_id)

    # ---------- Create ----------

    def add(
        self,
        source: str | Path,
        *,
        name: str | None = None,
        description: str | None = None,
        png_path: str | Path | None = None,
        detail_level: DetailLevel = "L",
        role: ObjectRole = "main",
        concept_key: str | None = None,
    ) -> SVGImage:
        svg, path = self._read_source(source)
        _validate_svg(svg)

        digest = compute_hash(svg)
        existing = self.get_by_hash(digest)
        if existing:
            raise DuplicateSVGError(existing.id)

        if png_path is not None and not Path(png_path).is_file():
            raise FileNotFoundError(f"PNG не найден: {png_path}")

        norm_level: DetailLevel = "S" if str(detail_level).upper() == "S" else "L"
        norm_role: ObjectRole = "background" if str(role).lower() == "background" else "main"

        svg_id = uuid.uuid4().hex
        png = str(png_path) if png_path is not None else self._render_png(svg, svg_id)

        resolved_name = name if name is not None else (path.stem if path else None)
        resolved_concept = concept_key or normalize_concept_key(resolved_name or (path.stem if path else svg_id))
        resolved_desc = description or resolved_name or "SVG illustration"
        fallback_text = f"Name: {resolved_name}. Role: {norm_role}. Size: {norm_level}. Description: {resolved_desc}"

        vector = self._get_image_embedding(png, fallback_text=fallback_text)

        image = SVGImage(
            id=svg_id,
            svg=svg,
            hash=digest,
            name=resolved_name,
            source_path=str(path) if path else None,
            png_path=png,
            description=description,
            detail_level=norm_level,
            role=norm_role,
            concept_key=resolved_concept,
        )

        row = self._to_row(image, vector=vector, dim=self.dim)
        self._table.add([row])
        return image

    # ---------- Read ----------

    def get(self, svg_id: str) -> SVGImage:
        rows = self._table.search().where(f"id = {_quote(svg_id)}").limit(1).to_list()
        if not rows:
            raise SVGNotFoundError(f"Нет записи с id={svg_id}")
        return self._to_image(rows[0])

    def get_by_hash(self, digest: str) -> SVGImage | None:
        rows = self._table.search().where(f"hash = {_quote(digest)}").limit(1).to_list()
        return self._to_image(rows[0]) if rows else None

    def find_variant(self, svg_id: str, target_detail_level: str) -> SVGImage | None:
        """Находит парный вариант того же объекта (по concept_key и role) в размере L или S."""
        level = "S" if str(target_detail_level).upper() == "S" else "L"
        cur = self.get(svg_id)
        if cur.detail_level == level:
            return cur
        if not cur.concept_key:
            return None

        all_rows = self._table.search().limit(2000).to_list()
        for r in all_rows:
            if (
                r.get("concept_key") == cur.concept_key
                and (r.get("role") or "main") == cur.role
                and (r.get("detail_level") or "L") == level
            ):
                return self._to_image(r)
        return None

    def find_role_variant(self, svg_id: str, target_role: str) -> SVGImage | None:
        """Находит парный вариант того же объекта (по concept_key и detail_level) в роли 'main' или 'background'."""
        role = "background" if str(target_role).lower() == "background" else "main"
        cur = self.get(svg_id)
        if cur.role == role:
            return cur
        if not cur.concept_key:
            return None

        all_rows = self._table.search().limit(2000).to_list()
        for r in all_rows:
            if (
                r.get("concept_key") == cur.concept_key
                and (r.get("detail_level") or "L") == cur.detail_level
                and (r.get("role") or "main") == role
            ):
                return self._to_image(r)
        return None

    def list(
        self,
        limit: int | None = None,
        offset: int = 0,
        *,
        detail_level: str | None = None,
        role: str | None = None,
    ) -> list[SVGImage]:
        all_rows = self._table.search().limit(2000).to_list()
        items = [self._to_image(r) for r in all_rows]
        if detail_level:
            lvl = detail_level.upper()
            items = [it for it in items if it.detail_level == lvl]
        if role:
            rl = role.lower()
            items = [it for it in items if it.role == rl]
        if offset:
            items = items[offset:]
        if limit is not None:
            items = items[:limit]
        return items

    def exists(self, svg_id: str) -> bool:
        return self._table.count_rows(filter=f"id = {_quote(svg_id)}") > 0

    def count(self) -> int:
        return self._table.count_rows()

    # ---------- Гибридный поиск (векторный + текстовый буст) ----------

    def search(
        self,
        query: str,
        limit: int = 10,
        *,
        detail_level: str | None = None,
        role: str | None = None,
    ) -> list[SVGImage]:
        q_clean = (query or "").strip()
        lvl = detail_level.upper() if detail_level else None
        rl = role.lower() if role else None

        def _matches_filter(img: SVGImage) -> bool:
            if lvl and img.detail_level != lvl:
                return False
            if rl and img.role != rl:
                return False
            return True

        if not q_clean:
            return self.list(limit=limit, detail_level=lvl, role=rl)

        all_rows = self._table.search().limit(2000).to_list()
        candidates = [self._to_image(r) for r in all_rows]
        candidates = [img for img in candidates if _matches_filter(img)]
        if not candidates:
            return []

        q_lower = q_clean.lower()
        words = [w for w in re.split(r"[\s,;]+", q_lower) if len(w) >= 2]

        # Текстовый скоринг
        text_scores: dict[str, float] = {}
        for img in candidates:
            name = (img.name or "").lower()
            desc = (img.description or "").lower()
            concept = (img.concept_key or "").lower()
            sc = 0.0
            if q_lower in name or q_lower in concept:
                sc += 6.0
            if q_lower in desc:
                sc += 3.0
            for w in words:
                if w in name or w in concept:
                    sc += 3.0
                if w in desc:
                    sc += 1.5
            text_scores[img.id] = sc

        # Векторный скоринг
        vec_rank_scores: dict[str, float] = {}
        if self.dim and self.api_key:
            vec = self._get_text_embedding(q_clean)
            if vec is not None and any(v != 0.0 for v in vec):
                try:
                    rows = self._table.search(vec).limit(max(limit * 6, 40)).to_list()
                    filtered_vec = [self._to_image(r) for r in rows]
                    filtered_vec = [img for img in filtered_vec if _matches_filter(img)]
                    for rank_idx, img in enumerate(filtered_vec):
                        vec_rank_scores[img.id] = max(0.0, 5.0 - rank_idx * 0.25)
                except Exception:
                    pass

        combined: list[tuple[float, SVGImage]] = []
        for img in candidates:
            total = text_scores.get(img.id, 0.0) + vec_rank_scores.get(img.id, 0.0)
            combined.append((total, img))

        combined.sort(key=lambda pair: pair[0], reverse=True)
        return [img for _, img in combined[:limit]]

    # ---------- Update & Delete ----------

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

        fallback_text = f"Name: {old.name}. Role: {old.role}. Size: {old.detail_level}. Description: {old.description or ''}"
        vector = self._get_image_embedding(png, fallback_text=fallback_text)

        values: dict[str, Any] = {
            "svg": svg,
            "hash": digest,
            "source_path": str(path) if path else None,
            "png_path": png,
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
        detail_level: Any = _UNSET,
        role: Any = _UNSET,
        concept_key: Any = _UNSET,
    ) -> SVGImage:
        image = self.get(svg_id)
        values: dict[str, Any] = {}

        new_name = name if name is not _UNSET else image.name
        new_desc = description if description is not _UNSET else image.description
        new_png = png_path if png_path is not _UNSET else image.png_path

        if name is not _UNSET:
            values["name"] = name
        if description is not _UNSET:
            values["description"] = description
        if detail_level is not _UNSET:
            values["detail_level"] = "S" if str(detail_level).upper() == "S" else "L"
        if role is not _UNSET:
            values["role"] = "background" if str(role).lower() == "background" else "main"
        if concept_key is not _UNSET:
            values["concept_key"] = concept_key
        if png_path is not _UNSET:
            if png_path is not None and not Path(png_path).is_file():
                raise FileNotFoundError(f"PNG не найден: {png_path}")
            values["png_path"] = None if png_path is None else str(png_path)

        if any(x is not _UNSET for x in (name, description, png_path, detail_level, role)) and self.dim:
            fallback_text = f"Name: {new_name}. Description: {new_desc or ''}"
            vector = self._get_image_embedding(new_png or "", fallback_text=fallback_text)
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
        if values:
            self._table.update(where=f"id = {_quote(svg_id)}", values=values)

    @staticmethod
    def _to_row(
        image: SVGImage,
        vector: list[float] | None = None,
        dim: int | None = None,
    ) -> dict[str, Any]:
        row: dict[str, Any] = {
            "id": image.id,
            "svg": image.svg,
            "hash": image.hash,
            "name": image.name,
            "source_path": image.source_path,
            "png_path": image.png_path,
            "description": image.description,
            "detail_level": image.detail_level,
            "role": image.role,
            "concept_key": image.concept_key,
        }
        if dim is not None and dim > 0:
            row["vector"] = vector if (vector is not None and len(vector) == dim) else [0.0] * dim
        return row

    @staticmethod
    def _to_image(row: dict[str, Any]) -> SVGImage:
        return SVGImage(
            id=row["id"],
            svg=row["svg"],
            hash=row["hash"],
            name=row.get("name"),
            source_path=row.get("source_path"),
            png_path=row.get("png_path"),
            description=row.get("description"),
            detail_level="S" if str(row.get("detail_level") or "L").upper() == "S" else "L",
            role="background" if str(row.get("role") or "main").lower() == "background" else "main",
            concept_key=row.get("concept_key"),
        )
