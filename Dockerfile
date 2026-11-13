# Этап 1: Сборка React-фронтенда (выполняется на мощном билдере Render)
FROM node:20-slim AS frontend-build
WORKDIR /frontend

COPY . .
RUN if [ -f "package.json" ]; then \
        npm install && npm run build; \
    elif [ -f "src/frontend/package.json" ]; then \
        cd src/frontend && npm install && npm run build && mv dist /frontend/dist; \
    elif [ -f "frontend/package.json" ]; then \
        cd frontend && npm install && npm run build && mv dist /frontend/dist; \
    fi

# Этап 2: Лёгкий Python-рантайм (укладывается в ~220 МБ RAM из 512 МБ на Render)
FROM python:3.11-slim
WORKDIR /app

# Оптимизация памяти для лимита 512 MB на Render Free Tier
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONOPTIMIZE=1 \
    MALLOC_TRIM_THRESHOLD_=65536

# Устанавливаем зависимости Python
RUN pip install --no-cache-dir \
    fastapi "uvicorn[standard]" pydantic python-dotenv \
    langchain-core langchain-openai langgraph \
    lancedb pyarrow resvg-py pillow openai

# Копируем весь проект и собранный фронтенд
COPY . /app
COPY --from=frontend-build /frontend/dist /app/dist

# Дублируем собранный фронтенд (dist) и базу картинок (data) рядом с server.py, если он лежит в подпапке
RUN if [ -d "/app/src/backend" ]; then cp -r /app/dist /app/src/backend/dist; fi && \
    if [ -d "/app/backend" ]; then cp -r /app/dist /app/backend/dist; fi && \
    if [ -d "/app/data" ] && [ -d "/app/src/backend" ] && [ ! -d "/app/src/backend/data" ]; then cp -r /app/data /app/src/backend/data; fi && \
    if [ -d "/app/data" ] && [ -d "/app/backend" ] && [ ! -d "/app/backend/data" ]; then cp -r /app/data /app/backend/data; fi

EXPOSE 8000

# Запускаем сервер в 1 воркер из папки, где лежит server.py
CMD ["sh", "-c", "if [ -f src/backend/server.py ]; then cd src/backend; elif [ -f backend/server.py ]; then cd backend; fi && uvicorn server:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]