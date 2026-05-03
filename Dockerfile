FROM node:18-bullseye AS frontend-build
WORKDIR /app/frontend

COPY frontend/package.json ./
COPY frontend/package-lock.json ./
RUN npm install --silent

COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim
WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=7860

COPY backend/ /app/backend/

RUN pip install --no-cache-dir -r /app/backend/requirements.txt

COPY --from=frontend-build /app/frontend/dist /app/backend/app/static

WORKDIR /app/backend
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]