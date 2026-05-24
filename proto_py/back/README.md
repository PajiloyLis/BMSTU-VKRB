# Backend

FastAPI слой для CYK-разбора.

## Запуск

```bash
cd /home/ivan/Study/dicklom/proto_py
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m uvicorn back.main:app --reload
```

## Структура

- `controllers/` — HTTP endpoints (`/api/parse`, `/api/health`);
- `services/` — сценарии приложения и orchestration;
- `repositories/` — файловая система и артефакты;
- `schemas/` — Pydantic DTO;
- `core/` — конфигурация путей.

