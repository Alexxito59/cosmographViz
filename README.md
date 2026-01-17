# cosmographViz

Небольшой веб-интерфейс для просмотра команд/графа соавторства из `lis.duckdb`.

## Быстрый старт

```bash
pip install -r requirements.txt
python -m uvicorn teams_app.teams_app.main:app --reload --port 8000
```

Открыть в браузере: `http://127.0.0.1:8000`

## Где фронтенд

Фронтенд находится в `teams_app/teams_app/static/index.html` — это одна HTML‑страница
со встроенным CSS и JavaScript. Логика на клиенте делает запросы к API:

- `GET /api/periods`
- `GET /api/teams?period=...&query=...`
- `GET /api/graph?period=...`
- `GET /api/teams/{team_id}?period=...`

## База данных

Сервер использует `lis.duckdb` в корне проекта в режиме read‑only.
