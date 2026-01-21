# LIS Teams Explorer

Веб-интерфейс для визуализации графа соавторства и анализа команд исследователей на основе данных из DuckDB.

## Требования

- Python 3.8 или выше
- База данных `lis.duckdb` в корне проекта

## Установка

Установите зависимости проекта, выполнив в терминале команду:

```bash
pip install -r requirements.txt
```

## Запуск сервера

Запустите веб-сервер с помощью uvicorn, выполнив в терминале команду:

```bash
python -m uvicorn teams_app.teams_app.main:app --reload --port 8000
```


Сервер будет доступен по адресу: 
```
localhost:8000
```

### Параметры запуска

- `--reload` — автоматическая перезагрузка при изменении кода 
- `--port 8000` — порт сервера


## База данных

Сервер использует базу данных `lis.duckdb`, которая должна находиться в корне проекта. База данных открывается в режиме read-only.

## Структура проекта

```
cosmographViz/
├── teams_app/
│   └── teams_app/
│       ├── main.py              # Backend (FastAPI)
│       └── static/
│           └── index.html       # Frontend (одностраничное приложение)
├── lis.duckdb                   # База данных
├── requirements.txt             # Зависимости Python
└── README.md                    # Этот файл
```

## API Endpoints

- `GET /api/periods` — список доступных периодов
- `GET /api/teams?period=...&query=...` — список команд с фильтрацией
- `GET /api/graph?period=...` — данные графа для периода
- `GET /api/teams/{team_id}?period=...` — информация о команде
- `GET /api/teams/{team_id}/graph?period=...` — граф команды с окружением
