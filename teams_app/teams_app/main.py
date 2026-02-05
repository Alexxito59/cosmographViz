from __future__ import annotations

import json
import logging
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any

import duckdb
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

# Настройка логирования для детального анализа производительности
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent.parent
DB_PATH = PROJECT_ROOT / "lis.duckdb"
STATIC_DIR = APP_DIR / "static"

app = FastAPI(title="LIS Teams Explorer", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def connect() -> duckdb.DuckDBPyConnection:
    if not DB_PATH.exists():
        raise HTTPException(status_code=500, detail=f"DuckDB file not found at {DB_PATH}")
    return duckdb.connect(str(DB_PATH), read_only=True)


def parse_period(period: str) -> tuple[int, int]:
    try:
        start_str, end_str = period.split("-")
        start, end = int(start_str), int(end_str)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail="Invalid period format. Use YYYY-YYYY") from exc
    if start > end:
        raise HTTPException(status_code=400, detail="Invalid period: start year > end year")
    return start, end


def json_serialize_value(value: Any) -> Any:
    """
    Преобразует значение в JSON-совместимый формат.
    Обрабатывает date, datetime и другие специальные типы.
    """
    if value is None:
        return None
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, (int, float, str, bool)):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    # Для других типов пробуем преобразовать в строку
    try:
        return str(value)
    except Exception:  # noqa: BLE001
        return None


@app.get("/", response_class=FileResponse)
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/favicon.ico")
async def favicon() -> Response:
    return Response(status_code=204)


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/api/periods")
async def get_periods() -> JSONResponse:
    with connect() as con:
        rows = con.execute("SELECT DISTINCT period FROM teams ORDER BY period").fetchall()
    return JSONResponse({"periods": [row[0] for row in rows]})


@app.get("/api/teams")
async def list_teams(
    period: str = Query(..., description="Sliding window period, e.g. 2023-2025"),
    query: str | None = Query(None, description="Filter teams by author name"),
) -> JSONResponse:
    _, _ = parse_period(period)
    query_param = f"%{query.lower()}%" if query else None
    sql = """
        WITH team_members AS (
            SELECT period, team_id, author_id, status
            FROM teams
            WHERE period = ?
        ),
        team_stats AS (
            SELECT
                period,
                team_id,
                COUNT(*) AS authors_count,
                SUM(CASE WHEN status = 'core' THEN 1 ELSE 0 END) AS core_count,
                SUM(CASE WHEN status = 'periphery' THEN 1 ELSE 0 END) AS periphery_count
            FROM team_members
            GROUP BY period, team_id
        ),
        filtered_teams AS (
            SELECT ts.*
            FROM team_stats ts
            WHERE
                ? IS NULL
                OR EXISTS (
                    SELECT 1
                    FROM team_members tm
                    JOIN authors a ON a.id = tm.author_id
                    WHERE tm.team_id = ts.team_id
                        AND tm.period = ts.period
                        AND LOWER(a.lastname || ' ' || COALESCE(a.givenname, '')) LIKE ?
                )
        )
        SELECT
            ft.team_id,
            ft.authors_count,
            ft.core_count,
            ft.periphery_count,
            COALESCE(
                string_agg(
                    DISTINCT a.lastname || ' ' || COALESCE(a.givenname, ''),
                    ', '
                ),
                ''
            ) AS sample_authors
        FROM filtered_teams ft
        LEFT JOIN LATERAL (
            SELECT a.lastname, a.givenname
            FROM teams tm
            JOIN authors a ON a.id = tm.author_id
            WHERE tm.period = ft.period AND tm.team_id = ft.team_id
            ORDER BY (tm.status = 'periphery'), a.lastname
            LIMIT 3
        ) a ON true
        GROUP BY ft.team_id, ft.authors_count, ft.core_count, ft.periphery_count
        ORDER BY ft.team_id
    """
    params: list[Any] = [period, query_param, query_param]
    with connect() as con:
        rows = con.execute(sql, params).fetchall()
    data = [
        {
            "team_id": int(row[0]),
            "authors_count": int(row[1]),
            "core_count": int(row[2]),
            "periphery_count": int(row[3]),
            "sample_authors": row[4],
        }
        for row in rows
    ]
    return JSONResponse({"teams": data})


@app.get("/api/graph")
async def get_graph(
    period: str = Query(..., description="Sliding window period, e.g. 2023-2025"),
    max_authors_per_doc: int = Query(
        100,
        ge=0,
        le=2000,
        description="Exclude documents with more than this many authors (0 = no limit). Helps avoid huge cliques.",
    ),
    include_single_pub: bool = Query(
        False,
        description="Include authors with exactly 1 publication in the period (by default they are excluded).",
    ),
    explain: bool = Query(False, description="Return EXPLAIN ANALYZE for SQL queries (for debugging)"),
) -> JSONResponse:
    total_start = time.perf_counter()
    start_year, end_year = parse_period(period)
    min_pubs = 1 if include_single_pub else 2
    
    logger.info(f"[PERF] Starting graph load: period={period}, max_authors={max_authors_per_doc}, include_single={include_single_pub}")
    sql = """
        WITH period_docs AS (
            SELECT eid, year
            FROM docs
            WHERE year BETWEEN ? AND ?
        ),
        doc_authors_raw AS (
            SELECT DISTINCT d.eid AS doc_id, ad.auth_id
            FROM period_docs d
            JOIN auth_doc ad ON ad.doc_id = d.eid
        ),
        doc_author_counts AS (
            SELECT doc_id, COUNT(*) AS k
            FROM doc_authors_raw
            GROUP BY doc_id
        ),
        doc_authors AS (
            SELECT dar.doc_id, dar.auth_id
            FROM doc_authors_raw dar
            JOIN doc_author_counts dac USING (doc_id)
            WHERE (? = 0 OR dac.k <= ?)
        ),
        node_stats AS (
            SELECT auth_id, COUNT(DISTINCT doc_id) AS pubs
            FROM doc_authors
            GROUP BY auth_id
        ),
        filtered_authors AS (
            SELECT auth_id, pubs
            FROM node_stats
            WHERE pubs >= ?
        )
        SELECT
            fa.auth_id,
            a.lastname,
            a.givenname,
            fa.pubs
        FROM filtered_authors fa
        LEFT JOIN authors a ON a.id = fa.auth_id
        ORDER BY fa.pubs DESC;
    """
    edges_sql = """
        WITH period_docs AS (
            SELECT eid, year
            FROM docs
            WHERE year BETWEEN ? AND ?
        ),
        doc_authors_raw AS (
            SELECT DISTINCT d.eid AS doc_id, ad.auth_id
            FROM period_docs d
            JOIN auth_doc ad ON ad.doc_id = d.eid
        ),
        doc_author_counts AS (
            SELECT doc_id, COUNT(*) AS k
            FROM doc_authors_raw
            GROUP BY doc_id
        ),
        doc_authors AS (
            SELECT dar.doc_id, dar.auth_id
            FROM doc_authors_raw dar
            JOIN doc_author_counts dac USING (doc_id)
            WHERE (? = 0 OR dac.k <= ?)
        ),
        node_stats AS (
            SELECT auth_id, COUNT(DISTINCT doc_id) AS pubs
            FROM doc_authors
            GROUP BY auth_id
        ),
        filtered_authors AS (
            SELECT auth_id
            FROM node_stats
            WHERE pubs >= ?
        )
        SELECT
            LEAST(a1.auth_id, a2.auth_id) AS source,
            GREATEST(a1.auth_id, a2.auth_id) AS target,
            COUNT(DISTINCT a1.doc_id) AS weight
        FROM doc_authors a1
        JOIN doc_authors a2
            ON a1.doc_id = a2.doc_id AND a1.auth_id < a2.auth_id
        JOIN filtered_authors f1 ON f1.auth_id = a1.auth_id
        JOIN filtered_authors f2 ON f2.auth_id = a2.auth_id
        GROUP BY source, target;
    """
    with connect() as con:
        # Анализ запроса узлов
        if explain:
            logger.info("[PERF] EXPLAIN ANALYZE for nodes query:")
            # Подставляем параметры для EXPLAIN ANALYZE
            explain_sql = sql.replace("?", str(start_year)).replace("?", str(end_year)).replace("?", str(max_authors_per_doc)).replace("?", str(max_authors_per_doc)).replace("?", str(min_pubs))
            try:
                explain_nodes = con.execute(f"EXPLAIN ANALYZE {explain_sql}").fetchall()
                for row in explain_nodes:
                    logger.info(f"[PERF] {row[0]}")
            except Exception as e:
                logger.warning(f"[PERF] Failed to get EXPLAIN ANALYZE: {e}")
        
        node_query_start = time.perf_counter()
        node_rows = con.execute(
            sql, [start_year, end_year, max_authors_per_doc, max_authors_per_doc, min_pubs]
        ).fetchall()
        node_query_time = time.perf_counter() - node_query_start
        logger.info(f"[PERF] Nodes query: {node_query_time:.3f}s, rows: {len(node_rows)}")
        
        # Анализ запроса рёбер
        if explain:
            logger.info("[PERF] EXPLAIN ANALYZE for edges query:")
            # Подставляем параметры для EXPLAIN ANALYZE
            explain_edges_sql = edges_sql.replace("?", str(start_year)).replace("?", str(end_year)).replace("?", str(max_authors_per_doc)).replace("?", str(max_authors_per_doc)).replace("?", str(min_pubs))
            try:
                explain_edges = con.execute(f"EXPLAIN ANALYZE {explain_edges_sql}").fetchall()
                for row in explain_edges:
                    logger.info(f"[PERF] {row[0]}")
            except Exception as e:
                logger.warning(f"[PERF] Failed to get EXPLAIN ANALYZE: {e}")
        
        edge_query_start = time.perf_counter()
        edge_rows = con.execute(
            edges_sql, [start_year, end_year, max_authors_per_doc, max_authors_per_doc, min_pubs]
        ).fetchall()
        edge_query_time = time.perf_counter() - edge_query_start
        logger.info(f"[PERF] Edges query: {edge_query_time:.3f}s, rows: {len(edge_rows)}")
    
    # Обработка данных узлов
    nodes_process_start = time.perf_counter()
    nodes = [
        {
            "id": str(row[0]),
            "lastname": row[1] or "",
            "givenname": row[2] or "",
            "pubs": int(row[3]),
        }
        for row in node_rows
    ]
    nodes_process_time = time.perf_counter() - nodes_process_start
    logger.info(f"[PERF] Nodes processing: {nodes_process_time:.3f}s")
    
    # Обработка данных рёбер
    edges_process_start = time.perf_counter()
    edges = [
        {
            "source": str(row[0]),
            "target": str(row[1]),
            "weight": int(row[2]),
        }
        for row in edge_rows
    ]
    edges_process_time = time.perf_counter() - edges_process_start
    logger.info(f"[PERF] Edges processing: {edges_process_time:.3f}s")
    
    # Сериализация JSON
    json_serialize_start = time.perf_counter()
    response_data = {"nodes": nodes, "edges": edges}
    if explain:
        response_data["_perf"] = {
            "node_query_time": node_query_time,
            "edge_query_time": edge_query_time,
            "nodes_process_time": nodes_process_time,
            "edges_process_time": edges_process_time,
        }
    json_serialize_time = time.perf_counter() - json_serialize_start
    logger.info(f"[PERF] JSON serialization: {json_serialize_time:.3f}s")
    
    total_time = time.perf_counter() - total_start
    logger.info(f"[PERF] Total graph load time: {total_time:.3f}s (nodes: {len(nodes)}, edges: {len(edges)})")
    
    return JSONResponse(response_data)


@app.get("/api/teams/{team_id}")
async def get_team_detail(
    team_id: int,
    period: str = Query(..., description="Sliding window period for the team"),
) -> JSONResponse:
    start_year, end_year = parse_period(period)
    authors_sql = """
        SELECT a.id, a.lastname, a.givenname, t.status
        FROM teams t
        JOIN authors a ON a.id = t.author_id
        WHERE t.period = ? AND t.team_id = ?
        ORDER BY (t.status = 'periphery'), a.lastname
    """
    publications_sql = """
        WITH period_docs AS (
            SELECT eid, title, year
            FROM docs
            WHERE year BETWEEN ? AND ?
        ),
        team_members AS (
            SELECT author_id, status
            FROM teams
            WHERE period = ? AND team_id = ?
        ),
        doc_authors AS (
            SELECT
                d.eid AS doc_id,
                d.title,
                d.year,
                ad.auth_id,
                ad.auth_seqn,
                a.lastname,
                a.givenname,
                tm.status AS team_status
            FROM period_docs d
            JOIN auth_doc ad ON ad.doc_id = d.eid
            JOIN authors a ON a.id = ad.auth_id
            LEFT JOIN team_members tm ON tm.author_id = ad.auth_id
        ),
        team_docs AS (
            SELECT DISTINCT doc_id
            FROM doc_authors
            WHERE team_status IS NOT NULL
        )
        SELECT *
        FROM doc_authors
        WHERE doc_id IN (SELECT doc_id FROM team_docs)
        ORDER BY year DESC, doc_id, auth_seqn;
    """
    with connect() as con:
        author_rows = con.execute(authors_sql, [period, team_id]).fetchall()
        doc_rows = con.execute(publications_sql, [start_year, end_year, period, team_id]).fetchall()

    authors = [
        {
            "id": int(row[0]),
            "lastname": row[1] or "",
            "givenname": row[2] or "",
            "status": row[3] or "core",
        }
        for row in author_rows
    ]

    publications: dict[str, dict[str, Any]] = {}
    for row in doc_rows:
        doc_id = str(row[0])
        title = row[1] or ""
        year = row[2]
        auth_id = str(row[3])
        seq = row[4]
        lastname = row[5] or ""
        givenname = row[6] or ""
        team_status = row[7]
        status = team_status if team_status else "other"
        if doc_id not in publications:
            publications[doc_id] = {"doc_id": doc_id, "title": title, "year": year, "authors": []}
        publications[doc_id]["authors"].append(
            {
                "id": auth_id,
                "lastname": lastname,
                "givenname": givenname,
                "status": status,
                "seq": seq,
            }
        )

    for pub in publications.values():
        pub["authors"] = sorted(pub["authors"], key=lambda a: (a["seq"] if a["seq"] is not None else 1e9))

    response = {
        "authors": authors,
        "publications": sorted(publications.values(), key=lambda p: (p["year"] or 0, p["doc_id"]), reverse=True),
    }
    return JSONResponse(response)


@app.get("/api/publications/{doc_id}")
async def get_publication_detail(
    doc_id: str,
) -> JSONResponse:
    """
    Получить полную метаинформацию о публикации по её ID.
    Возвращает все доступные поля из таблицы docs.
    """
    sql = """
        SELECT *
        FROM docs
        WHERE eid = ?
    """
    with connect() as con:
        result = con.execute(sql, [doc_id])
        # Получаем названия колонок из описания курсора
        column_names = [desc[0] for desc in result.description] if result.description else []
        rows = result.fetchall()
    
    if not rows:
        raise HTTPException(status_code=404, detail=f"Publication with id {doc_id} not found")
    
    # Создаём словарь из данных
    row = rows[0]
    publication_data = {}
    for i, col_name in enumerate(column_names):
        value = row[i]
        # Преобразуем значение в JSON-совместимый формат
        publication_data[col_name] = json_serialize_value(value)
    
    # Также получаем список авторов публикации
    authors_sql = """
        SELECT 
            a.id,
            a.lastname,
            a.givenname,
            ad.auth_seqn
        FROM auth_doc ad
        JOIN authors a ON a.id = ad.auth_id
        WHERE ad.doc_id = ?
        ORDER BY ad.auth_seqn
    """
    with connect() as con:
        author_rows = con.execute(authors_sql, [doc_id]).fetchall()
    
    authors = [
        {
            "id": int(row[0]),
            "lastname": row[1] or "",
            "givenname": row[2] or "",
            "seq": row[3] if row[3] is not None else None,
        }
        for row in author_rows
    ]
    
    response = {
        "publication": publication_data,
        "authors": authors,
    }
    return JSONResponse(response)


@app.get("/api/teams/multiple/graph")
async def get_multiple_teams_graph(
    team_ids: str = Query(..., description="Comma-separated team IDs, e.g. '1,2,3'"),
    period: str = Query(..., description="Sliding window period for the teams"),
    max_authors_per_doc: int = Query(
        100,
        ge=0,
        le=2000,
        description="Exclude documents with more than this many authors (0 = no limit).",
    ),
) -> JSONResponse:
    """
    Получить объединённый граф для нескольких команд.
    Узлы авторов объединяются - если автор есть в нескольких командах, создаётся один узел (без дубликатов).
    """
    try:
        total_start = time.perf_counter()
        # Парсим список ID команд
        team_id_list = [int(tid.strip()) for tid in team_ids.split(",") if tid.strip()]
        if not team_id_list:
            raise HTTPException(status_code=400, detail="At least one team_id required")
        
        logger.info(f"[PERF] Loading graph for teams {team_id_list}, period {period}")
        start_year, end_year = parse_period(period)
        
        # Получаем данные для каждой команды
        all_nodes = {}  # {node_id: node_data}
        all_edges = []  # список рёбер соавторства
        author_to_teams = {}  # {author_id: [team_ids]} - для поиска дубликатов
        
        for team_id in team_id_list:
            # SQL для получения узлов команды
            nodes_sql = """
                WITH team_members AS (
                    SELECT author_id, status
                    FROM teams
                    WHERE period = ? AND team_id = ?
                ),
                period_docs AS (
                    SELECT eid, year
                    FROM docs
                    WHERE year BETWEEN ? AND ?
                ),
                team_docs AS (
                    SELECT DISTINCT d.eid AS doc_id
                    FROM period_docs d
                    JOIN auth_doc ad ON ad.doc_id = d.eid
                    JOIN team_members tm ON tm.author_id = ad.auth_id
                ),
                doc_authors_raw AS (
                    SELECT DISTINCT td.doc_id, ad.auth_id
                    FROM team_docs td
                    JOIN auth_doc ad ON ad.doc_id = td.doc_id
                ),
                doc_author_counts AS (
                    SELECT doc_id, COUNT(*) AS k
                    FROM doc_authors_raw
                    GROUP BY doc_id
                ),
                doc_authors AS (
                    SELECT dar.doc_id, dar.auth_id
                    FROM doc_authors_raw dar
                    JOIN doc_author_counts dac USING (doc_id)
                    WHERE (? = 0 OR dac.k <= ?)
                ),
                node_stats AS (
                    SELECT 
                        da.auth_id,
                        COUNT(DISTINCT da.doc_id) AS pubs,
                        MAX(tm.status) AS status
                    FROM doc_authors da
                    LEFT JOIN team_members tm ON tm.author_id = da.auth_id
                    GROUP BY da.auth_id
                )
                SELECT
                    ns.auth_id,
                    a.lastname,
                    a.givenname,
                    ns.pubs,
                    ns.status
                FROM node_stats ns
                LEFT JOIN authors a ON a.id = ns.auth_id
                ORDER BY ns.status IS NULL, ns.pubs DESC;
            """
            
            # SQL для получения рёбер команды
            edges_sql = """
                WITH team_members AS (
                    SELECT author_id, status
                    FROM teams
                    WHERE period = ? AND team_id = ?
                ),
                period_docs AS (
                    SELECT eid, year
                    FROM docs
                    WHERE year BETWEEN ? AND ?
                ),
                team_docs AS (
                    SELECT DISTINCT d.eid AS doc_id
                    FROM period_docs d
                    JOIN auth_doc ad ON ad.doc_id = d.eid
                    JOIN team_members tm ON tm.author_id = ad.auth_id
                ),
                doc_authors_raw AS (
                    SELECT DISTINCT td.doc_id, ad.auth_id
                    FROM team_docs td
                    JOIN auth_doc ad ON ad.doc_id = td.doc_id
                ),
                doc_author_counts AS (
                    SELECT doc_id, COUNT(*) AS k
                    FROM doc_authors_raw
                    GROUP BY doc_id
                ),
                doc_authors AS (
                    SELECT dar.doc_id, dar.auth_id
                    FROM doc_authors_raw dar
                    JOIN doc_author_counts dac USING (doc_id)
                    WHERE (? = 0 OR dac.k <= ?)
                )
                SELECT
                    LEAST(a1.auth_id, a2.auth_id) AS source,
                    GREATEST(a1.auth_id, a2.auth_id) AS target,
                    COUNT(DISTINCT a1.doc_id) AS weight
                FROM doc_authors a1
                JOIN doc_authors a2
                    ON a1.doc_id = a2.doc_id AND a1.auth_id < a2.auth_id
                GROUP BY source, target;
            """
            
            with connect() as con:
                node_rows = con.execute(
                    nodes_sql, [period, team_id, start_year, end_year, max_authors_per_doc, max_authors_per_doc]
                ).fetchall()
                edge_rows = con.execute(
                    edges_sql, [period, team_id, start_year, end_year, max_authors_per_doc, max_authors_per_doc]
                ).fetchall()
            
            # Объединяем узлы - если автор уже есть, не создаём дубликат
            for row in node_rows:
                auth_id = str(row[0])
                
                # Отслеживаем, в каких командах автор
                if auth_id not in author_to_teams:
                    author_to_teams[auth_id] = []
                if team_id not in author_to_teams[auth_id]:
                    author_to_teams[auth_id].append(team_id)
                
                # Если узел уже есть, обновляем его данные
                if auth_id in all_nodes:
                    existing = all_nodes[auth_id]
                    # Обновляем статус: core > periphery > None
                    if row[4] == "core" or (row[4] == "periphery" and existing["status"] != "core"):
                        existing["status"] = row[4]
                    # Обновляем публикации (берём максимум)
                    existing["pubs"] = max(existing["pubs"], int(row[3]))
                    # Обновляем список команд
                    existing["team_ids"] = author_to_teams[auth_id]
                else:
                    # Создаём новый узел (используем auth_id как ID, без дубликатов)
                    all_nodes[auth_id] = {
                        "id": auth_id,  # Используем author_id как ID узла (без дубликатов)
                        "lastname": row[1] or "",
                        "givenname": row[2] or "",
                        "pubs": int(row[3]),
                        "status": row[4] or None,
                        "team_ids": author_to_teams[auth_id],  # Список команд, в которых состоит автор
                    }
            
            # Создаём рёбра соавторства (объединяем, избегая дубликатов)
            for row in edge_rows:
                source_auth = str(row[0])
                target_auth = str(row[1])
                weight = int(row[2])
                
                # Проверяем, что оба узла существуют
                if source_auth in all_nodes and target_auth in all_nodes:
                    # Ищем существующее ребро
                    existing_edge = None
                    for e in all_edges:
                        if (e["source"] == source_auth and e["target"] == target_auth) or \
                           (e["source"] == target_auth and e["target"] == source_auth):
                            existing_edge = e
                            break
                    
                    if existing_edge:
                        # Обновляем вес (берём максимум)
                        existing_edge["weight"] = max(existing_edge["weight"], weight)
                    else:
                        # Создаём новое ребро
                        all_edges.append({
                            "source": source_auth,
                            "target": target_auth,
                            "weight": weight,
                        })
        
        # Преобразуем узлы в список
        nodes_list = list(all_nodes.values())
        
        total_time = time.perf_counter() - total_start
        logger.info(f"[PERF] Total multiple teams graph load time: {total_time:.3f}s (nodes: {len(nodes_list)}, edges: {len(all_edges)})")
        
        response_data = {
            "nodes": nodes_list,
            "edges": all_edges,
            "teams": team_id_list,
            "author_duplicates": {auth_id: teams for auth_id, teams in author_to_teams.items() if len(teams) > 1},
        }
        return JSONResponse(response_data)
        
    except Exception as e:
        logger.error(f"Error loading graph for multiple teams {team_ids}, period {period}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to load multiple teams graph: {str(e)}")


@app.get("/api/authors/{author_id}/teams")
async def get_author_teams(
    author_id: str,
    period: str = Query(..., description="Sliding window period, e.g. 2023-2025"),
) -> JSONResponse:
    """
    Return list of teams (in a given period) where the author is a member.
    This is used by the frontend 'author info' popup.
    """
    _, _ = parse_period(period)
    sql = """
        WITH team_members AS (
            SELECT period, team_id, author_id, status
            FROM teams
            WHERE period = ?
        ),
        team_stats AS (
            SELECT
                period,
                team_id,
                COUNT(*) AS authors_count,
                SUM(CASE WHEN status = 'core' THEN 1 ELSE 0 END) AS core_count,
                SUM(CASE WHEN status = 'periphery' THEN 1 ELSE 0 END) AS periphery_count
            FROM team_members
            GROUP BY period, team_id
        ),
        author_teams AS (
            SELECT DISTINCT tm.team_id, tm.status AS author_status
            FROM team_members tm
            WHERE tm.author_id = ?
        )
        SELECT
            aut.team_id,
            aut.author_status,
            ts.authors_count,
            ts.core_count,
            ts.periphery_count,
            COALESCE(
                string_agg(
                    DISTINCT a.lastname || ' ' || COALESCE(a.givenname, ''),
                    ', '
                ),
                ''
            ) AS sample_authors
        FROM author_teams aut
        JOIN team_stats ts ON ts.team_id = aut.team_id AND ts.period = ?
        LEFT JOIN LATERAL (
            SELECT a.lastname, a.givenname
            FROM team_members tm
            JOIN authors a ON a.id = tm.author_id
            WHERE tm.period = ? AND tm.team_id = aut.team_id
            ORDER BY (tm.status = 'periphery'), a.lastname
            LIMIT 3
        ) a ON true
        GROUP BY aut.team_id, aut.author_status, ts.authors_count, ts.core_count, ts.periphery_count
        ORDER BY aut.team_id;
    """
    params: list[Any] = [period, str(author_id), period, period]
    with connect() as con:
        rows = con.execute(sql, params).fetchall()
    teams = [
        {
            "team_id": int(row[0]),
            "author_status": row[1] or "core",
            "authors_count": int(row[2]),
            "core_count": int(row[3]),
            "periphery_count": int(row[4]),
            "sample_authors": row[5],
        }
        for row in rows
    ]
    return JSONResponse({"author_id": str(author_id), "period": period, "teams": teams})


@app.get("/api/authors/search")
async def search_authors(
    query: str = Query(..., description="Search query (author name)"),
    period: str | None = Query(None, description="Optional period filter"),
) -> JSONResponse:
    """
    Search for authors by name in the current graph nodes.
    Returns authors that match the query and are present in the graph.
    """
    try:
        query_lower = query.lower().strip()
        if not query_lower:
            return JSONResponse({"authors": []})
        
        # Если указан период, ищем только в узлах графа этого периода
        # Иначе ищем во всех авторах
        if period:
            start_year, end_year = parse_period(period)
            # Получаем авторов из графа (те, кто есть в публикациях периода)
            sql = """
            WITH period_docs AS (
                SELECT DISTINCT eid
                FROM docs
                WHERE year BETWEEN ? AND ?
            ),
            graph_authors AS (
                SELECT DISTINCT ad.auth_id
                FROM period_docs d
                JOIN auth_doc ad ON ad.doc_id = d.eid
            )
            SELECT DISTINCT
                a.id,
                a.lastname,
                a.givenname
            FROM graph_authors ga
            JOIN authors a ON a.id = ga.auth_id
            WHERE 
                LOWER(a.lastname || ' ' || COALESCE(a.givenname, '')) LIKE ?
                OR LOWER(COALESCE(a.givenname, '') || ' ' || a.lastname) LIKE ?
            ORDER BY a.lastname, a.givenname
            LIMIT 50
            """
            search_pattern = f"%{query_lower}%"
            params = [start_year, end_year, search_pattern, search_pattern]
        else:
            # Поиск по всем авторам
            sql = """
            SELECT DISTINCT
                a.id,
                a.lastname,
                a.givenname
            FROM authors a
            WHERE 
                LOWER(a.lastname || ' ' || COALESCE(a.givenname, '')) LIKE ?
                OR LOWER(COALESCE(a.givenname, '') || ' ' || a.lastname) LIKE ?
            ORDER BY a.lastname, a.givenname
            LIMIT 50
            """
            search_pattern = f"%{query_lower}%"
            params = [search_pattern, search_pattern]
        
        with connect() as con:
            rows = con.execute(sql, params).fetchall()
        
        authors = [
            {
                "id": str(row[0]),
                "lastname": row[1] or "",
                "givenname": row[2] or "",
            }
            for row in rows
        ]
        
        return JSONResponse({"authors": authors})
    except Exception as e:
        logger.error(f"Error searching authors: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to search authors: {str(e)}")


@app.get("/api/teams/{team_id}/graph")
async def get_team_graph(
    team_id: int,
    period: str = Query(..., description="Sliding window period for the team"),
    max_authors_per_doc: int = Query(
        100,
        ge=0,
        le=2000,
        description="Exclude documents with more than this many authors (0 = no limit).",
    ),
    explain: bool = Query(False, description="Return EXPLAIN ANALYZE for SQL queries (for debugging)"),
) -> JSONResponse:
    try:
        total_start = time.perf_counter()
        logger.info(f"[PERF] Loading graph for team {team_id}, period {period}")
        start_year, end_year = parse_period(period)
        
        # SQL для получения узлов: команда + окружение
        nodes_sql = """
            WITH team_members AS (
                SELECT author_id, status
                FROM teams
                WHERE period = ? AND team_id = ?
            ),
            period_docs AS (
                SELECT eid, year
                FROM docs
                WHERE year BETWEEN ? AND ?
            ),
            -- Находим все публикации команды
            team_docs AS (
                SELECT DISTINCT d.eid AS doc_id
                FROM period_docs d
                JOIN auth_doc ad ON ad.doc_id = d.eid
                JOIN team_members tm ON tm.author_id = ad.auth_id
            ),
            -- Находим всех авторов публикаций команды (включая не из команды)
            doc_authors_raw AS (
                SELECT DISTINCT td.doc_id, ad.auth_id
                FROM team_docs td
                JOIN auth_doc ad ON ad.doc_id = td.doc_id
            ),
            doc_author_counts AS (
                SELECT doc_id, COUNT(*) AS k
                FROM doc_authors_raw
                GROUP BY doc_id
            ),
            doc_authors AS (
                SELECT dar.doc_id, dar.auth_id
                FROM doc_authors_raw dar
                JOIN doc_author_counts dac USING (doc_id)
                WHERE (? = 0 OR dac.k <= ?)
            ),
            -- Статистика по узлам: команда (со статусом) и окружение (без статуса)
            node_stats AS (
                SELECT 
                    da.auth_id,
                    COUNT(DISTINCT da.doc_id) AS pubs,
                    MAX(tm.status) AS status
                FROM doc_authors da
                LEFT JOIN team_members tm ON tm.author_id = da.auth_id
                GROUP BY da.auth_id
            )
            SELECT
                ns.auth_id,
                a.lastname,
                a.givenname,
                ns.pubs,
                ns.status
            FROM node_stats ns
            LEFT JOIN authors a ON a.id = ns.auth_id
            ORDER BY ns.status IS NULL, ns.pubs DESC;
        """
        
        # SQL для получения рёбер: внутри команды, команда-окружение, внутри окружения
        edges_sql = """
            WITH team_members AS (
                SELECT author_id, status
                FROM teams
                WHERE period = ? AND team_id = ?
            ),
            period_docs AS (
                SELECT eid, year
                FROM docs
                WHERE year BETWEEN ? AND ?
            ),
            -- Находим все публикации команды
            team_docs AS (
                SELECT DISTINCT d.eid AS doc_id
                FROM period_docs d
                JOIN auth_doc ad ON ad.doc_id = d.eid
                JOIN team_members tm ON tm.author_id = ad.auth_id
            ),
            -- Находим всех авторов публикаций команды
            doc_authors_raw AS (
                SELECT DISTINCT td.doc_id, ad.auth_id
                FROM team_docs td
                JOIN auth_doc ad ON ad.doc_id = td.doc_id
            ),
            doc_author_counts AS (
                SELECT doc_id, COUNT(*) AS k
                FROM doc_authors_raw
                GROUP BY doc_id
            ),
            doc_authors AS (
                SELECT dar.doc_id, dar.auth_id
                FROM doc_authors_raw dar
                JOIN doc_author_counts dac USING (doc_id)
                WHERE (? = 0 OR dac.k <= ?)
            )
            -- Рёбра: все связи между авторами публикаций команды
            SELECT
                LEAST(a1.auth_id, a2.auth_id) AS source,
                GREATEST(a1.auth_id, a2.auth_id) AS target,
                COUNT(DISTINCT a1.doc_id) AS weight
            FROM doc_authors a1
            JOIN doc_authors a2
                ON a1.doc_id = a2.doc_id AND a1.auth_id < a2.auth_id
            GROUP BY source, target;
        """
        
        with connect() as con:
            # Анализ запроса узлов
            if explain:
                logger.info("[PERF] EXPLAIN ANALYZE for team nodes query:")
                # Подставляем параметры для EXPLAIN ANALYZE (осторожно с типами)
                explain_nodes_sql = nodes_sql.replace("?", f"'{period}'").replace("?", str(team_id)).replace("?", str(start_year)).replace("?", str(end_year)).replace("?", str(max_authors_per_doc)).replace("?", str(max_authors_per_doc))
                try:
                    explain_nodes = con.execute(f"EXPLAIN ANALYZE {explain_nodes_sql}").fetchall()
                    for row in explain_nodes:
                        logger.info(f"[PERF] {row[0]}")
                except Exception as e:
                    logger.warning(f"[PERF] Failed to get EXPLAIN ANALYZE: {e}")
            
            node_query_start = time.perf_counter()
            node_rows = con.execute(
                nodes_sql, [period, team_id, start_year, end_year, max_authors_per_doc, max_authors_per_doc]
            ).fetchall()
            node_query_time = time.perf_counter() - node_query_start
            logger.info(f"[PERF] Team nodes query: {node_query_time:.3f}s, rows: {len(node_rows)}")
            
            # Анализ запроса рёбер
            if explain:
                logger.info("[PERF] EXPLAIN ANALYZE for team edges query:")
                # Подставляем параметры для EXPLAIN ANALYZE (осторожно с типами)
                explain_edges_sql = edges_sql.replace("?", f"'{period}'").replace("?", str(team_id)).replace("?", str(start_year)).replace("?", str(end_year)).replace("?", str(max_authors_per_doc)).replace("?", str(max_authors_per_doc))
                try:
                    explain_edges = con.execute(f"EXPLAIN ANALYZE {explain_edges_sql}").fetchall()
                    for row in explain_edges:
                        logger.info(f"[PERF] {row[0]}")
                except Exception as e:
                    logger.warning(f"[PERF] Failed to get EXPLAIN ANALYZE: {e}")
            
            edge_query_start = time.perf_counter()
            edge_rows = con.execute(
                edges_sql, [period, team_id, start_year, end_year, max_authors_per_doc, max_authors_per_doc]
            ).fetchall()
            edge_query_time = time.perf_counter() - edge_query_start
            logger.info(f"[PERF] Team edges query: {edge_query_time:.3f}s, rows: {len(edge_rows)}")
        
        # Обработка данных узлов
        nodes_process_start = time.perf_counter()
        nodes = []
        team_count = 0
        environment_count = 0
        
        for row in node_rows:
            auth_id = str(row[0])
            status = row[4]  # status может быть None для окружения
            
            node = {
                "id": auth_id,
                "lastname": row[1] or "",
                "givenname": row[2] or "",
                "pubs": int(row[3]),
                "status": status or None,  # None для окружения
            }
            nodes.append(node)
            
            if status:
                team_count += 1
            else:
                environment_count += 1
        
        nodes_process_time = time.perf_counter() - nodes_process_start
        logger.info(f"[PERF] Team nodes processing: {nodes_process_time:.3f}s ({team_count} team, {environment_count} env)")
        
        # Обработка данных рёбер
        edges_process_start = time.perf_counter()
        edges = [
            {
                "source": str(row[0]),
                "target": str(row[1]),
                "weight": int(row[2]),
            }
            for row in edge_rows
        ]
        edges_process_time = time.perf_counter() - edges_process_start
        logger.info(f"[PERF] Team edges processing: {edges_process_time:.3f}s")
        
        # Сериализация JSON
        json_serialize_start = time.perf_counter()
        response_data = {"nodes": nodes, "edges": edges}
        if explain:
            response_data["_perf"] = {
                "node_query_time": node_query_time,
                "edge_query_time": edge_query_time,
                "nodes_process_time": nodes_process_time,
                "edges_process_time": edges_process_time,
            }
        json_serialize_time = time.perf_counter() - json_serialize_start
        logger.info(f"[PERF] Team JSON serialization: {json_serialize_time:.3f}s")
        
        total_time = time.perf_counter() - total_start
        logger.info(f"[PERF] Total team graph load time: {total_time:.3f}s (nodes: {len(nodes)}, edges: {len(edges)})")
        
        return JSONResponse(response_data)
        
    except Exception as e:
        logger.error(f"Error loading graph for team {team_id}, period {period}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to load team graph: {str(e)}")
