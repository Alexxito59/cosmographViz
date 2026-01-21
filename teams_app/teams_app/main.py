from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import duckdb
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

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
) -> JSONResponse:
    start_year, end_year = parse_period(period)
    min_pubs = 1 if include_single_pub else 2
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
        node_rows = con.execute(
            sql, [start_year, end_year, max_authors_per_doc, max_authors_per_doc, min_pubs]
        ).fetchall()
        edge_rows = con.execute(
            edges_sql, [start_year, end_year, max_authors_per_doc, max_authors_per_doc, min_pubs]
        ).fetchall()
    nodes = [
        {
            "id": str(row[0]),
            "lastname": row[1] or "",
            "givenname": row[2] or "",
            "pubs": int(row[3]),
        }
        for row in node_rows
    ]
    edges = [
        {
            "source": str(row[0]),
            "target": str(row[1]),
            "weight": int(row[2]),
        }
        for row in edge_rows
    ]
    return JSONResponse({"nodes": nodes, "edges": edges})


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
) -> JSONResponse:
    try:
        logger.info(f"Loading graph for team {team_id}, period {period}")
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
            logger.debug(f"Executing nodes query for team {team_id}")
            node_rows = con.execute(
                nodes_sql, [period, team_id, start_year, end_year, max_authors_per_doc, max_authors_per_doc]
            ).fetchall()
            logger.debug(f"Found {len(node_rows)} nodes")
            
            logger.debug(f"Executing edges query for team {team_id}")
            edge_rows = con.execute(
                edges_sql, [period, team_id, start_year, end_year, max_authors_per_doc, max_authors_per_doc]
            ).fetchall()
            logger.debug(f"Found {len(edge_rows)} edges")
        
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
        
        logger.info(f"Team {team_id}: {team_count} team members, {environment_count} environment nodes")
        
        edges = [
            {
                "source": str(row[0]),
                "target": str(row[1]),
                "weight": int(row[2]),
            }
            for row in edge_rows
        ]
        
        return JSONResponse({"nodes": nodes, "edges": edges})
        
    except Exception as e:
        logger.error(f"Error loading graph for team {team_id}, period {period}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to load team graph: {str(e)}")
