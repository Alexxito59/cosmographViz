from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple, Dict, Optional

import duckdb
import igraph as ig
import pandas as pd


# =========================
# ЛОГГЕР
# =========================

logger = logging.getLogger("coauthor_graph")


# =========================
# ПОДКЛЮЧЕНИЕ К DUCKDB
# =========================

def connect(path: Path):
    return duckdb.connect(str(path), read_only=True)


# =========================
# СТАТИСТИКА ПО ГОДАМ
# =========================

def fetch_publication_year_stats(con, year_from: int, year_to: int) -> pd.DataFrame:
    """
    Вернуть количество публикаций по годам в заданном диапазоне.
    Используется поле publications.publication_year (INTEGER).
    """
    query = """
    SELECT
        p.publication_year AS year,
        COUNT(*) AS n_publications
    FROM publications p
    WHERE p.publication_year BETWEEN ? AND ?
    GROUP BY p.publication_year
    ORDER BY p.publication_year
    """
    df = con.execute(query, [year_from, year_to]).df()
    logger.info("Publication stats by year:")
    for _, row in df.iterrows():
        logger.info("  %s: %s publications", row["year"], row["n_publications"])
    return df


# =========================
# РЁБРА СОАВТОРСТВА
# =========================

def fetch_coauthor_edges(
    con,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
    min_authors: int = 2,
    max_authors: Optional[int] = 10,
) -> pd.DataFrame:
    """
    Aggregated co-author edges with:
      - weight_frac: SUM(1/(n_authors - 1)) по публикациям, попавшим в фильтр
      - publications: целое число общих публикаций (k_ij) по тем же публикациям

    Фильтруем по:
      - годам year_from/year_to (поле p.publication_year)
      - числу авторов в публикации: min_authors..max_authors (включительно).
        Если max_authors=None -> только нижняя граница min_authors.
    """

    if min_authors is None or min_authors < 2:
        raise ValueError("min_authors must be >= 2 (for weight_frac denominator n_authors-1).")

    conditions = ["p.authors IS NOT NULL"]
    params: list = []

    # годы
    if year_from is not None:
        conditions.append("p.publication_year >= ?")
        params.append(year_from)
    if year_to is not None:
        conditions.append("p.publication_year <= ?")
        params.append(year_to)

    where_clause = " AND ".join(conditions)

    # фильтр по числу авторов
    # применяем в pairs (как и раньше), потому что n_authors вычисляется в expanded
    authors_filter_sql = "e1.n_authors >= ?"
    params_auth = [min_authors]
    if max_authors is not None:
        authors_filter_sql += " AND e1.n_authors <= ?"
        params_auth.append(max_authors)

    query = f"""
    WITH expanded AS (
        SELECT
            p.eid,
            CAST(a.key AS INTEGER) + 1 AS author_idx,
            COALESCE(
                a.value->>'id',
                LOWER(a.value->>'fullname'),
                LOWER(a.value->>'name')
            ) AS author_id,
            COALESCE(
                a.value->>'fullname',
                a.value->>'name'
            ) AS author_name,
            COUNT(*) OVER (PARTITION BY p.eid) AS n_authors
        FROM publications p, json_each(p.authors) a
        WHERE {where_clause}
    ),
    pairs AS (
        SELECT
            CASE WHEN e1.author_id <= e2.author_id THEN e1.author_id ELSE e2.author_id END AS a_id,
            CASE WHEN e1.author_id <= e2.author_id THEN e2.author_id ELSE e1.author_id END AS b_id,
            CASE WHEN e1.author_id <= e2.author_id THEN e1.author_name ELSE e2.author_name END AS a_name,
            CASE WHEN e1.author_id <= e2.author_id THEN e2.author_name ELSE e1.author_name END AS b_name,
            e1.n_authors AS n_authors
        FROM expanded e1
        JOIN expanded e2
          ON e1.eid = e2.eid AND e1.author_idx < e2.author_idx
        WHERE e1.author_id IS NOT NULL
          AND e2.author_id IS NOT NULL
          AND {authors_filter_sql}
    )
    SELECT
        a_id AS source,
        b_id AS target,
        MIN(a_name) AS source_name,
        MIN(b_name) AS target_name,
        COUNT(*) AS publications,                    -- k_ij
        SUM(1.0 / (n_authors - 1)) AS weight_frac    -- дробный вес для Лейдена
    FROM pairs
    GROUP BY 1, 2
    ORDER BY weight_frac DESC
    """

    params_final = params + params_auth
    return con.execute(query, params_final).df()


# =========================
# k_i: ЧИСЛО ПУБЛИКАЦИЙ АВТОРА В КОЛЛАБОРАЦИЯХ (ПО ТОМУ ЖЕ ФИЛЬТРУ min/max authors)
# =========================

def fetch_author_pub_counts(
    con,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
    min_authors: int = 3,
    max_authors: Optional[int] = 39,
) -> pd.DataFrame:
    """
    k_i: количество публикаций автора i в коллаборации в заданном годовом диапазоне
    по тем же публикациям, что используются для рёбер:
      - min_authors..max_authors (если max_authors=None -> только нижняя граница)
      - p.publication_year в [year_from, year_to]
    """

    if min_authors is None or min_authors < 2:
        raise ValueError("min_authors must be >= 2")

    conditions = ["p.authors IS NOT NULL"]
    params: list = []

    if year_from is not None:
        conditions.append("p.publication_year >= ?")
        params.append(year_from)
    if year_to is not None:
        conditions.append("p.publication_year <= ?")
        params.append(year_to)

    where_clause = " AND ".join(conditions)

    # фильтр по числу авторов (на уровне expanded/collab)
    authors_filter_sql = "n_authors >= ?"
    params_auth = [min_authors]
    if max_authors is not None:
        authors_filter_sql += " AND n_authors <= ?"
        params_auth.append(max_authors)

    query = f"""
    WITH expanded AS (
        SELECT
            p.eid,
            COALESCE(
                a.value->>'id',
                LOWER(a.value->>'fullname'),
                LOWER(a.value->>'name')
            ) AS author_id,
            COUNT(*) OVER (PARTITION BY p.eid) AS n_authors
        FROM publications p, json_each(p.authors) a
        WHERE {where_clause}
    ),
    collab AS (
        SELECT DISTINCT
            eid,
            author_id
        FROM expanded
        WHERE author_id IS NOT NULL
          AND {authors_filter_sql}
    )
    SELECT
        author_id,
        COUNT(*) AS k
    FROM collab
    GROUP BY author_id
    """

    return con.execute(query, params + params_auth).df()


# =========================
# ГОД НАЧАЛА КАРЬЕРЫ АВТОРА (year_start)
# =========================

def fetch_author_year_start(con) -> pd.DataFrame:
    """
    year_start: первый год, когда автор появляется в ЛЮБОЙ публикации
    (без фильтра по годам и числу авторов).
    """
    query = """
    WITH expanded AS (
        SELECT
            p.publication_year AS year,
            COALESCE(
                a.value->>'id',
                LOWER(a.value->>'fullname'),
                LOWER(a.value->>'name')
            ) AS author_id
        FROM publications p, json_each(p.authors) a
        WHERE p.authors IS NOT NULL
          AND p.publication_year IS NOT NULL
    )
    SELECT
        author_id,
        MIN(year) AS year_start
    FROM expanded
    WHERE author_id IS NOT NULL
    GROUP BY author_id
    """
    return con.execute(query).df()


# =========================
# ПОСТРОЕНИЕ ВЕРШИН
# =========================

def build_vertices(edges: pd.DataFrame) -> pd.DataFrame:
    vertices = pd.concat(
        [
            edges[["source", "source_name"]].rename(
                columns={"source": "name", "source_name": "label"}
            ),
            edges[["target", "target_name"]].rename(
                columns={"target": "name", "target_name": "label"}
            ),
        ],
        ignore_index=True,
    )
    vertices = vertices.drop_duplicates(subset=["name"]).reset_index(drop=True)
    return vertices


# =========================
# ПОСТРОЕНИЕ ГРАФА
# =========================

def build_graph(edges: pd.DataFrame) -> Tuple[ig.Graph, pd.DataFrame]:
    vertices = build_vertices(edges).reset_index(drop=True)

    idx_map = {name: idx for idx, name in enumerate(vertices["name"])}

    edge_idx = edges.assign(
        source_idx=edges["source"].map(idx_map),
        target_idx=edges["target"].map(idx_map),
    )

    tuples = list(zip(edge_idx["source_idx"], edge_idx["target_idx"]))

    g = ig.Graph(
        n=len(vertices),
        edges=tuples,
        directed=False,
    )

    # Лейден использует этот вес
    g.es["weight"] = edge_idx["weight_frac"].tolist()
    g.vs["name"] = vertices["name"].tolist()
    g.vs["label"] = vertices["label"].tolist()

    return g, vertices


# =========================
# ЗАПУСК LEIDEN
# =========================

def run_leiden(g: ig.Graph, resolution: float) -> ig.clustering.VertexClustering:
    return g.community_leiden(
        weights=g.es["weight"],
        resolution=resolution
    )


# =========================
# СОХРАНЕНИЕ MEMBERSHIP В CSV
# =========================

def write_membership(
    vertices: pd.DataFrame,
    clustering: ig.clustering.VertexClustering,
    path: Path
) -> None:
    community_sizes = clustering.sizes()

    membership = pd.DataFrame(
        {
            "author_id": vertices["name"],
            "author_name": vertices["label"],
            "community": clustering.membership,
            "community_size": [community_sizes[c] for c in clustering.membership],
        }
    )

    membership.to_csv(path, index=False, encoding="utf-8")
    logger.info("Saved community membership to %s", path)


# =========================
# PHI-ИНДЕКС
# =========================

def compute_phi(neigh_dict: Dict[str, int]) -> int:
    if not neigh_dict:
        return 0
    counts = sorted(neigh_dict.values(), reverse=True)
    phi = 0
    for idx, c in enumerate(counts, start=1):
        if c >= idx:
            phi = idx
    return phi


# =========================
# ВЫЧИСЛЕНИЕ МЕТРИК ДЛЯ АВТОРОВ
# =========================

def compute_author_metrics(
    edges_enriched: pd.DataFrame,
    author_communities: Dict[str, int],
) -> pd.DataFrame:
    """
    На вход:
      edges_enriched: содержит source, target, weight_pub, source_community, target_community
      author_communities: dict author_id -> community (или -1)

    Возвращает DataFrame с колонками:
      author_id, degree, degree_in_community, phi, phi_in_community
    """

    neigh_all: Dict[str, Dict[str, int]] = {}
    neigh_intra: Dict[str, Dict[str, int]] = {}

    for _, row in edges_enriched.iterrows():
        s = row["source"]
        t = row["target"]
        k_ij = int(row["weight_pub"])
        cs = author_communities.get(s, -1)
        ct = author_communities.get(t, -1)

        if s not in neigh_all:
            neigh_all[s] = {}
        if t not in neigh_all:
            neigh_all[t] = {}
        neigh_all[s][t] = k_ij
        neigh_all[t][s] = k_ij

        if cs != -1 and cs == ct:
            if s not in neigh_intra:
                neigh_intra[s] = {}
            if t not in neigh_intra:
                neigh_intra[t] = {}
            neigh_intra[s][t] = k_ij
            neigh_intra[t][s] = k_ij

    authors = set(author_communities.keys())
    metrics = []
    for author_id in authors:
        nd_all = neigh_all.get(author_id, {})
        nd_com = neigh_intra.get(author_id, {})

        degree = len(nd_all)
        degree_in_community = len(nd_com)
        phi_all = compute_phi(nd_all)
        phi_comm = compute_phi(nd_com)

        metrics.append(
            {
                "author_id": author_id,
                "degree": degree,
                "degree_in_community": degree_in_community,
                "phi": phi_all,
                "phi_in_community": phi_comm,
            }
        )

    return pd.DataFrame(metrics)


# =========================
# СОХРАНЕНИЕ В DUCKDB: РЁБРА И АВТОРЫ
# =========================

def save_to_duckdb(
    path: Path,
    edges_enriched: pd.DataFrame,
    authors_table: pd.DataFrame,
) -> None:
    con_out = duckdb.connect(str(path))
    try:
        con_out.register("edges_df", edges_enriched)
        con_out.register("authors_df", authors_table)

        con_out.execute("CREATE OR REPLACE TABLE coauthor_edges AS SELECT * FROM edges_df")
        con_out.execute("CREATE OR REPLACE TABLE coauthor_authors AS SELECT * FROM authors_df")

        logger.info("Saved coauthor_edges and coauthor_authors to %s", path)
    finally:
        con_out.close()


# =========================
# MAIN
# =========================

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s"
    )

    # ====== НАСТРОЙКИ ЗДЕСЬ ======
    db_path = Path("data/topics.duckdb")                # входной duckdb с publications
    resolution = 0.00001                                # параметр Leiden
    communities_csv = Path("coauthor_communities.csv")  # CSV с membership

    year_from = 2023
    year_to = 2025

    # НОВОЕ: фильтр по числу авторов в публикации
    min_authors = 2
    max_authors = 10   # None -> без верхней границы

    output_db_path = Path("coauthor_graph_pub10.duckdb")  # новый duckdb с рёбрами и авторами
    # ============================

    logger.info("Connecting to %s", db_path)
    con = connect(db_path)

    # 1) Статистика по годам (как было)
    logger.info("Fetching publication stats by year [%d, %d]...", year_from, year_to)
    fetch_publication_year_stats(con, year_from, year_to)

    # 2) Рёбра соавторства с фильтром по годам и числу авторов
    logger.info("Building co-author edges with n_authors in [%s, %s]...",
                min_authors, ("∞" if max_authors is None else max_authors))

    edges = fetch_coauthor_edges(
        con,
        year_from=year_from,
        year_to=year_to,
        min_authors=min_authors,
        max_authors=max_authors,
    )
    logger.info("Collected %d edges", len(edges))

    if edges.empty:
        logger.warning("No edges after filtering. Exiting.")
        return

    # 3) Число коллаборационных публикаций по авторам (k_i) — тот же фильтр min/max authors и годы
    logger.info("Fetching author publication counts (k_i) with same n_authors filter...")
    author_pub_counts = fetch_author_pub_counts(
        con,
        year_from=year_from,
        year_to=year_to,
        min_authors=min_authors,
        max_authors=max_authors,
    )

    # 4) Год старта карьеры авторов (по всем публикациям)
    logger.info("Fetching author career start years (year_start)...")
    author_year_start = fetch_author_year_start(con)

    # 5) Граф и Leiden
    g, vertices = build_graph(edges)
    logger.info("Graph built: %d authors, %d edges", g.vcount(), g.ecount())

    clustering = run_leiden(g, resolution=resolution)
    logger.info(
        "Leiden detected %d communities (modularity=%.4f)",
        len(clustering),
        clustering.modularity,
    )

    # добавим community к vertices
    community_sizes = clustering.sizes()
    vertices = vertices.copy()
    vertices["community"] = clustering.membership
    vertices["community_size"] = [community_sizes[c] for c in clustering.membership]

    # CSV membership
    write_membership(vertices, clustering, communities_csv)

    # 6) Обогащение таблицы рёбер: communities, weight_pub, weight_frac, weight_ci
    logger.info("Enriching edges with communities and weights...")

    author_communities: Dict[str, int] = dict(zip(vertices["name"], vertices["community"]))

    edges_enriched = edges.copy()

    # weight_pub
    edges_enriched["weight_pub"] = edges_enriched["publications"].astype(int)
    edges_enriched.drop(columns=["publications"], inplace=True)

    # source/target community
    edges_enriched = edges_enriched.merge(
        vertices[["name", "community"]].rename(columns={"name": "source", "community": "source_community"}),
        on="source",
        how="left",
    )
    edges_enriched = edges_enriched.merge(
        vertices[["name", "community"]].rename(columns={"name": "target", "community": "target_community"}),
        on="target",
        how="left",
    )

    # k_i / k_j (same filter)
    edges_enriched = edges_enriched.merge(
        author_pub_counts.rename(columns={"author_id": "source", "k": "k_i"}),
        on="source",
        how="left",
    )
    edges_enriched = edges_enriched.merge(
        author_pub_counts.rename(columns={"author_id": "target", "k": "k_j"}),
        on="target",
        how="left",
    )

    # weight_ci
    edges_enriched["weight_ci"] = None
    mask = (
        edges_enriched["k_i"].notna()
        & edges_enriched["k_j"].notna()
        & (edges_enriched["k_i"] > 0)
        & (edges_enriched["k_j"] > 0)
    )
    edges_enriched.loc[mask, "weight_ci"] = (
        (edges_enriched.loc[mask, "weight_pub"].astype(float) ** 2)
        / (edges_enriched.loc[mask, "k_i"] * edges_enriched.loc[mask, "k_j"])
    )

    # порядок полей с весами: weight_pub, weight_frac, weight_ci
    cols_order = [
        "source",
        "target",
        "source_name",
        "target_name",
        "source_community",
        "target_community",
        "weight_pub",
        "weight_frac",
        "weight_ci",
        "k_i",
        "k_j",
    ]
    remaining_cols = [c for c in edges_enriched.columns if c not in cols_order]
    edges_enriched = edges_enriched[cols_order + remaining_cols]

    # 7) Таблица авторов + метрики
    logger.info("Building authors table with communities, metrics and year_start...")

    authors_table = author_pub_counts.rename(columns={"k": "k_collab"})

    authors_table = authors_table.merge(
        author_year_start,
        on="author_id",
        how="left",
    )

    authors_table = authors_table.merge(
        vertices[["name", "label", "community"]],
        left_on="author_id",
        right_on="name",
        how="left",
    ).drop(columns=["name"])

    authors_table = authors_table.rename(columns={"label": "author_name"})
    authors_table["community"] = authors_table["community"].fillna(-1).astype(int)

    # обновим author_communities (включая -1)
    for _, row in authors_table.iterrows():
        author_communities[str(row["author_id"])] = int(row["community"])

    metrics_df = compute_author_metrics(edges_enriched, author_communities)
    authors_table = authors_table.merge(metrics_df, on="author_id", how="left")

    for col in ["degree", "degree_in_community", "phi", "phi_in_community"]:
        authors_table[col] = authors_table[col].fillna(0).astype(int)

    author_cols_order = [
        "author_id",
        "author_name",
        "year_start",
        "community",
        "k_collab",
        "degree",
        "degree_in_community",
        "phi",
        "phi_in_community",
    ]
    remaining_author_cols = [c for c in authors_table.columns if c not in author_cols_order]
    authors_table = authors_table[author_cols_order + remaining_author_cols]

    # 8) Сохранение в DuckDB
    save_to_duckdb(output_db_path, edges_enriched, authors_table)


if __name__ == "__main__":
    main()