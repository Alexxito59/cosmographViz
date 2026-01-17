#!/usr/bin/env python3
"""
Build a weighted co-authorship graph from DuckDB and run Leiden community detection.

- Each pair of co-authors in a publication receives weight = 1 / (num_authors - 1).
- We aggregate weights across all publications to get edge weights for the graph.
- Optional CSV export with author-to-community mapping; graph is kept in memory only.

python coauthor_graph --db data/topics.duckdb --min-weight 0.1 --resolution 1.0 --communities-csv coauthor_communities.csv

"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Tuple

import igraph as ig
import pandas as pd

from utils.db import connect

logger = logging.getLogger("coauthor_graph")


def fetch_coauthor_edges(con, min_weight: float) -> pd.DataFrame:
    """
    Return aggregated co-author edges with weights summed across publications.
    """
    query = """
    WITH expanded AS (
        SELECT
            p.eid,
            CAST(a.key AS INTEGER) + 1 AS author_idx,
            COALESCE(a.value->>'id', LOWER(a.value->>'fullname'), LOWER(a.value->>'name')) AS author_id,
            COALESCE(a.value->>'fullname', a.value->>'name') AS author_name,
            COUNT(*) OVER (PARTITION BY p.eid) AS n_authors
        FROM publications p, json_each(p.authors) a
        WHERE p.authors IS NOT NULL
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
        WHERE e1.author_id IS NOT NULL AND e2.author_id IS NOT NULL AND e1.n_authors > 1
    )
    SELECT
        a_id AS source,
        b_id AS target,
        MIN(a_name) AS source_name,
        MIN(b_name) AS target_name,
        COUNT(*) AS publications,
        SUM(1.0 / (n_authors - 1)) AS weight
    FROM pairs
    GROUP BY 1, 2
    HAVING SUM(1.0 / (n_authors - 1)) >= ?
    ORDER BY weight DESC
    """
    return con.execute(query, [min_weight]).df()


def build_vertices(edges: pd.DataFrame) -> pd.DataFrame:
    vertices = pd.concat(
        [
            edges[["source", "source_name"]].rename(columns={"source": "name", "source_name": "label"}),
            edges[["target", "target_name"]].rename(columns={"target": "name", "target_name": "label"}),
        ],
        ignore_index=True,
    )
    vertices = vertices.drop_duplicates(subset=["name"]).reset_index(drop=True)
    return vertices


def build_graph(edges: pd.DataFrame) -> Tuple[ig.Graph, pd.DataFrame]:
    vertices = build_vertices(edges)
    vertices = vertices.reset_index(drop=True)
    idx_map = {name: idx for idx, name in enumerate(vertices["name"])}

    edge_idx = edges.assign(
        source_idx=edges["source"].map(idx_map), target_idx=edges["target"].map(idx_map)
    )
    tuples = list(zip(edge_idx["source_idx"], edge_idx["target_idx"]))

    g = ig.Graph(
        n=len(vertices),
        edges=tuples,
        directed=False,
    )
    g.es["weight"] = edge_idx["weight"].tolist()
    g.vs["name"] = vertices["name"].tolist()
    g.vs["label"] = vertices["label"].tolist()
    return g, vertices


def run_leiden(g: ig.Graph, resolution: float) -> ig.clustering.VertexClustering:
    return g.community_leiden(weights=g.es["weight"], resolution=resolution)


def write_membership(vertices: pd.DataFrame, clustering: ig.clustering.VertexClustering, path: Path) -> None:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Build weighted co-author graph and Leiden clusters.")
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("data/topics.duckdb"),
        help="Path to DuckDB database file (default: data/topics.duckdb).",
    )
    parser.add_argument("--min-weight", type=float, default=0.0, help="Drop edges with aggregated weight below this.")
    parser.add_argument("--resolution", type=float, default=1.0, help="Leiden resolution parameter.")
    parser.add_argument(
        "--communities-csv", type=Path, default=None, help="Optional path to save author -> community mapping."
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    con = connect(args.db)

    logger.info("Building co-author edges from %s", args.db)
    edges = fetch_coauthor_edges(con, min_weight=args.min_weight)
    logger.info("Collected %d edges", len(edges))

    g, vertices = build_graph(edges)
    logger.info("Graph built: %d authors, %d edges", g.vcount(), g.ecount())

    clustering = run_leiden(g, resolution=args.resolution)
    logger.info("Leiden detected %d communities (modularity=%.4f)", len(clustering), clustering.modularity)

    if args.communities_csv:
        write_membership(vertices, clustering, args.communities_csv)


if __name__ == "__main__":
    main()
