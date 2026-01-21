"""
Step_2_Core_v2_mode3.py

Изменения (по сравнению с предыдущими версиями):
1) boundary_flag удалён полностью (и из JSONL, и из CSV).
2) density_* считаются из G_CI в process_one_layer() (быстро, без DuckDB) — для GLOBAL и COMMUNITIES.
3) MODE3 post-merge: плотности+phi пересчитываются через DuckDB ТОЛЬКО для реально объединённых команд (len(component)>1).
   Для не объединённых (len(component)==1) полностью переиспользуем raw-команду.
4) Вывод:
   - JSONL: финальные команды
   - team_members.csv: team_id, author_id, phi_in_team, status(core/periphery)
   - teams_summary.csv: team_id, mode, sizes, phi_mean_all/core/periphery, max_phi, density_unweighted, density_weighted_ci
5) Логирование:
   - VERBOSE_PER_LAYER=False по умолчанию: не печатаем per-layer.
   - В режимах с сообществами печатаем прогресс раз в LOG_EVERY_N_LAYERS.

MODE:
  "GLOBAL"
  "COMMUNITIES"
  "COMMUNITY_NEIGHBORHOOD" (MODE3: community + boundary Variant A + post-merge)
"""

import os
import json
import time
import csv
from collections import defaultdict
from datetime import datetime
from itertools import combinations

import duckdb
import networkx as nx
import pandas as pd


# ===== INPUT/OUTPUT =====

DB_PATH = "data/coauthor_graph_pub10.duckdb"

OUT_DIR = "output_core"
CORES_JSONL = os.path.join(OUT_DIR, "teams_GLOBAL_10pub.jsonl")
STATS_JSON = os.path.join(OUT_DIR, "core_stats_GLOBAL_10pub.json")

TEAM_MEMBERS_CSV = os.path.join(OUT_DIR, "team_members_GLOBAL_10pub.csv")
TEAMS_SUMMARY_CSV = os.path.join(OUT_DIR, "teams_summary_GLOBAL_10pub.csv")

# ===== MODE =====
MODE = "GLOBAL"  # "COMMUNITIES" | "GLOBAL" | "COMMUNITY_NEIGHBORHOOD"

# ===== ALGO PARAMS =====
MIN_COMMUNITY_SIZE = 4
PHI_MIN = 2
K_CORE_K = 2
MIN_CLIQUE_SIZE = 3

# strict merge threshold
JACCARD_EPS = 0.5

# optional year filter
ENABLE_YEAR_FILTER = False
YEAR_START_MAX = 2020

ENABLE_TEAM_EXPANSION = False  # rudiment, off

# ===== LOGGING =====
VERBOSE_PER_LAYER = False      # детальные логи на каждый слой
LOG_EVERY_N_LAYERS = 50        # прогресс раз в N слоёв (для community modes)

_T0 = time.time()


def log(msg: str):
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now} +{time.time()-_T0:7.2f}s] {msg}")


def compute_phi(neigh_dict):
    if not neigh_dict:
        return 0
    counts = sorted(neigh_dict.values(), reverse=True)
    phi = 0
    for idx, c in enumerate(counts, start=1):
        if c >= idx:
            phi = idx
    return phi


def density_unweighted(n: int, m: int) -> float:
    return (2.0 * m) / (n * (n - 1)) if n > 1 else 0.0


# ===== DUCKDB =====

def connect_duckdb(path: str, read_only: bool = True):
    return duckdb.connect(path, read_only=read_only)


def get_communities(con) -> list:
    df = con.execute(
        """
        SELECT community AS community_id,
               COUNT(*) AS size
        FROM coauthor_authors
        WHERE community >= 0
        GROUP BY community
        HAVING COUNT(*) >= ?
        ORDER BY community
        """,
        [MIN_COMMUNITY_SIZE],
    ).df()
    return df.to_dict("records")


def load_edges_global(con) -> pd.DataFrame:
    return con.execute(
        """
        SELECT source, target, weight_pub, weight_ci, source_community, target_community
        FROM coauthor_edges
        """
    ).df()


def load_edges_for_community(con, cid: int) -> pd.DataFrame:
    return con.execute(
        """
        SELECT source, target, weight_pub, weight_ci, source_community, target_community
        FROM coauthor_edges
        WHERE source_community = ? AND target_community = ?
        """,
        [cid, cid],
    ).df()


# ===== MODE3 neighborhood Variant A =====

def load_neighborhood_nodes_variant_A(con, cid: int) -> pd.DataFrame:
    return con.execute(
        """
        WITH comm_nodes AS (
            SELECT author_id
            FROM coauthor_authors
            WHERE community = ?
        ),
        boundary_nodes AS (
            SELECT source AS author_id
            FROM coauthor_edges
            WHERE source_community = ? OR target_community = ?
            UNION
            SELECT target AS author_id
            FROM coauthor_edges
            WHERE source_community = ? OR target_community = ?
        )
        SELECT DISTINCT author_id
        FROM (
            SELECT author_id FROM comm_nodes
            UNION ALL
            SELECT author_id FROM boundary_nodes
        ) t
        """,
        [cid, cid, cid, cid, cid],
    ).df()


def load_edges_for_nodeset(con, nodes: list) -> pd.DataFrame:
    return con.execute(
        """
        SELECT source, target, weight_pub, weight_ci, source_community, target_community
        FROM coauthor_edges
        WHERE source IN (SELECT UNNEST(?))
          AND target IN (SELECT UNNEST(?))
        """,
        [nodes, nodes],
    ).df()


def load_authors_for_nodeset(con, nodes: list) -> pd.DataFrame:
    return con.execute(
        """
        SELECT author_id, community, phi AS phi_val, year_start, k_collab
        FROM coauthor_authors
        WHERE author_id IN (SELECT UNNEST(?))
        """,
        [nodes],
    ).df()


# ===== BUILD CI GRAPH =====

def build_ci_graph(authors_df: pd.DataFrame, edges_df: pd.DataFrame):
    valid = set(
        a for a, phi in zip(authors_df["author_id"], authors_df["phi_val"])
        if phi is not None and phi >= PHI_MIN
    )

    if ENABLE_YEAR_FILTER and YEAR_START_MAX is not None:
        valid_year = set()
        for _, row in authors_df.iterrows():
            aid = row["author_id"]
            if aid not in valid:
                continue
            ys = row["year_start"]
            if ys is None:
                continue
            try:
                ys_int = int(ys)
            except Exception:
                continue
            if ys_int <= YEAR_START_MAX:
                valid_year.add(aid)
        valid = valid_year

    if not valid:
        return nx.Graph(), set()

    G = nx.Graph()
    authors_sub = authors_df[authors_df["author_id"].isin(valid)].copy()
    for _, row in authors_sub.iterrows():
        aid = row["author_id"]
        G.add_node(
            aid,
            community=row.get("community", None),
            year_start=row.get("year_start", None),
            phi_global=int(row["phi_val"]) if row["phi_val"] is not None else 0,
        )

    edges_sub = edges_df[edges_df["source"].isin(valid) & edges_df["target"].isin(valid)].copy()
    for _, row in edges_sub.iterrows():
        s = row["source"]
        t = row["target"]
        wci = row["weight_ci"]
        if wci is None or wci <= 0:
            continue
        kij = row["weight_pub"]
        G.add_edge(
            s,
            t,
            weight=float(wci),                   # для density_weighted_ci
            coauth=int(kij) if kij is not None else 0,  # для phi внутри команды
        )

    return G, valid


# ===== PHI within set =====

def phi_within_set(G: nx.Graph, nodes_set: set) -> dict:
    out = {}
    nodes = set(nodes_set)
    for u in nodes:
        neigh = {}
        for v in G.neighbors(u):
            if v in nodes:
                kij = int(G[u][v].get("coauth", 0) or 0)
                if kij > 0:
                    neigh[v] = kij
        out[u] = int(compute_phi(neigh))
    return out


# ===== weighted merge metric =====

def weighted_overlap_J(A_set, B_set, phiA: dict, phiB: dict):
    n1 = len(A_set)
    n2 = len(B_set)
    if n1 == 0 or n2 == 0:
        return 0.0
    inter = A_set & B_set
    if not inter:
        return 0.0

    onlyA = A_set - B_set
    onlyB = B_set - A_set

    denom_sizes = float(n1 + n2)

    x = 0.0
    for u in inter:
        v1 = float(phiA.get(u, 0))
        v2 = float(phiB.get(u, 0))
        x += (v1 * n1 + v2 * n2) / denom_sizes

    y = sum(float(phiA.get(u, 0)) for u in onlyA)
    z = sum(float(phiB.get(u, 0)) for u in onlyB)

    denom = x + y + z
    return (x / denom) if denom > 0 else 0.0


# ===== fast merge =====

def merge_groups_fast(G: nx.Graph, initial_sets: list, eps: float):
    import heapq

    groups = {}
    active = set()
    inv = defaultdict(set)

    next_id = 1
    for s in initial_sets:
        gid = next_id
        next_id += 1
        members = set(s)
        phi_map = phi_within_set(G, members)
        groups[gid] = {"id": gid, "members": members, "phi": phi_map}
        active.add(gid)
        for u in members:
            inv[u].add(gid)

    heap = []
    seen_pairs = set()

    def push_pair(i, j):
        if i == j or i not in active or j not in active:
            return
        if i > j:
            i, j = j, i

        keypair = (i, j)
        if keypair in seen_pairs:
            return
        seen_pairs.add(keypair)

        Gi, Gj = groups[i], groups[j]
        if not (Gi["members"] & Gj["members"]):
            return
        J = weighted_overlap_J(Gi["members"], Gj["members"], Gi["phi"], Gj["phi"])
        if J <= eps:  # strict >
            return

        union_size = len(Gi["members"] | Gj["members"])
        inter_size = len(Gi["members"] & Gj["members"])
        key = (-J, -inter_size, -union_size, Gi["id"], Gj["id"])
        heapq.heappush(heap, (key, i, j))

    for u, gids in inv.items():
        gids = sorted(gids)
        for a, b in combinations(gids, 2):
            push_pair(a, b)

    while heap:
        _, i, j = heapq.heappop(heap)
        if i not in active or j not in active:
            continue

        Gi, Gj = groups[i], groups[j]
        if not (Gi["members"] & Gj["members"]):
            continue

        J = weighted_overlap_J(Gi["members"], Gj["members"], Gi["phi"], Gj["phi"])
        if J <= eps:
            continue

        merged_members = Gi["members"] | Gj["members"]
        merged_phi = phi_within_set(G, merged_members)

        new_id = next_id
        next_id += 1
        groups[new_id] = {"id": new_id, "members": merged_members, "phi": merged_phi}
        active.add(new_id)

        active.remove(i)
        active.remove(j)

        affected = Gi["members"] | Gj["members"]
        for u in affected:
            s = inv[u]
            if i in s:
                s.remove(i)
            if j in s:
                s.remove(j)
            s.add(new_id)

        neighbors = set()
        for u in merged_members:
            neighbors.update(inv[u])
        neighbors.discard(new_id)
        neighbors = {k for k in neighbors if k in active}

        for k in sorted(neighbors):
            a, b = (new_id, k) if new_id < k else (k, new_id)
            if (a, b) in seen_pairs:
                seen_pairs.remove((a, b))
            push_pair(new_id, k)

    out = [{"members": groups[gid]["members"], "phi": groups[gid]["phi"]} for gid in sorted(active)]
    return out


# ===== split core/periphery by phi =====

def split_core_periphery_by_phi(members_set: set, phi_map: dict):
    if not members_set:
        return set(), set()

    vals = [int(phi_map.get(u, 0) or 0) for u in members_set]
    max_phi = max(vals) if vals else 0

    # Новое правило:
    # 1) при max_phi <= 2 периферии нет
    if max_phi <= 2:
        return set(members_set), set()

    # 2) иначе периферия: phi <= floor(max_phi/2)
    delta = max_phi // 2

    periph = {u for u in members_set if int(phi_map.get(u, 0) or 0) <= delta}
    core_inner = set(members_set) - periph
    return core_inner, periph


# ===== fast densities from local graph =====

def densities_from_graph(G_layer: nx.Graph, members: list):
    mem = list(members)
    n = len(mem)
    if n <= 1:
        return 0.0, 0.0

    H = G_layer.subgraph(mem)
    m = H.number_of_edges()
    d_unw = density_unweighted(n, m)

    if m == 0:
        return d_unw, 0.0

    w_sum = 0.0
    for _, _, ed in H.edges(data=True):
        w_sum += float(ed.get("weight", 0.0) or 0.0)
    d_w = w_sum / float(m)

    return d_unw, d_w


# ===== MODE3: post-merge components (connectivity) =====

def global_merge_teams_components(teams: list, eps: float):
    inv = defaultdict(set)
    for idx, t in enumerate(teams):
        for a in t["team_members"]:
            inv[a].add(idx)

    parent = list(range(len(teams)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    seen = set()
    for a, idxs in inv.items():
        idxs = sorted(idxs)
        for i, j in combinations(idxs, 2):
            if (i, j) in seen:
                continue
            seen.add((i, j))

            A = teams[i]
            B = teams[j]
            setA = set(A["team_members"])
            setB = set(B["team_members"])
            if not (setA & setB):
                continue

            phiA = {u: int(A["phi_in_team"].get(str(u), 0) or 0) for u in setA}
            phiB = {u: int(B["phi_in_team"].get(str(u), 0) or 0) for u in setB}
            J = weighted_overlap_J(setA, setB, phiA, phiB)
            if J > eps:
                union(i, j)

    comp = defaultdict(list)
    for i in range(len(teams)):
        comp[find(i)].append(i)
    return list(comp.values())


# ===== recompute phi + density from DuckDB (ONLY for merged MODE3 teams) =====

def recompute_phi_and_density_from_duckdb(con, members: list):
    if not members:
        return {}, 0.0, 0.0

    df = con.execute(
        """
        SELECT source, target, weight_pub, weight_ci
        FROM coauthor_edges
        WHERE source IN (SELECT UNNEST(?))
          AND target IN (SELECT UNNEST(?))
        """,
        [members, members],
    ).df()

    n = len(members)
    m = len(df)
    d_unw = density_unweighted(n, m)
    d_wci = float(df["weight_ci"].mean()) if m > 0 else 0.0

    G = nx.Graph()
    for u in members:
        G.add_node(u)

    for row in df.itertuples(index=False):
        s = row.source
        t = row.target
        kij = int(row.weight_pub) if row.weight_pub is not None else 0
        if kij <= 0:
            continue
        G.add_edge(s, t, coauth=kij)

    phi_map_int = phi_within_set(G, set(members))
    phi_map = {str(u): int(phi_map_int.get(u, 0) or 0) for u in members}

    return phi_map, d_unw, d_wci


# ===== CSV helpers =====

def phi_stats(phi_in_team: dict, team_members: list, core_members: list, periphery_members: list):
    def mean_of(ids):
        if not ids:
            return 0.0
        vals = [int(phi_in_team.get(str(a), 0) or 0) for a in ids]
        return float(sum(vals)) / float(len(vals))

    mean_all = mean_of(team_members)
    mean_core = mean_of(core_members)
    mean_per = mean_of(periphery_members)

    max_phi = 0
    for a in team_members:
        v = int(phi_in_team.get(str(a), 0) or 0)
        if v > max_phi:
            max_phi = v

    return mean_all, mean_core, mean_per, max_phi


def write_team_member_rows(csv_writer, team_id: int, core_members: list, periphery_members: list, phi_map: dict):
    for a in core_members:
        csv_writer.writerow([team_id, str(a), int(phi_map.get(str(a), 0) or 0), "core"])
    for a in periphery_members:
        csv_writer.writerow([team_id, str(a), int(phi_map.get(str(a), 0) or 0), "periphery"])


# ===== layer processing =====

def process_one_layer(layer_id, authors_df, edges_df, mode_tag, team_id_start=1, teams_collect=None):
    G_CI, _ = build_ci_graph(authors_df, edges_df)

    if VERBOSE_PER_LAYER:
        log(f"[layer={layer_id}] CI graph: n={G_CI.number_of_nodes()}, m={G_CI.number_of_edges()}")

    if G_CI.number_of_nodes() == 0 or G_CI.number_of_edges() == 0:
        return team_id_start, 0

    try:
        Hk = nx.k_core(G_CI, k=K_CORE_K)
    except nx.NetworkXError:
        Hk = nx.Graph()

    if VERBOSE_PER_LAYER:
        log(f"[layer={layer_id}] k-core(k={K_CORE_K}): n={Hk.number_of_nodes()}, m={Hk.number_of_edges()}")

    if Hk.number_of_nodes() == 0 or Hk.number_of_edges() == 0:
        return team_id_start, 0

    cliques = []
    for cset in nx.find_cliques(Hk):
        if len(cset) >= MIN_CLIQUE_SIZE:
            cliques.append(set(cset))

    if VERBOSE_PER_LAYER:
        log(f"[layer={layer_id}] cliques found: {len(cliques)}")

    if not cliques:
        return team_id_start, 0

    merged_groups = merge_groups_fast(G_CI, cliques, eps=JACCARD_EPS)

    if VERBOSE_PER_LAYER:
        log(f"[layer={layer_id}] cliques after phi-merge: {len(merged_groups)}")

    next_team_id = team_id_start
    written = 0

    for g in sorted(merged_groups, key=lambda x: (-len(x["members"]), sorted(x["members"]))):
        members = sorted(set(g["members"]))
        phi_map_int = dict(g["phi"])  # {u:int}
        phi_map = {str(u): int(phi_map_int.get(u, 0) or 0) for u in members}

        core_inner, periph_inner = split_core_periphery_by_phi(set(members), phi_map_int)

        core_members = sorted(core_inner)
        per_members = sorted(periph_inner)

        d_unw, d_wci = densities_from_graph(G_CI, members)

        rec = {
            "team_id": next_team_id,
            "mode": mode_tag,
            "core_members": core_members,
            "periphery_members": per_members,
            "team_members": members,
            "core_size": len(core_members),
            "periphery_size": len(per_members),
            "team_size": len(members),
            "phi_in_team": phi_map,
            "density_unweighted": d_unw,
            "density_weighted_ci": d_wci,
            "params": {
                "phi_min": PHI_MIN,
                "k_core_k": K_CORE_K,
                "min_clique_size": MIN_CLIQUE_SIZE,
                "phi_merge_eps_strict": JACCARD_EPS,
                "enable_year_filter": ENABLE_YEAR_FILTER,
                "year_start_max": YEAR_START_MAX if ENABLE_YEAR_FILTER else None,
                "enable_team_expansion": ENABLE_TEAM_EXPANSION,
            },
        }

        if teams_collect is not None:
            teams_collect.append(rec)

        next_team_id += 1
        written += 1

    return next_team_id, written


# ===== MAIN =====

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    con = connect_duckdb(DB_PATH, read_only=True)
    log(f"Connected: {DB_PATH}")

    total_layers = 0
    total_teams = 0
    team_id = 1

    teams_all = []
    final_records = []
    teams_summary_rows = []

    # CSV writers
    f_members = open(TEAM_MEMBERS_CSV, "w", encoding="utf-8", newline="")
    w_members = csv.writer(f_members)
    w_members.writerow(["team_id", "author_id", "phi_in_team", "status"])

    if MODE == "GLOBAL":
        log("=== GLOBAL mode ===")
        authors_df = con.execute(
            """
            SELECT author_id, community, phi AS phi_val, year_start, k_collab
            FROM coauthor_authors
            WHERE community >= 0
            """
        ).df()
        edges_df = load_edges_global(con)

        team_id, nteams = process_one_layer("GLOBAL", authors_df, edges_df, "GLOBAL", team_id, teams_all)
        total_layers = 1
        total_teams = nteams
        final_records = teams_all

    elif MODE == "COMMUNITIES":
        comms = get_communities(con)
        total = len(comms)
        log(f"COMMUNITIES to process (size ≥ {MIN_COMMUNITY_SIZE}): {total}")

        for i, c in enumerate(comms, start=1):
            cid = int(c["community_id"])
            authors_df = con.execute(
                """
                SELECT author_id, community, phi_in_community AS phi_val, year_start, k_collab
                FROM coauthor_authors
                WHERE community = ?
                """,
                [cid],
            ).df()
            edges_df = load_edges_for_community(con, cid)

            team_id, nteams = process_one_layer(cid, authors_df, edges_df, "COMMUNITIES", team_id, teams_all)
            total_layers += 1
            total_teams += nteams

            if (i % LOG_EVERY_N_LAYERS == 0) or (i == total):
                pct = 100.0 * i / total
                log(f"COMMUNITIES progress: {i}/{total} ({pct:.1f}%), teams so far={total_teams}")

        final_records = teams_all

    elif MODE == "COMMUNITY_NEIGHBORHOOD":
        comms = get_communities(con)
        total = len(comms)
        log(f"MODE3 communities to process (size ≥ {MIN_COMMUNITY_SIZE}): {total}")
        log("MODE3: Variant A neighborhood")

        # stage1 raw teams per layer
        for i, c in enumerate(comms, start=1):
            cid = int(c["community_id"])
            nodes_df = load_neighborhood_nodes_variant_A(con, cid)
            nodes = nodes_df["author_id"].astype(str).tolist()
            if len(nodes) < MIN_COMMUNITY_SIZE:
                continue

            authors_df = load_authors_for_nodeset(con, nodes)
            edges_df = load_edges_for_nodeset(con, nodes)

            team_id, nteams = process_one_layer(cid, authors_df, edges_df, "COMMUNITY_NEIGHBORHOOD", team_id, teams_all)
            total_layers += 1
            total_teams += nteams

            if (i % LOG_EVERY_N_LAYERS == 0) or (i == total):
                pct = 100.0 * i / total
                log(f"MODE3 progress: {i}/{total} ({pct:.1f}%), raw teams so far={len(teams_all)}")

        log(f"MODE3 stage1 done. Raw teams: {len(teams_all)}")

        # stage2 merge components
        comps = global_merge_teams_components(teams_all, eps=JACCARD_EPS)
        log(f"MODE3 stage2: components found: {len(comps)} (from {len(teams_all)} raw teams)")

        next_tid = 1
        merged_cnt = 0

        for comp_idxs in comps:
            if len(comp_idxs) == 1:
                raw = teams_all[comp_idxs[0]]
                rec = dict(raw)
                rec["team_id"] = next_tid
                final_records.append(rec)
            else:
                merged_cnt += 1
                members = set()
                for idx in comp_idxs:
                    members.update(teams_all[idx]["team_members"])
                members = sorted(set(members))

                phi_map, d_unw, d_wci = recompute_phi_and_density_from_duckdb(con, members)

                phi_map_int = {a: int(phi_map.get(str(a), 0) or 0) for a in members}
                core_inner, periph_inner = split_core_periphery_by_phi(set(members), phi_map_int)

                core_members = sorted(core_inner)
                per_members = sorted(periph_inner)

                rec = {
                    "team_id": next_tid,
                    "mode": "COMMUNITY_NEIGHBORHOOD",
                    "core_members": core_members,
                    "periphery_members": per_members,
                    "team_members": members,
                    "core_size": len(core_members),
                    "periphery_size": len(per_members),
                    "team_size": len(members),
                    "phi_in_team": {str(a): int(phi_map.get(str(a), 0) or 0) for a in members},
                    "density_unweighted": d_unw,
                    "density_weighted_ci": d_wci,
                    "params": {
                        "phi_min": PHI_MIN,
                        "k_core_k": K_CORE_K,
                        "min_clique_size": MIN_CLIQUE_SIZE,
                        "phi_merge_eps_strict": JACCARD_EPS,
                        "enable_year_filter": ENABLE_YEAR_FILTER,
                        "year_start_max": YEAR_START_MAX if ENABLE_YEAR_FILTER else None,
                        "enable_team_expansion": ENABLE_TEAM_EXPANSION,
                        "mode3_variant": "A",
                        "mode3_post_merge": True,
                    },
                }
                final_records.append(rec)

            next_tid += 1

        total_teams = len(final_records)
        log(f"MODE3 final teams: {total_teams}, merged components: {merged_cnt}")

    else:
        f_members.close()
        con.close()
        raise ValueError(f"Unknown MODE={MODE}")

    # ===== write JSONL + CSVs from final_records =====
    with open(CORES_JSONL, "w", encoding="utf-8") as fout:
        for rec in final_records:
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

            team_members = rec["team_members"]
            core_members = rec["core_members"]
            per_members = rec["periphery_members"]
            phi_map = rec["phi_in_team"]

            phi_mean_all, phi_mean_core, phi_mean_per, max_phi = phi_stats(
                phi_map, team_members, core_members, per_members
            )

            row = {
                "team_id": rec["team_id"],
                "mode": rec["mode"],
                "team_size": rec["team_size"],
                "core_size": rec["core_size"],
                "periphery_size": rec["periphery_size"],
                "phi_mean_all": phi_mean_all,
                "phi_mean_core": phi_mean_core,
                "phi_mean_periphery": phi_mean_per,
                "max_phi": max_phi,
                "density_unweighted": float(rec.get("density_unweighted", 0.0) or 0.0),
                "density_weighted_ci": float(rec.get("density_weighted_ci", 0.0) or 0.0),
            }
            teams_summary_rows.append(row)

            write_team_member_rows(w_members, rec["team_id"], core_members, per_members, phi_map)

    f_members.close()

    pd.DataFrame(teams_summary_rows).to_csv(TEAMS_SUMMARY_CSV, index=False, encoding="utf-8")
    log(f"Teams summary CSV → {TEAMS_SUMMARY_CSV}")
    log(f"Team members CSV → {TEAM_MEMBERS_CSV}")

    stats = {
        "mode": MODE,
        "layers_processed": total_layers,
        "total_teams": total_teams,
        "params": {
            "min_community_size": MIN_COMMUNITY_SIZE,
            "phi_min": PHI_MIN,
            "k_core_k": K_CORE_K,
            "min_clique_size": MIN_CLIQUE_SIZE,
            "phi_merge_eps_strict": JACCARD_EPS,
            "enable_year_filter": ENABLE_YEAR_FILTER,
            "year_start_max": YEAR_START_MAX if ENABLE_YEAR_FILTER else None,
            "enable_team_expansion": ENABLE_TEAM_EXPANSION,
            "verbose_per_layer": VERBOSE_PER_LAYER,
            "log_every_n_layers": LOG_EVERY_N_LAYERS if MODE != "GLOBAL" else None,
        },
    }
    with open(STATS_JSON, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    log(f"Done. Stats → {STATS_JSON}")
    log(f"Teams JSONL → {CORES_JSONL}")
    log(f"final teams found: {total_teams}")

    con.close()


if __name__ == "__main__":
    main()