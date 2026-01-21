# -*- coding: utf-8 -*-
"""
assign_publications_to_teams_json.py

Назначение публикаций командам в 2 прохода + финальный скоринг метрикой S.

Определения:
  inter_phi(team, pub) = sum(phi_in_team[u] for u in pub ∩ team)
  score_phi_share      = inter_phi / team_phi_sum

Метрика для финального ранжирования:
  S = (score_phi_share ** GAMMA) * (inter_phi / (inter_phi + C_SAT))

Процедура:
  PASS 1 (санитарный фильтр):
    - inter_authors >= MIN_TEAM_AUTHORS_IN_PUB
    - score_phi_share >= SCORE_MIN

  PASS 2 (окрестность по S):
    - для оставшихся считаем S, находим S_max
    - оставляем кандидатов с S >= BETA_S * S_max
    - (опционально) TOP_K лучших по S

Выход JSONL (по 1 строке на публикацию):
{
  "eid": "...",
  "doi": "...",
  "publication_year": 2024,
  "n_authors": 9,
  "max_S": 0.3571,
  "assigned": [
    {
      "team_id": 1,
      "score_phi_share": 0.6957,
      "sat_phi": 0.6154,
      "S": 0.3571,
      "s_rel": 1.0,
      "inter_authors": 9,
      "inter_phi": 32,
      "team_phi_sum": 46
    },
    ...
  ]
}

Входы:
  - DuckDB: publications(eid, doi, publication_year, authors JSON)
  - JSONL команд: team_id, team_members/core_members, phi map (phi_in_team/phi_by_member/phi_members/phi)

Примечание:
  - В DuckDB используется json_array_length(p.authors) для фильтра по размеру массива authors.
  - После нормализации author_id дополнительно проверяем MIN_AUTHORS/MAX_AUTHORS по уникальным id.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import duckdb
from tqdm import tqdm


# ===== SETTINGS =====

PUB_DB_PATH = Path("data/topics.duckdb")                 # где таблица publications
TEAMS_JSONL = Path("output_core/teams_GLOBAL_10pub.jsonl")      # JSONL с командами
OUT_JSONL = Path("team_publications.jsonl")             # результат

# Фильтры для воспроизводимости по годам и числу авторов публикации
YEAR_FROM: Optional[int] = 2023
YEAR_TO: Optional[int] = 2025

MIN_AUTHORS: Optional[int] = 2
MAX_AUTHORS: Optional[int] = 10  # None если без верхнего ограничения

# Минимальное требование: сколько авторов команды должно быть в публикации
MIN_TEAM_AUTHORS_IN_PUB = 3

# PASS 1
SCORE_MIN = 0.50  # score_phi_share >= SCORE_MIN

# Финальная метрика S
C_SAT = 20.0      # c в sat = inter_phi/(inter_phi+c)
GAMMA = 1.5       # степень для score_phi_share

# PASS 2: оставляем окрестность по S
BETA_S = 0.95     # keep if S >= BETA_S * S_max
TOP_K: Optional[int] = None  # например 3; None => без ограничения

# производительность
PUB_BATCH_SIZE = 50_000


# ===== HELPERS =====

def _extract_author_id(a: Any) -> Optional[str]:
    """
    Под типичную структуру authors:
    - dict с 'id' (предпочтительно), иначе строки, иначе fullname/name (fallback).
    """
    if a is None:
        return None
    if isinstance(a, str):
        s = a.strip()
        return s if s else None
    if isinstance(a, dict):
        v = a.get("id") or a.get("author_id") or a.get("_id")
        if v:
            s = str(v).strip()
            return s if s else None
        v = a.get("fullname") or a.get("name")
        if v:
            s = str(v).strip().lower()
            return s if s else None
    return None


def load_jsonl_teams(path: Path) -> List[Dict[str, Any]]:
    """
    Грузим команды и phi-карту.
    Ожидаемые поля:
      - team_id
      - team_members (или core_members)
      - phi map: phi_in_team / phi_by_member / phi_members / phi
    """
    teams: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading teams JSONL"):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)

            team_id = rec.get("team_id")
            if team_id is None:
                continue

            members = rec.get("team_members") or rec.get("core_members") or []
            members = [str(x) for x in members if x is not None]
            if not members:
                continue

            phi_map = None
            for k in ("phi_in_team", "phi_by_member", "phi_members", "phi"):
                if isinstance(rec.get(k), dict):
                    phi_map = rec[k]
                    break
            if not isinstance(phi_map, dict):
                continue

            phi_norm: Dict[str, int] = {}
            for a, p in phi_map.items():
                if a is None:
                    continue
                try:
                    phi_norm[str(a)] = int(p)
                except Exception:
                    try:
                        phi_norm[str(a)] = int(float(p))
                    except Exception:
                        phi_norm[str(a)] = 0

            # phi_sum — сумма по всем ключам карты (даже если кто-то не в team_members)
            # Чтобы быть строгими, оставляем только участников команды:
            phi_sum = 0
            mem_set = set(members)
            for a in mem_set:
                phi_sum += max(0, int(phi_norm.get(a, 0) or 0))

            if phi_sum <= 0:
                continue

            teams.append(
                {
                    "team_id": int(team_id) if str(team_id).isdigit() else team_id,
                    "members": mem_set,
                    "phi": phi_norm,
                    "phi_sum": int(phi_sum),
                }
            )
    return teams


def build_author_to_teams(teams: List[Dict[str, Any]]) -> Dict[str, List[int]]:
    idx: Dict[str, List[int]] = {}
    for ti, t in enumerate(teams):
        for a in t["members"]:
            idx.setdefault(a, []).append(ti)
    return idx


def score_pub_team(pub_authors: set[str], team: Dict[str, Any]) -> Tuple[float, int, int]:
    """
    returns:
      score_phi_share, inter_authors_count, inter_phi_sum
    """
    inter = pub_authors & team["members"]
    if not inter:
        return 0.0, 0, 0

    phi_map = team["phi"]
    inter_phi = 0
    for a in inter:
        inter_phi += int(phi_map.get(a, 0) or 0)

    denom = int(team["phi_sum"])
    if denom <= 0:
        return 0.0, len(inter), inter_phi

    return inter_phi / denom, len(inter), inter_phi


def compute_S(score_phi_share: float, inter_phi: int) -> Tuple[float, float]:
    """
    returns:
      (S, sat_phi)
    sat_phi = inter_phi/(inter_phi + C_SAT)
    S = (score_phi_share ** GAMMA) * sat_phi
    """
    if inter_phi <= 0:
        return 0.0, 0.0
    sat = float(inter_phi) / (float(inter_phi) + float(C_SAT)) if C_SAT > 0 else 1.0
    s = (float(score_phi_share) ** float(GAMMA)) * sat if score_phi_share > 0 else 0.0
    return s, sat


def _build_where_and_params() -> Tuple[str, List[Any]]:
    where = ["p.authors IS NOT NULL"]
    params: List[Any] = []

    if YEAR_FROM is not None:
        where.append("p.publication_year >= ?")
        params.append(int(YEAR_FROM))
    if YEAR_TO is not None:
        where.append("p.publication_year <= ?")
        params.append(int(YEAR_TO))

    # предварительный фильтр по длине JSON-массива authors
    if MIN_AUTHORS is not None:
        where.append("json_array_length(p.authors) >= ?")
        params.append(int(MIN_AUTHORS))
    if MAX_AUTHORS is not None:
        where.append("json_array_length(p.authors) <= ?")
        params.append(int(MAX_AUTHORS))

    return " AND ".join(where), params


# ===== MAIN =====

def main():
    teams = load_jsonl_teams(TEAMS_JSONL)
    print(f"Teams loaded: {len(teams):,}")
    if not teams:
        print("No teams with phi maps found. Exiting.")
        return

    author2teams = build_author_to_teams(teams)

    con = duckdb.connect(str(PUB_DB_PATH), read_only=True)
    where_sql, params = _build_where_and_params()

    total_pubs = con.execute(
        f"SELECT COUNT(*) FROM publications p WHERE {where_sql}",
        params
    ).fetchone()[0]
    print(f"Publications to scan: {total_pubs:,}")

    offset = 0

    with OUT_JSONL.open("w", encoding="utf-8") as fout, tqdm(total=total_pubs, desc="Scanning publications") as pbar:
        while True:
            batch = con.execute(
                f"""
                SELECT p.eid, p.doi, p.publication_year, p.authors
                FROM publications p
                WHERE {where_sql}
                ORDER BY p.eid
                LIMIT {PUB_BATCH_SIZE} OFFSET {offset}
                """,
                params
            ).fetchall()

            if not batch:
                break

            for eid, doi, pub_year, authors_json in batch:
                if authors_json is None:
                    pbar.update(1)
                    continue

                # authors может прийти уже как объект или как строка
                if isinstance(authors_json, str):
                    try:
                        authors_list = json.loads(authors_json)
                    except Exception:
                        pbar.update(1)
                        continue
                else:
                    authors_list = authors_json

                if not isinstance(authors_list, list):
                    pbar.update(1)
                    continue

                pub_authors = set()
                for a in authors_list:
                    aid = _extract_author_id(a)
                    if aid:
                        pub_authors.add(aid)

                # уточняющий фильтр по уникальным id авторов
                n_auth = len(pub_authors)
                if n_auth < 2:
                    pbar.update(1)
                    continue
                if MIN_AUTHORS is not None and n_auth < MIN_AUTHORS:
                    pbar.update(1)
                    continue
                if MAX_AUTHORS is not None and n_auth > MAX_AUTHORS:
                    pbar.update(1)
                    continue

                # кандидаты команд (по авторам публикации)
                cand_team_idxs = set()
                for a in pub_authors:
                    for ti in author2teams.get(a, []):
                        cand_team_idxs.add(ti)

                if not cand_team_idxs:
                    pbar.update(1)
                    continue

                # PASS 1: score>=SCORE_MIN и inter_authors>=MIN_TEAM_AUTHORS_IN_PUB
                cand = []
                for ti in cand_team_idxs:
                    team = teams[ti]
                    score, inter_cnt, inter_phi = score_pub_team(pub_authors, team)

                    if inter_cnt < MIN_TEAM_AUTHORS_IN_PUB:
                        continue
                    if score < SCORE_MIN:
                        continue

                    S, sat = compute_S(score, inter_phi)
                    cand.append((ti, score, inter_cnt, inter_phi, S, sat))

                if not cand:
                    pbar.update(1)
                    continue

                # PASS 2: окрестность по S вокруг максимума
                S_max = max(x[4] for x in cand)
                if S_max <= 0:
                    pbar.update(1)
                    continue

                thr = BETA_S * S_max
                cand2 = [x for x in cand if x[4] >= thr]

                # сортировка по S, затем по inter_phi, затем по inter_authors, затем по team_id
                cand2.sort(key=lambda x: (-x[4], -x[3], -x[2], teams[x[0]]["team_id"]))

                if TOP_K is not None and TOP_K > 0:
                    cand2 = cand2[: int(TOP_K)]

                assigned = []
                for ti, score, inter_cnt, inter_phi, S, sat in cand2:
                    t = teams[ti]
                    s_rel = float(S) / float(S_max) if S_max > 0 else 0.0
                    assigned.append(
                        {
                            "team_id": t["team_id"],
                            "score_phi_share": float(score),
                            "sat_phi": float(sat),
                            "S": float(S),
                            "s_rel": float(s_rel),
                            "inter_authors": int(inter_cnt),
                            "inter_phi": int(inter_phi),
                            "team_phi_sum": int(t["phi_sum"]),
                        }
                    )

                out_rec = {
                    "eid": str(eid),
                    "doi": (str(doi) if doi is not None else None),
                    "publication_year": int(pub_year) if pub_year is not None else None,
                    "n_authors": int(n_auth),
                    "max_S": float(S_max),
                    "assigned": assigned,
                }

                fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
                pbar.update(1)

            offset += PUB_BATCH_SIZE

    con.close()
    print(f"Saved: {OUT_JSONL}")


if __name__ == "__main__":
    main()