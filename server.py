"""
Fantasy Lineup Optimizer — Python/PuLP solver server
Strategy: solve first batch of MAX_WORKERS lineups in parallel,
then fill remaining slots sequentially (each sees all prior exclusions).
This gets the speed benefit of parallelism while guaranteeing 20 unique lineups.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pulp
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

app = Flask(__name__)
CORS(app)

MAX_WORKERS = 4


def build_and_solve(players, cap, size, pos_c, team_c, excl_sets, unique_by_name, round_i):
    """Build and solve a single lineup ILP. Returns (lineup, elapsed_ms)."""
    name_to_players = {}
    for p in players:
        name_to_players.setdefault(p["name"], []).append(p)

    prob = pulp.LpProblem(f"lineup_{round_i}", pulp.LpMaximize)

    x = {p["uid"]: pulp.LpVariable(f"x_{round_i}_{i}", cat="Binary")
         for i, p in enumerate(players)}

    prob += pulp.lpSum(p["pts"] * x[p["uid"]] for p in players)
    prob += pulp.lpSum(p["salary"] * x[p["uid"]] for p in players) <= cap
    prob += pulp.lpSum(x[p["uid"]] for p in players) == size

    for pos, req in pos_c.items():
        prob += pulp.lpSum(x[p["uid"]] for p in players if p["pos"] == pos) >= req

    for team, tc in team_c.items():
        expr = pulp.lpSum(x[p["uid"]] for p in players if p["team"] == team)
        mode, val = tc["mode"], tc["val"]
        if mode == "exact":
            prob += expr == val
        elif mode == "min":
            prob += expr >= val
        elif mode == "max":
            prob += expr <= val

    for name, dupes in name_to_players.items():
        if len(dupes) > 1:
            prob += pulp.lpSum(x[p["uid"]] for p in dupes) <= 1

    for excl in excl_sets:
        if unique_by_name:
            terms = []
            for name in excl:
                name_uids = [p["uid"] for p in players if p["name"] == name]
                if name_uids:
                    terms.append(pulp.lpSum(x[uid] for uid in name_uids))
            if terms:
                prob += pulp.lpSum(terms) <= len(excl) - 1
        else:
            in_pool = [uid for uid in excl if uid in x]
            if in_pool:
                prob += pulp.lpSum(x[uid] for uid in in_pool) <= len(in_pool) - 1

    t1 = time.time()
    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=10))
    elapsed = round((time.time() - t1) * 1000)

    if pulp.LpStatus[prob.status] != "Optimal":
        return None, elapsed

    lineup = [p for p in players
              if pulp.value(x[p["uid"]]) is not None
              and pulp.value(x[p["uid"]]) > 0.5]

    return (lineup if len(lineup) == size else None), elapsed


def lineup_key(lineup, unique_by_name):
    if unique_by_name:
        return frozenset(p["name"] for p in lineup)
    return frozenset(p["uid"] for p in lineup)


def solve_lineups(players, cap, size, pos_c, team_c, n_lineups, unique_by_name=False):
    results   = []
    timings   = []
    seen_keys = set()
    excl_sets = []  # grows as unique lineups are confirmed

    round_i = 0

    while len(results) < n_lineups:
        remaining = n_lineups - len(results)

        if len(results) == 0 and remaining >= MAX_WORKERS:
            # ── First batch: parallel, all use empty exclusion set ────────────
            batch_size = MAX_WORKERS
            futures = {}
            with ThreadPoolExecutor(max_workers=batch_size) as executor:
                for b in range(batch_size):
                    future = executor.submit(
                        build_and_solve,
                        players, cap, size, pos_c, team_c,
                        [],   # no exclusions yet for first batch
                        unique_by_name,
                        round_i + b
                    )
                    futures[future] = round_i + b

            # Collect in submission order
            ordered = {}
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    lineup, elapsed = future.result()
                except Exception:
                    lineup, elapsed = None, 0
                ordered[idx] = (lineup, elapsed)

            # Add unique lineups in order, updating excl_sets as we go
            for b in range(batch_size):
                idx = round_i + b
                lineup, elapsed = ordered.get(idx, (None, 0))
                timings.append(elapsed)
                if lineup is None:
                    continue
                key = lineup_key(lineup, unique_by_name)
                if key not in seen_keys:
                    seen_keys.add(key)
                    results.append(lineup)
                    if unique_by_name:
                        excl_sets.append(set(p["name"] for p in lineup))
                    else:
                        excl_sets.append(set(p["uid"] for p in lineup))

            round_i += batch_size

        else:
            # ── Sequential: each lineup sees all confirmed exclusions ──────────
            lineup, elapsed = build_and_solve(
                players, cap, size, pos_c, team_c,
                excl_sets, unique_by_name, round_i
            )
            timings.append(elapsed)
            round_i += 1

            if lineup is None:
                break  # no more feasible lineups

            key = lineup_key(lineup, unique_by_name)
            if key not in seen_keys:
                seen_keys.add(key)
                results.append(lineup)
                if unique_by_name:
                    excl_sets.append(set(p["name"] for p in lineup))
                else:
                    excl_sets.append(set(p["uid"] for p in lineup))
            # if duplicate (shouldn't happen sequentially), just continue

    return results, timings


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "solver": "PuLP/CBC (parallel+sequential)"})


@app.route("/solve", methods=["POST"])
def solve():
    t0 = time.time()
    try:
        body = request.get_json(force=True)

        players_raw    = body.get("players", [])
        cap            = float(body.get("cap", 100000))
        size           = int(body.get("size", 9))
        pos_c          = {str(k): int(v) for k, v in body.get("posC", {}).items()}
        team_c_raw     = body.get("teamC", {})
        n_lineups      = min(int(body.get("n", 5)), 20)
        unique_by_name = bool(body.get("uniqueByName", False))

        team_c = {}
        for team, tc in team_c_raw.items():
            val = tc.get("val")
            if val is not None and val != "":
                team_c[team] = {"val": int(val), "mode": tc.get("mode", "max")}

        players = []
        for p in players_raw:
            name   = str(p["name"]).strip()
            pos    = str(p["pos"]).strip()
            salary = float(p["salary"])
            pts    = float(p["pts"])
            team   = str(p.get("team", "") or "").strip().upper()
            if team in ("XX", ""):
                team = None
            players.append({
                "name":   name,
                "uid":    name + "|" + pos,
                "pos":    pos,
                "salary": salary,
                "pts":    pts,
                "team":   team,
                "value":  pts / salary * 1000 if salary > 0 else 0,
            })

        if not players:
            return jsonify({"error": "No players provided"}), 400

        results, timings = solve_lineups(
            players, cap, size, pos_c, team_c, n_lineups, unique_by_name
        )

        if not results:
            return jsonify({"error": "No feasible lineups found. Try relaxing constraints."}), 400

        total_ms = round((time.time() - t0) * 1000)

        lineups_out = [
            [{"name": p["name"], "uid": p["uid"], "pos": p["pos"],
              "salary": p["salary"], "pts": p["pts"],
              "team": p["team"], "value": p["value"]}
             for p in lineup]
            for lineup in results
        ]

        return jsonify({
            "lineups":  lineups_out,
            "timings":  timings,
            "totalMs":  total_ms,
            "timedOut": False,
            "solver":   "PuLP/CBC (parallel+sequential)",
        })

    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


if __name__ == "__main__":
    print("Fantasy Lineup Solver — listening on http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
