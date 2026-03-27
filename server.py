"""
Fantasy Lineup Optimizer — Python/PuLP solver server
Parallel ILP: solves lineups in batches using ThreadPoolExecutor.
Each thread runs its own CBC instance simultaneously.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pulp
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

app = Flask(__name__)
CORS(app)

# Number of parallel CBC solvers — Railway gives 1-2 vCPUs, 4 threads is safe
MAX_WORKERS = 4


def build_and_solve(players, cap, size, pos_c, team_c, excl_sets, unique_by_name, round_i):
    """
    Build and solve a single lineup ILP.
    excl_sets: list of sets (names or uids) from already-found lineups to exclude.
    Returns (lineup, elapsed_ms) or (None, elapsed_ms).
    """
    name_to_players = {}
    for p in players:
        name_to_players.setdefault(p["name"], []).append(p)

    prob = pulp.LpProblem(f"lineup_{round_i}", pulp.LpMaximize)

    x = {p["uid"]: pulp.LpVariable(f"x_{round_i}_{i}", cat="Binary")
         for i, p in enumerate(players)}

    # Objective
    prob += pulp.lpSum(p["pts"] * x[p["uid"]] for p in players)

    # Salary cap
    prob += pulp.lpSum(p["salary"] * x[p["uid"]] for p in players) <= cap

    # Roster size
    prob += pulp.lpSum(x[p["uid"]] for p in players) == size

    # Position minimums
    for pos, req in pos_c.items():
        prob += pulp.lpSum(x[p["uid"]] for p in players if p["pos"] == pos) >= req

    # Team constraints
    for team, tc in team_c.items():
        expr = pulp.lpSum(x[p["uid"]] for p in players if p["team"] == team)
        mode, val = tc["mode"], tc["val"]
        if mode == "exact":
            prob += expr == val
        elif mode == "min":
            prob += expr >= val
        elif mode == "max":
            prob += expr <= val

    # Same player in multiple positions -> pick at most 1
    for name, dupes in name_to_players.items():
        if len(dupes) > 1:
            prob += pulp.lpSum(x[p["uid"]] for p in dupes) <= 1

    # Exclusion constraints
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

    status = pulp.LpStatus[prob.status]
    if status != "Optimal":
        return None, elapsed

    lineup = [p for p in players
              if pulp.value(x[p["uid"]]) is not None
              and pulp.value(x[p["uid"]]) > 0.5]

    if len(lineup) != size:
        return None, elapsed

    return lineup, elapsed


def solve_lineups(players, cap, size, pos_c, team_c, n_lineups, unique_by_name=False):
    """
    Solve N lineups in parallel batches.
    Batch size = MAX_WORKERS. Each batch uses exclusions from all confirmed
    lineups found so far. Within a batch, lineups are solved simultaneously.
    """
    results  = []
    timings  = []
    excl_sets = []

    batch_size = MAX_WORKERS

    i = 0
    while i < n_lineups:
        # How many to solve in this batch
        batch_count = min(batch_size, n_lineups - i)

        # Each lineup in the batch excludes all confirmed results so far
        # (lineups within the same batch may overlap — we deduplicate after)
        futures = {}
        with ThreadPoolExecutor(max_workers=batch_count) as executor:
            for b in range(batch_count):
                future = executor.submit(
                    build_and_solve,
                    players, cap, size, pos_c, team_c,
                    list(excl_sets),   # snapshot of confirmed exclusions
                    unique_by_name,
                    i + b
                )
                futures[future] = i + b

        # Collect results in order
        batch_results = {}
        for future in as_completed(futures):
            idx = futures[future]
            try:
                lineup, elapsed = future.result()
                batch_results[idx] = (lineup, elapsed)
            except Exception:
                batch_results[idx] = (None, 0)

        # Process batch results in order, deduplicating
        any_found = False
        for b in range(batch_count):
            idx = i + b
            lineup, elapsed = batch_results.get(idx, (None, 0))
            timings.append(elapsed)

            if lineup is None:
                continue

            # Deduplicate against already-confirmed results
            if unique_by_name:
                key = frozenset(p["name"] for p in lineup)
            else:
                key = frozenset(p["uid"] for p in lineup)

            already = any(
                (frozenset(p["name"] for p in r) if unique_by_name
                 else frozenset(p["uid"] for p in r)) == key
                for r in results
            )
            if not already:
                results.append(lineup)
                if unique_by_name:
                    excl_sets.append(set(p["name"] for p in lineup))
                else:
                    excl_sets.append(set(p["uid"] for p in lineup))
                any_found = True

        if not any_found:
            break  # no new lineups found in this batch — stop

        i += batch_count

    return results, timings


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "solver": "PuLP/CBC (parallel)"})


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

        # Normalise team constraints
        team_c = {}
        for team, tc in team_c_raw.items():
            val = tc.get("val")
            if val is not None and val != "":
                team_c[team] = {"val": int(val), "mode": tc.get("mode", "max")}

        # Parse players
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
            [
                {
                    "name":   p["name"],
                    "uid":    p["uid"],
                    "pos":    p["pos"],
                    "salary": p["salary"],
                    "pts":    p["pts"],
                    "team":   p["team"],
                    "value":  p["value"],
                }
                for p in lineup
            ]
            for lineup in results
        ]

        return jsonify({
            "lineups":  lineups_out,
            "timings":  timings,
            "totalMs":  total_ms,
            "timedOut": False,
            "solver":   "PuLP/CBC (parallel)",
        })

    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


if __name__ == "__main__":
    print("Fantasy Lineup Solver — listening on http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
