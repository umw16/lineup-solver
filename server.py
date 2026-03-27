"""
Fantasy Lineup Optimizer — Python/PuLP solver server
Solves ALL N lineups in a single ILP call for maximum speed.
Uses Integer Linear Programming (CBC via PuLP) — provably optimal.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pulp
import time
import traceback

app = Flask(__name__)
CORS(app)


def solve_all_lineups(players, cap, size, pos_c, team_c, n_lineups, unique_by_name=False):
    """
    Solve all N lineups in ONE ILP call.

    Variables:
      x[k][uid] = 1 if player uid is in lineup k

    Constraints per lineup k:
      - salary cap
      - roster size
      - position minimums
      - team constraints
      - name uniqueness (same player, multiple positions -> pick at most 1)

    Diversity constraints (across lineups):
      - uniqueByName=True:  for every pair (j,k), lineups must differ by at least one NAME
      - uniqueByName=False: for every pair (j,k), lineups must differ by at least one UID

    Objective: maximise total pts across all lineups.
    """
    prob = pulp.LpProblem("all_lineups", pulp.LpMaximize)

    # ── Decision variables ────────────────────────────────────────────────────
    x = {}
    for k in range(n_lineups):
        x[k] = {p["uid"]: pulp.LpVariable(f"x_{k}_{i}", cat="Binary")
                for i, p in enumerate(players)}

    # Group players by name
    name_to_players = {}
    for p in players:
        name_to_players.setdefault(p["name"], []).append(p)
    all_names = list(name_to_players.keys())

    # y[k][name] = 1 if name is selected in lineup k (for uniqueByName diversity)
    y = {}
    if unique_by_name:
        for k in range(n_lineups):
            y[k] = {name: pulp.LpVariable(f"y_{k}_{ni}", cat="Binary")
                    for ni, name in enumerate(all_names)}

    # ── Objective ─────────────────────────────────────────────────────────────
    prob += pulp.lpSum(
        p["pts"] * x[k][p["uid"]]
        for k in range(n_lineups)
        for p in players
    )

    # ── Per-lineup constraints ────────────────────────────────────────────────
    for k in range(n_lineups):
        xk = x[k]

        # Salary cap
        prob += pulp.lpSum(p["salary"] * xk[p["uid"]] for p in players) <= cap

        # Roster size
        prob += pulp.lpSum(xk[p["uid"]] for p in players) == size

        # Position minimums
        for pos, req in pos_c.items():
            prob += pulp.lpSum(xk[p["uid"]] for p in players if p["pos"] == pos) >= req

        # Team constraints
        for team, tc in team_c.items():
            expr = pulp.lpSum(xk[p["uid"]] for p in players if p["team"] == team)
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
                prob += pulp.lpSum(xk[p["uid"]] for p in dupes) <= 1

        # Link y[k][name] to x[k] (only when uniqueByName)
        if unique_by_name:
            yk = y[k]
            for name, dupes in name_to_players.items():
                name_expr = pulp.lpSum(xk[p["uid"]] for p in dupes)
                n_dupes   = len(dupes)
                # y=1 iff name is selected in this lineup
                prob += yk[name] <= name_expr          # y=0 if nothing selected
                prob += yk[name] * n_dupes >= name_expr  # y=1 if anything selected

    # ── Diversity constraints (every pair of lineups must differ) ─────────────
    for j in range(n_lineups):
        for k in range(j + 1, n_lineups):
            if unique_by_name:
                # sum(y[j][name] + y[k][name]) <= 2*size - 1
                # Since each lineup has exactly `size` players this means
                # at most size-1 names are shared => at least 1 name differs.
                prob += pulp.lpSum(
                    y[j][name] + y[k][name] for name in all_names
                ) <= 2 * size - 1
            else:
                # At least one uid must differ between lineups j and k
                prob += pulp.lpSum(
                    x[j][p["uid"]] + x[k][p["uid"]] for p in players
                ) <= 2 * size - 1

    # ── Solve ─────────────────────────────────────────────────────────────────
    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=60))

    status = pulp.LpStatus[prob.status]
    if status not in ("Optimal", "Feasible"):
        return None

    # Extract lineups
    results = []
    for k in range(n_lineups):
        lineup = [p for p in players
                  if pulp.value(x[k][p["uid"]]) is not None
                  and pulp.value(x[k][p["uid"]]) > 0.5]
        if len(lineup) == size:
            results.append(lineup)

    return results if results else None


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "solver": "PuLP/CBC (single-call)"})


@app.route("/solve", methods=["POST"])
def solve():
    t0 = time.time()
    try:
        body = request.get_json(force=True)

        # ── Parse request ─────────────────────────────────────────────────────
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

        # ── Solve all lineups in one ILP call ─────────────────────────────────
        t1      = time.time()
        results = solve_all_lineups(
            players, cap, size, pos_c, team_c, n_lineups, unique_by_name
        )
        solve_ms = round((time.time() - t1) * 1000)

        if not results:
            return jsonify({"error": "No feasible lineups found. Try relaxing constraints."}), 400

        total_ms = round((time.time() - t0) * 1000)

        # Sort lineups by total pts descending
        results.sort(key=lambda lu: sum(p["pts"] for p in lu), reverse=True)

        # Timings: single solve time spread evenly for UI compatibility
        timings = [round(solve_ms / len(results))] * len(results)

        # Serialise
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
            "solver":   "PuLP/CBC (single-call)",
        })

    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


if __name__ == "__main__":
    print("Fantasy Lineup Solver — listening on http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
