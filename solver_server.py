"""
Fantasy Lineup Optimizer — Python/PuLP solver server
Accepts POST /solve with JSON, returns optimal lineups.
Uses Integer Linear Programming (CBC via PuLP) — provably optimal.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pulp
import time
import traceback

app = Flask(__name__)
CORS(app)  # allow requests from any origin (the browser app)


def solve_lineup(players, cap, size, pos_c, team_c, exclude_lineups):
    """
    Solve one lineup via ILP.
    Returns list of player dicts, or None if infeasible.
    """
    prob = pulp.LpProblem("lineup", pulp.LpMaximize)

    # Binary variable per player-position entry
    uid_to_player = {p["uid"]: p for p in players}
    x = {p["uid"]: pulp.LpVariable(f"x_{i}", cat="Binary")
         for i, p in enumerate(players)}

    # ── Objective ─────────────────────────────────────────────────────────────
    prob += pulp.lpSum(p["pts"] * x[p["uid"]] for p in players)

    # ── Salary cap ────────────────────────────────────────────────────────────
    prob += pulp.lpSum(p["salary"] * x[p["uid"]] for p in players) <= cap

    # ── Roster size ───────────────────────────────────────────────────────────
    prob += pulp.lpSum(x[p["uid"]] for p in players) == size

    # ── Position slot minimums ────────────────────────────────────────────────
    for pos, req in pos_c.items():
        prob += pulp.lpSum(
            x[p["uid"]] for p in players if p["pos"] == pos
        ) >= req

    # ── Team constraints ──────────────────────────────────────────────────────
    for team, tc in team_c.items():
        expr = pulp.lpSum(
            x[p["uid"]] for p in players if p["team"] == team
        )
        mode = tc["mode"]
        val  = tc["val"]
        if mode == "exact":
            prob += expr == val
        elif mode == "min":
            prob += expr >= val
        elif mode == "max":
            prob += expr <= val

    # ── Name uniqueness (same player, multiple positions) ─────────────────────
    names = {}
    for p in players:
        names.setdefault(p["name"], []).append(p)
    for name, dupes in names.items():
        if len(dupes) > 1:
            prob += pulp.lpSum(x[p["uid"]] for p in dupes) <= 1

    # ── Exclude previous lineups ──────────────────────────────────────────────
    for excl_uids in exclude_lineups:
        in_pool = [uid for uid in excl_uids if uid in x]
        if in_pool:
            prob += pulp.lpSum(x[uid] for uid in in_pool) <= len(in_pool) - 1

    # ── Solve ─────────────────────────────────────────────────────────────────
    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=30))

    status = pulp.LpStatus[prob.status]
    if status != "Optimal":
        return None

    return [p for p in players if pulp.value(x[p["uid"]]) is not None
            and pulp.value(x[p["uid"]]) > 0.5]


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "solver": "PuLP/CBC"})


@app.route("/solve", methods=["POST"])
def solve():
    t0 = time.time()
    try:
        body = request.get_json(force=True)

        # ── Parse request ─────────────────────────────────────────────────────
        players_raw = body.get("players", [])
        cap         = float(body.get("cap", 100000))
        size        = int(body.get("size", 9))
        pos_c       = {str(k): int(v) for k, v in body.get("posC", {}).items()}
        team_c_raw  = body.get("teamC", {})
        n_lineups   = min(int(body.get("n", 5)), 20)

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

        # ── Solve top N lineups ───────────────────────────────────────────────
        results   = []
        timings   = []
        timed_out = False

        for i in range(n_lineups):
            excl = [set(p["uid"] for p in lineup) for lineup in results]
            t1   = time.time()
            lineup = solve_lineup(players, cap, size, pos_c, team_c, excl)
            elapsed = (time.time() - t1) * 1000  # ms
            timings.append(round(elapsed))

            if lineup is None:
                break  # no more feasible lineups

            results.append(lineup)

        total_ms = round((time.time() - t0) * 1000)

        # Serialise lineups
        lineups_out = []
        for lineup in results:
            lineups_out.append([
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
            ])

        return jsonify({
            "lineups":  lineups_out,
            "timings":  timings,
            "totalMs":  total_ms,
            "timedOut": timed_out,
            "solver":   "PuLP/CBC",
        })

    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


if __name__ == "__main__":
    print("Fantasy Lineup Solver — listening on http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
