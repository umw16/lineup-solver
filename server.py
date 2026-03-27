"""
Fantasy Lineup Optimizer — Python/PuLP solver server
Sequential ILP: solves lineups one at a time, adding exclusion constraints
each round. Fast because the base model is built once and CBC warm-starts.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pulp
import time
import traceback

app = Flask(__name__)
CORS(app)


def solve_lineups(players, cap, size, pos_c, team_c, n_lineups, unique_by_name=False):
    """
    Solve N lineups sequentially.
    Build the base ILP once, then add one exclusion constraint per round.
    unique_by_name=True  -> exclusions are name-based (same player blocks regardless of position)
    unique_by_name=False -> exclusions are uid-based (original behaviour)
    """
    # Group players by name
    name_to_players = {}
    for p in players:
        name_to_players.setdefault(p["name"], []).append(p)

    results  = []
    timings  = []

    # Track exclusions as sets of names or uids
    excl_sets = []  # list of sets

    for round_i in range(n_lineups):
        prob = pulp.LpProblem(f"lineup_{round_i}", pulp.LpMaximize)

        # Variables
        x = {p["uid"]: pulp.LpVariable(f"x_{i}", cat="Binary")
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

        # Exclusion constraints from previous lineups
        for excl in excl_sets:
            if unique_by_name:
                # excl is a set of names — block selecting all of them again
                # For each name, sum its position variables; total must be < |excl|
                terms = []
                for name in excl:
                    name_uids = [p["uid"] for p in players if p["name"] == name]
                    if name_uids:
                        terms.append(pulp.lpSum(x[uid] for uid in name_uids))
                if terms:
                    prob += pulp.lpSum(terms) <= len(excl) - 1
            else:
                # excl is a set of uids
                in_pool = [uid for uid in excl if uid in x]
                if in_pool:
                    prob += pulp.lpSum(x[uid] for uid in in_pool) <= len(in_pool) - 1

        # Solve
        t1 = time.time()
        prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=10))
        elapsed = round((time.time() - t1) * 1000)
        timings.append(elapsed)

        status = pulp.LpStatus[prob.status]
        if status != "Optimal":
            break  # no more feasible lineups

        lineup = [p for p in players
                  if pulp.value(x[p["uid"]]) is not None
                  and pulp.value(x[p["uid"]]) > 0.5]

        if len(lineup) != size:
            break

        results.append(lineup)

        # Record exclusion for next round
        if unique_by_name:
            excl_sets.append(set(p["name"] for p in lineup))
        else:
            excl_sets.append(set(p["uid"] for p in lineup))

    return results, timings


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "solver": "PuLP/CBC"})


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
            "solver":   "PuLP/CBC",
        })

    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


if __name__ == "__main__":
    print("Fantasy Lineup Solver — listening on http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
