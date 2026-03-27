# Fantasy Lineup Solver — Python Backend

ILP solver using PuLP/CBC. Finds provably optimal lineups in milliseconds.

## Performance vs JS branch-and-bound
- Contest 32660 (hard): **824ms** vs 25+ seconds in JS
- Typical contest: **~50ms** for 5 lineups

## Local setup
```bash
pip install -r requirements.txt
python server.py
# Server runs on http://localhost:5000
```

## Cloud deployment options

### Option A: Railway (easiest, free tier)
1. Push this folder to a GitHub repo
2. Go to railway.app → New Project → Deploy from GitHub
3. Select the repo — Railway auto-detects Dockerfile
4. Done. You get a public URL like `https://your-app.up.railway.app`

### Option B: Render (free tier, sleeps after 15min)
1. Push to GitHub
2. render.com → New Web Service → connect repo
3. Build command: `pip install -r requirements.txt`
4. Start command: `python server.py`

### Option C: Fly.io (free tier, always-on)
```bash
fly launch
fly deploy
```

### Option D: VPS (DigitalOcean $5/month, most control)
```bash
git clone your-repo
cd solver_server
pip install -r requirements.txt
# Run with gunicorn for production
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 server:app
```
Add nginx + systemd service for reliability.

## API

### POST /solve
```json
{
  "players": [
    {"name": "J.Smith - 1234", "pos": "2", "salary": 14500, "pts": 95.3, "team": "T1"},
    {"name": "B.Jones - 5678", "pos": "1", "salary": 8000,  "pts": 61.7, "team": null}
  ],
  "cap":    100000,
  "size":   9,
  "posC":   {"1": 2, "2": 4, "3": 2, "4": 1},
  "teamC":  {"T1": {"val": 8, "mode": "exact"}},
  "n":      5
}
```

Response:
```json
{
  "lineups":  [[...players...], ...],
  "timings":  [45, 62, 71, 88, 120],
  "totalMs":  386,
  "solver":   "PuLP/CBC"
}
```

### GET /health
Returns `{"status": "ok", "solver": "PuLP/CBC"}`
