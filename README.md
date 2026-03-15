# Solar Panel Spacing Calculator

A web application that calculates inter-row spacing for solar arrays to
minimise self-shading, based on site location, panel geometry, and an
allowable shading tolerance.

## What it calculates

Using the winter solstice (worst-case shading day) at a user-specified time
or solar noon:

| Output | Description |
|--------|-------------|
| **Back-to-front spacing** | Minimum gap between the rear edge of one row and the front edge of the next |
| **Front-to-front spacing** | Total row pitch (back-to-front + panel ground projection) |
| **Solar elevation / azimuth** | Sun position at the chosen time and location |

**Formula:**
```
back_to_front = (1 − shading_fraction) × panel_height / tan(solar_elevation)
front_to_front = back_to_front + panel_length × cos(tilt)
```

## Features

- Interactive Leaflet map — click to set site coordinates
- Solar noon or specific time-of-day input
- Configurable shading tolerance (0–100 % of rear panel height)
- Timezone-aware solar position via `pvlib`

## Running locally

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py
# Open http://localhost:5000
```

## Deploying to Railway

1. Fork or connect this repository in [Railway](https://railway.app).
2. Railway auto-detects Python via Nixpacks and uses the `Procfile`.
3. No environment variables required — the app reads `$PORT` automatically.

## License

MIT
