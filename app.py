from flask import Flask, render_template, request, jsonify
from datetime import datetime
import pandas as pd
import pvlib
import pytz
from math import tan, radians, cos, sin, ceil

app = Flask(__name__)


def get_solar_elevation(latitude, longitude, user_tz_string, time_string):
    """
    Returns (solar_elevation_deg, solar_azimuth_deg, timestamp_str) for the
    winter solstice at the given location and time. Raises ValueError if the
    sun is below the horizon.
    """
    year = datetime.now().year
    tz = pytz.timezone(user_tz_string)
    location = pvlib.location.Location(
        latitude=latitude, longitude=longitude, tz=user_tz_string
    )

    if time_string.lower() == "solar":
        times = pd.date_range(
            start=tz.localize(datetime(year, 12, 21, 0, 0, 0)),
            end=tz.localize(datetime(year, 12, 22, 0, 0, 0)),
            freq="1min",
            tz=tz,
        )
        sp = location.get_solarposition(times)
        idx = sp["elevation"].idxmax()
        t = idx.to_pydatetime()
        elev = float(sp.loc[idx, "elevation"])
        az   = float(sp.loc[idx, "azimuth"])
    else:
        hour, minute = map(int, time_string.split(":"))
        t = tz.localize(datetime(year, 12, 21, hour, minute))
        sp = location.get_solarposition([t])
        elev = float(sp["elevation"].iloc[0])
        az   = float(sp["azimuth"].iloc[0])

    if elev <= 0:
        raise ValueError("Sun is below the horizon at that time.")

    return elev, az, t.strftime("%Y-%m-%d %H:%M:%S %Z")


def spacing_for_shading(shading_frac, panel_length, tilt, solar_elevation):
    """
    Returns (back_to_front_m, front_to_front_m) for a given shading fraction.
    shading_frac = 0 means no shading allowed (full shadow clearance).
    """
    height           = panel_length * sin(radians(tilt))
    projected_length = panel_length * cos(radians(tilt))
    back_to_front    = (1.0 - shading_frac) * height / tan(radians(solar_elevation))
    front_to_front   = back_to_front + projected_length
    return back_to_front, front_to_front


def bypass_loss(shading_frac, n_bypass):
    """
    Returns the fraction of panel power lost due to bypass diode activation.

    Bypass diodes protect equal-height bands of the panel. Any shadow reaching
    a cell in a band activates that band's diode, losing 1/n_bypass of output.
    So even 1% shading triggers the first diode (1/n_bypass loss), and there is
    no ADDITIONAL penalty until shadow exceeds the next band boundary.

    shading_frac: fraction of rear panel HEIGHT in shadow (0–1)
    n_bypass:     number of bypass diode sections (typically 3)
    """
    if shading_frac <= 0:
        return 0.0
    sections_activated = ceil(shading_frac * n_bypass)
    return min(sections_activated, n_bypass) / n_bypass


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/calculate", methods=["POST"])
def calculate():
    try:
        data         = request.get_json(force=True)
        latitude     = float(data["latitude"])
        longitude    = float(data["longitude"])
        panel_length = float(data["panel_length"])
        tilt         = float(data["tilt"])
        shading_frac = float(data["shading"])
        time_string  = data["time"]
        tz_string    = data["timezone"]

        elev, az, ts = get_solar_elevation(latitude, longitude, tz_string, time_string)
        b2f, f2f     = spacing_for_shading(shading_frac, panel_length, tilt, elev)

        return jsonify({
            "timestamp":      ts,
            "solar_elevation": round(elev, 2),
            "solar_azimuth":   round(az,   2),
            "back_to_front":   round(b2f,  3),
            "front_to_front":  round(f2f,  3),
            "shading_note":    f"{int(shading_frac * 100)}% of rear panel height may be shaded",
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/optimize", methods=["POST"])
def optimize():
    """
    Sweeps shading tolerance from 0% to (n_bypass-1)/n_bypass in 1% steps
    and returns a cost-per-effective-kWp table so the user can find the
    spacing that minimises combined panel + land cost.

    Key insight: bypass diodes create a STEP function for power loss.
    Allowing any shadow up to 1/n_bypass of panel height costs the same
    power as allowing exactly 1/n_bypass — so there is a free spacing gain
    within each bypass zone.
    """
    try:
        data         = request.get_json(force=True)
        latitude     = float(data["latitude"])
        longitude    = float(data["longitude"])
        panel_length = float(data["panel_length"])
        tilt         = float(data["tilt"])
        time_string  = data["time"]
        tz_string    = data["timezone"]
        panel_wp     = float(data["panel_wp"])       # rated Watts per panel
        panel_cost   = float(data["panel_cost"])     # $ per panel
        panel_width  = float(data["panel_width"])    # m (for land area calculation)
        land_cost_m2 = float(data["land_cost_m2"])   # $/m²
        n_bypass     = int(data.get("n_bypass", 3))  # bypass diode sections

        elev, az, ts = get_solar_elevation(latitude, longitude, tz_string, time_string)

        rows = []
        # Evaluate at every 1% from 0% to (n_bypass-1)/n_bypass (above that,
        # all sections are bypassed and effective power approaches zero)
        max_shading_pct = int(((n_bypass - 1) / n_bypass) * 100)

        for pct in range(0, max_shading_pct + 1):
            sf = pct / 100.0

            b2f, f2f = spacing_for_shading(sf, panel_length, tilt, elev)

            land_area     = f2f * panel_width            # m² per panel
            land_cost_pan = land_area * land_cost_m2     # $ land per panel
            total_cost    = panel_cost + land_cost_pan   # $ per panel

            loss_frac        = bypass_loss(sf, n_bypass)
            effective_w      = panel_wp * (1.0 - loss_frac)
            sections_lost    = int(loss_frac * n_bypass)

            cost_per_kwp = (total_cost / (effective_w / 1000)) if effective_w > 0 else None

            rows.append({
                "shading_pct":      pct,
                "back_to_front":    round(b2f, 2),
                "front_to_front":   round(f2f, 2),
                "land_area_m2":     round(land_area, 2),
                "land_cost":        round(land_cost_pan, 2),
                "total_cost":       round(total_cost, 2),
                "sections_lost":    sections_lost,
                "power_loss_pct":   round(loss_frac * 100, 1),
                "effective_w":      round(effective_w, 1),
                "cost_per_kwp":     round(cost_per_kwp, 0) if cost_per_kwp else None,
            })

        # Find optimal: minimum cost/kWp among rows with positive effective power
        valid   = [r for r in rows if r["cost_per_kwp"] is not None]
        optimal = min(valid, key=lambda r: r["cost_per_kwp"]) if valid else None

        # Identify the bypass zone boundaries for annotation
        zone_starts = [int(i / n_bypass * 100) for i in range(n_bypass)]

        return jsonify({
            "timestamp":    ts,
            "solar_elevation": round(elev, 2),
            "rows":         rows,
            "optimal_pct":  optimal["shading_pct"] if optimal else None,
            "n_bypass":     n_bypass,
            "zone_starts":  zone_starts,
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
