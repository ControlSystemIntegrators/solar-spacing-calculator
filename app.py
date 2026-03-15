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
    Returns (back_to_front_m, front_to_front_m).
    shading_frac = fraction of TOTAL SLOPE HEIGHT allowed in shadow (0–1).
    For multi-row arrays, panel_length is the full slope length of the array block.
    """
    height           = panel_length * sin(radians(tilt))
    projected_length = panel_length * cos(radians(tilt))
    back_to_front    = (1.0 - shading_frac) * height / tan(radians(solar_elevation))
    front_to_front   = back_to_front + projected_length
    return back_to_front, front_to_front


def bypass_loss(bottom_panel_shading_frac, n_bypass):
    """
    Returns the fraction of ONE PANEL's power lost due to bypass diode activation.

    Bypass diodes protect equal-height bands of the panel. Any shadow reaching
    a cell in a band activates that band's diode, losing 1/n_bypass of output.
    So even 1% shading triggers the first diode (1/n_bypass loss), and there is
    no ADDITIONAL penalty until shadow exceeds the next band boundary.

    bottom_panel_shading_frac: fraction of the BOTTOM PANEL's HEIGHT in shadow (0–1)
    n_bypass:                  number of bypass diode sections (typically 3)
    """
    if bottom_panel_shading_frac <= 0:
        return 0.0
    sections_activated = ceil(bottom_panel_shading_frac * n_bypass)
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
        panel_length = float(data["panel_length"])   # total slope length (all rows)
        tilt         = float(data["tilt"])
        shading_frac = float(data["shading"])        # fraction of total slope
        time_string  = data["time"]
        tz_string    = data["timezone"]
        array_rows   = int(data.get("array_rows", 1))

        # Bottom panel shading is the relevant quantity for bypass diode physics
        bottom_panel_frac = min(1.0, shading_frac * array_rows)

        elev, az, ts = get_solar_elevation(latitude, longitude, tz_string, time_string)
        b2f, f2f     = spacing_for_shading(shading_frac, panel_length, tilt, elev)

        slope_pct  = round(shading_frac * 100, 1)
        bottom_pct = round(bottom_panel_frac * 100, 1)

        if array_rows > 1:
            shading_note = (
                f"{slope_pct}% of total slope height may be shaded "
                f"({bottom_pct}% of the bottom panel row)"
            )
        else:
            shading_note = f"{slope_pct}% of the panel height may be shaded"

        return jsonify({
            "timestamp":                ts,
            "solar_elevation":          round(elev, 2),
            "solar_azimuth":            round(az,   2),
            "back_to_front":            round(b2f,  3),
            "front_to_front":           round(f2f,  3),
            "shading_note":             shading_note,
            "bottom_panel_shading_pct": bottom_pct,
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/optimize", methods=["POST"])
def optimize():
    """
    Sweeps bottom-panel shading tolerance from 0% to (n_bypass-1)/n_bypass × 100%
    and returns a cost-per-effective-kWp table.

    The sweep variable is the fraction of the BOTTOM PANEL's height that is shaded.
    This maps directly to bypass diode physics regardless of array row count.
    Total slope shading fraction = bottom_panel_shading / array_rows.

    Land cost is computed per individual panel:
      land_area_per_panel = front_to_front × one_panel_width / array_rows

    Power loss per panel is the average across the array:
      array_loss = bypass_loss(bottom_panel_frac) / array_rows
    (only the bottom row is shaded; upper rows are unaffected)
    """
    try:
        data         = request.get_json(force=True)
        latitude     = float(data["latitude"])
        longitude    = float(data["longitude"])
        panel_length = float(data["panel_length"])   # total slope length (all rows)
        tilt         = float(data["tilt"])
        time_string  = data["time"]
        tz_string    = data["timezone"]
        panel_wp     = float(data["panel_wp"])
        panel_cost   = float(data["panel_cost"])
        panel_width  = float(data["panel_width"])    # single panel's across-row width
        land_cost_m2 = float(data["land_cost_m2"])
        n_bypass     = int(data.get("n_bypass", 3))
        array_rows   = int(data.get("array_rows", 1))

        elev, az, ts = get_solar_elevation(latitude, longitude, tz_string, time_string)

        rows = []
        # Sweep bottom-panel shading from 0% to just below all sections being bypassed
        max_bottom_pct = int(((n_bypass - 1) / n_bypass) * 100)

        for bottom_pct in range(0, max_bottom_pct + 1):
            bottom_sf = bottom_pct / 100.0
            # Fraction of TOTAL slope height in shadow
            sf = bottom_sf / array_rows

            b2f, f2f = spacing_for_shading(sf, panel_length, tilt, elev)

            # Land cost per individual panel (f2f spans all rows, divide by array_rows)
            land_area     = f2f * panel_width / array_rows
            land_cost_pan = land_area * land_cost_m2
            total_cost    = panel_cost + land_cost_pan

            # Bypass loss for the bottom panel; average that loss across the full array
            panel_loss_frac  = bypass_loss(bottom_sf, n_bypass)
            array_loss_frac  = panel_loss_frac / array_rows
            sections_lost    = ceil(bottom_sf * n_bypass) if bottom_sf > 0 else 0

            effective_w  = panel_wp * (1.0 - array_loss_frac)
            cost_per_kwp = (total_cost / (effective_w / 1000)) if effective_w > 0 else None

            rows.append({
                "shading_pct":           bottom_pct,             # bottom panel %
                "slope_shading_pct":     round(sf * 100, 1),     # total slope %
                "back_to_front":         round(b2f, 2),
                "front_to_front":        round(f2f, 2),
                "land_area_m2":          round(land_area, 2),
                "land_cost":             round(land_cost_pan, 2),
                "total_cost":            round(total_cost, 2),
                "sections_lost":         sections_lost,
                "power_loss_pct":        round(array_loss_frac * 100, 1),
                "bottom_panel_loss_pct": round(panel_loss_frac * 100, 1),
                "effective_w":           round(effective_w, 1),
                "cost_per_kwp":          round(cost_per_kwp, 0) if cost_per_kwp else None,
            })

        valid   = [r for r in rows if r["cost_per_kwp"] is not None]
        optimal = min(valid, key=lambda r: r["cost_per_kwp"]) if valid else None

        # Zone boundaries are in bottom-panel % (0, 33, 66 for n_bypass=3)
        zone_starts = [int(i / n_bypass * 100) for i in range(n_bypass)]

        return jsonify({
            "timestamp":       ts,
            "solar_elevation": round(elev, 2),
            "rows":            rows,
            "optimal_pct":     optimal["shading_pct"] if optimal else None,
            "n_bypass":        n_bypass,
            "array_rows":      array_rows,
            "zone_starts":     zone_starts,
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
