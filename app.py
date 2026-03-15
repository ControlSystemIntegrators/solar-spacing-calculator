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

    # Find solar noon regardless of mode (needed for "window" and "solar")
    day_times = pd.date_range(
        start=tz.localize(datetime(year, 12, 21, 0, 0, 0)),
        end=tz.localize(datetime(year, 12, 22, 0, 0, 0)),
        freq="1min",
        tz=tz,
    )
    day_sp   = location.get_solarposition(day_times)
    noon_idx = day_sp["elevation"].idxmax()
    noon_t   = noon_idx.to_pydatetime()

    if time_string.lower() == "solar":
        t    = noon_t
        elev = float(day_sp.loc[noon_idx, "elevation"])
        az   = float(day_sp.loc[noon_idx, "azimuth"])
    elif time_string.lower().startswith("window:"):
        from datetime import timedelta
        hours_offset = float(time_string.split(":")[1])
        t  = noon_t - timedelta(hours=hours_offset)
        sp = location.get_solarposition([t])
        elev = float(sp["elevation"].iloc[0])
        az   = float(sp["azimuth"].iloc[0])
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
    2-D sweep: n_rows (1–max_rows) × bottom-panel shading tolerance (0 to (n-1)/n).

    Cost model:
      hardware_per_panel = panel_cost
                         + rail_cost_per_row                (incremental rail per row)
                         + foundation_cost_per_col / n_rows (pile cost amortised)
      land_per_panel     = f2f × panel_width / n_rows       (constant with n_rows at 0 shading)
      total_per_panel    = hardware + land

    Power loss is bottom-row only (assumes bottom row on separate MPPT):
      array_loss = bypass_loss(bottom_panel_frac) / n_rows

    Returns a summary (best shading tolerance per row count) and detail rows for
    the globally optimal n_rows.
    """
    try:
        data                  = request.get_json(force=True)
        latitude              = float(data["latitude"])
        longitude             = float(data["longitude"])
        panel_slope_m         = float(data["panel_slope_m"])        # single-panel slope dim
        tilt                  = float(data["tilt"])
        time_string           = data["time"]
        tz_string             = data["timezone"]
        panel_wp              = float(data["panel_wp"])
        panel_cost            = float(data["panel_cost"])            # hardware only, per panel
        foundation_cost_col   = float(data["foundation_cost_per_col"])  # per pile position
        rail_cost_per_row     = float(data["rail_cost_per_row"])     # incremental rail per row
        panel_width           = float(data["panel_width"])           # across-row, single panel
        land_cost_m2          = float(data["land_cost_m2"])
        n_bypass              = int(data.get("n_bypass", 3))
        max_rows              = int(data.get("max_rows", 6))

        elev, az, ts = get_solar_elevation(latitude, longitude, tz_string, time_string)

        max_bottom_pct = int(((n_bypass - 1) / n_bypass) * 100)
        zone_starts    = [int(i / n_bypass * 100) for i in range(n_bypass)]

        summary = []          # best result per n_rows
        best_detail = []      # full shading sweep for the globally optimal n_rows
        global_best = None

        for n_rows in range(1, max_rows + 1):
            slope_length         = panel_slope_m * n_rows
            hardware_per_panel   = (panel_cost
                                    + rail_cost_per_row
                                    + foundation_cost_col / n_rows)

            best_for_n = None
            sweep      = []

            for bottom_pct in range(0, max_bottom_pct + 1):
                bottom_sf = bottom_pct / 100.0
                sf        = bottom_sf / n_rows     # fraction of total slope

                b2f, f2f = spacing_for_shading(sf, slope_length, tilt, elev)

                land_area     = f2f * panel_width / n_rows
                land_cost_pan = land_area * land_cost_m2
                total_cost    = hardware_per_panel + land_cost_pan

                panel_loss_frac = bypass_loss(bottom_sf, n_bypass)
                array_loss_frac = panel_loss_frac / n_rows
                sections_lost   = ceil(bottom_sf * n_bypass) if bottom_sf > 0 else 0

                effective_w  = panel_wp * (1.0 - array_loss_frac)
                cost_per_kwp = (total_cost / (effective_w / 1000)) if effective_w > 0 else None

                row = {
                    "shading_pct":       bottom_pct,
                    "slope_shading_pct": round(sf * 100, 1),
                    "back_to_front":     round(b2f, 2),
                    "front_to_front":    round(f2f, 2),
                    "land_area_m2":      round(land_area, 2),
                    "hardware_cost":     round(hardware_per_panel, 2),
                    "land_cost":         round(land_cost_pan, 2),
                    "total_cost":        round(total_cost, 2),
                    "sections_lost":     sections_lost,
                    "power_loss_pct":    round(array_loss_frac * 100, 1),
                    "effective_w":       round(effective_w, 1),
                    "cost_per_kwp":      round(cost_per_kwp, 0) if cost_per_kwp else None,
                }
                sweep.append(row)

                if cost_per_kwp and (best_for_n is None
                                     or cost_per_kwp < best_for_n["cost_per_kwp"]):
                    best_for_n = row

            if best_for_n:
                summary.append({
                    "n_rows":              n_rows,
                    "hardware_per_panel":  round(hardware_per_panel, 2),
                    "optimal_shading_pct": best_for_n["shading_pct"],
                    "front_to_front":      best_for_n["front_to_front"],
                    "land_cost":           best_for_n["land_cost"],
                    "total_cost":          best_for_n["total_cost"],
                    "power_loss_pct":      best_for_n["power_loss_pct"],
                    "effective_w":         best_for_n["effective_w"],
                    "cost_per_kwp":        best_for_n["cost_per_kwp"],
                })
                if global_best is None or best_for_n["cost_per_kwp"] < global_best["cost_per_kwp"]:
                    global_best    = best_for_n
                    global_best_n  = n_rows
                    best_detail    = sweep   # keep the full sweep for this n_rows

        # Filter detail to every-5% + zone boundaries + optimal row (keeps payload lean)
        important = {global_best["shading_pct"]} | set(zone_starts) | {z+1 for z in zone_starts}
        detail_filtered = [r for r in best_detail
                           if r["shading_pct"] % 5 == 0
                           or r["shading_pct"] in important]

        return jsonify({
            "timestamp":       ts,
            "solar_elevation": round(elev, 2),
            "summary":         summary,
            "detail":          detail_filtered,
            "global_opt_rows": global_best_n if global_best else None,
            "global_opt_pct":  global_best["shading_pct"] if global_best else None,
            "n_bypass":        n_bypass,
            "zone_starts":     zone_starts,
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
