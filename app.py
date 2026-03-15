from flask import Flask, render_template, request, jsonify
from datetime import datetime, timedelta
import pandas as pd
import pvlib
import pytz
from math import tan, radians, cos, sin

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/calculate", methods=["POST"])
def calculate():
    try:
        data = request.get_json(force=True)

        # Inputs
        latitude = float(data["latitude"])
        longitude = float(data["longitude"])
        panel_length = float(data["panel_length"])  # slope length (m)
        tilt = float(data["tilt"])                  # deg from horizontal
        azimuth = float(data["azimuth"])            # deg from North (180=South) - kept for future use
        shading_frac = float(data["shading"])       # 0..1 (e.g., 0.25)
        time_string = data["time"]                  # "solar" or "HH:MM"
        user_tz_string = data["timezone"]           # e.g., "America/New_York"

        # Derived: panel vertical height (m)
        height = panel_length * sin(radians(tilt))

        # Worst-case date for self-shading
        year = datetime.now().year
        date = datetime(year, 12, 21)

        # Use the user-provided time zone explicitly
        tz = pytz.timezone(user_tz_string)
        location = pvlib.location.Location(
            latitude=latitude,
            longitude=longitude,
            tz=user_tz_string
        )

        # Build a timezone-aware timestamp at the site
        if time_string.lower() == "solar":
            # Find solar noon by maximizing elevation that day (robust approach)
            times = pd.date_range(
                start=tz.localize(datetime(year, 12, 21, 0, 0, 0)),
                end=tz.localize(datetime(year, 12, 22, 0, 0, 0)),
                freq="1min",
                tz=tz
            )
            sp = location.get_solarposition(times)
            idx = sp["elevation"].idxmax()
            time = idx.to_pydatetime()
            solar_elevation = float(sp.loc[idx, "elevation"])
            solar_azimuth = float(sp.loc[idx, "azimuth"])
        else:
            hour, minute = map(int, time_string.split(":"))
            naive_time = datetime(year, 12, 21, hour, minute)
            time = tz.localize(naive_time)

            sp = location.get_solarposition([time])
            solar_elevation = float(sp["elevation"].iloc[0])
            solar_azimuth = float(sp["azimuth"].iloc[0])

        if solar_elevation <= 0:
            return jsonify({"error": "Sun is below the horizon at that time."}), 200

        # Back-to-front spacing (min spacing so only S fraction of rear panel height is shaded)
        # D = (1 - S) * H / tan(elevation)
        back_to_front = (1.0 - shading_frac) * height / tan(radians(solar_elevation))

        # Front-to-front spacing adds the projected panel length on the ground
        projected_length = panel_length * cos(radians(tilt))
        front_to_front = back_to_front + projected_length

        return jsonify({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
            "solar_elevation": round(solar_elevation, 2),
            "solar_azimuth": round(solar_azimuth, 2),
            "back_to_front": round(back_to_front, 3),
            "front_to_front": round(front_to_front, 3),
            "shading_note": f"{int(shading_frac * 100)}% of rear panel height may be shaded"
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
