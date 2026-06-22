"""
Flask API Server
=================
Serves the frontend (static/) and exposes REST endpoints
that call ds1_pipeline.py and ds2_pipeline.py on demand.

Run:
    cd backend/
    python server.py
"""

import os, sys, base64
from io import BytesIO
from pathlib import Path

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# ── path setup so sibling modules resolve ─────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── output dirs ────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent.parent
OUTPUTS_DS1 = BASE_DIR / "outputs" / "ds1"
OUTPUTS_DS2 = BASE_DIR / "outputs" / "ds2"
STATIC_DIR  = BASE_DIR / "frontend"

OUTPUTS_DS1.mkdir(parents=True, exist_ok=True)
OUTPUTS_DS2.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, static_folder=str(STATIC_DIR))
CORS(app)

# ── Global state ───────────────────────────────────────────────────────────────
STATE = {}


def img_to_b64(path: Path) -> str:
    """Read a saved PNG and return base64 string."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def collect_ds1_images():
    """Return dict of filename → base64 for every ds1 output PNG."""
    images = {}
    for p in sorted(OUTPUTS_DS1.glob("*.png")):
        images[p.name] = img_to_b64(p)
    return images


def collect_ds2_images():
    images = {}
    for p in sorted(OUTPUTS_DS2.glob("*.png")):
        images[p.name] = img_to_b64(p)
    return images


# ══════════════════════════════════════════════════════════════════════════════
# TRAIN ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/train/ds1", methods=["POST"])
def train_ds1():
    try:
        import ds1_pipeline as ds1
        model_artifacts = ds1.run(return_model=True)
        STATE["ds1"] = model_artifacts

        # Gather classifier accuracy from saved pipeline result
        # (re-run is cheap enough for demo; for production cache separately)
        images = collect_ds1_images()
        return jsonify({"status": "success", "images": images,
                        "best_classifier": model_artifacts["best_name"]})
    except Exception as e:
        import traceback
        return jsonify({"status": "error", "message": str(e),
                        "trace": traceback.format_exc()}), 500


@app.route("/api/train/ds2", methods=["POST"])
def train_ds2():
    try:
        import ds2_pipeline as ds2
        results = ds2.run()
        STATE["ds2"] = results

        images = collect_ds2_images()
        return jsonify({"status": "success", "images": images,
                        "results": {k: {m: round(v, 4) for m, v in v2.items()}
                                    for k, v2 in results.items()}})
    except Exception as e:
        import traceback
        return jsonify({"status": "error", "message": str(e),
                        "trace": traceback.format_exc()}), 500


# ══════════════════════════════════════════════════════════════════════════════
# RECOMMEND
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/recommend", methods=["POST"])
def recommend():
    if "ds1" not in STATE:
        return jsonify({"error": "DS1 model not trained yet. Click 'Train DS1' first."}), 400

    data = request.json
    art  = STATE["ds1"]

    enc      = art["encoders"]
    scaler   = art["scaler"]
    selector = art["selector"]
    clf      = art["best_clf"]

    input_map = {
        "buying":   data.get("buying",   "med"),
        "maint":    data.get("maint",    "med"),
        "doors":    data.get("doors",    "4"),
        "persons":  data.get("persons",  "4"),
        "lug_boot": data.get("lug_boot", "med"),
        "safety":   data.get("safety",   "med"),
    }

    row = []
    for col, val in input_map.items():
        le = enc[col]
        row.append(int(le.transform([val])[0]) if val in le.classes_ else 0)

    X_in       = np.array(row).reshape(1, -1)
    X_in_sc    = scaler.transform(X_in)
    X_in_sel   = selector.transform(X_in)
    pred       = clf.predict(X_in_sel)[0]
    pred_label = enc["class"].inverse_transform([pred])[0]

    descriptions = {
        "unacc": "❌ Unacceptable – Not recommended",
        "acc":   "✅ Acceptable – Decent choice",
        "good":  "👍 Good – Recommended",
        "vgood": "⭐ Very Good – Highly recommended",
    }

    return jsonify({
        "prediction":  pred_label,
        "description": descriptions.get(pred_label, pred_label),
        "model_used":  art["best_name"],
        "inputs":      input_map,
    })


# ══════════════════════════════════════════════════════════════════════════════
# STATUS
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/status", methods=["GET"])
def status():
    return jsonify({"ds1_trained": "ds1" in STATE, "ds2_trained": "ds2" in STATE})


# ══════════════════════════════════════════════════════════════════════════════
# SERVE FRONTEND
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve(path):
    if path and (STATIC_DIR / path).exists():
        return send_from_directory(str(STATIC_DIR), path)
    return send_from_directory(str(STATIC_DIR), "index.html")


if __name__ == "__main__":
    print("🚗  AutoSense server starting on http://localhost:5000")
    app.run(debug=True, port=5000)