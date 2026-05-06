"""
app.py
------
Flask REST API for the Cost-of-Living dashboard.

Endpoints
---------
GET  /city/<city_name>   → full cost breakdown + lifestyle mode for that city
POST /recommend          → top-5 cities matching your lifestyle preferences

Environment variables (see .env.example):
    FLASK_DEBUG   false | true   (default: false)
    PORT          integer        (default: 5000)
"""

import os
from flask import Flask, jsonify, request
from flask_cors import CORS

from main import main

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})


# ── City lookup ───────────────────────────────────────────────────────────────

@app.route("/city/<city_name>", methods=["GET"])
def get_city_data(city_name: str):
    """Return a full cost-of-living breakdown for the requested city."""
    result = main(city_name)
    status = 200 if result.get("mode") != "error" else 404
    return jsonify(result), status


# ── City recommendation ───────────────────────────────────────────────────────

@app.route("/recommend", methods=["POST"])
def recommend():
    """
    Accept user lifestyle preferences and return the top-5 recommended cities.

    Expected JSON body
    ------------------
    {
        "salary":           float,
        "location":         "center" | "outside",
        "housing_type":     "studio" | "family",
        "transport_type":   "public" | "car",
        "gas_liters":       int,
        "cheap_visits":     int,
        "mid_visits":       int,
        "fast_food_visits": int,
        "go_gym":           bool,
        "play_tennis":      bool,
        "tennis_frequency": int,
        "go_cinema":        bool,
        "cinema_visits":    int
    }
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be valid JSON."}), 400

    required = [
        "salary", "location", "housing_type", "transport_type",
        "gas_liters", "cheap_visits", "mid_visits", "fast_food_visits",
        "go_gym", "play_tennis", "tennis_frequency", "go_cinema", "cinema_visits",
    ]
    missing = [k for k in required if k not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400

    try:
        user_preferences = {
            "salary":           float(data["salary"]),
            "location":         str(data["location"]),
            "housing_type":     str(data["housing_type"]),
            "transport_type":   str(data["transport_type"]),
            "gas_liters":       int(data["gas_liters"]),
            "cheap_visits":     int(data["cheap_visits"]),
            "mid_visits":       int(data["mid_visits"]),
            "fast_food_visits": int(data["fast_food_visits"]),
            "go_gym":           bool(data["go_gym"]),
            "play_tennis":      bool(data["play_tennis"]),
            "tennis_frequency": int(data["tennis_frequency"]),
            "go_cinema":        bool(data["go_cinema"]),
            "cinema_visits":    int(data["cinema_visits"]),
        }
    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid field value: {e}"}), 400

    try:
        from model import knn_rank
        result = knn_rank(user_preferences)
    except ImportError:
        return jsonify({"error": "Recommendation model (model.py) not found."}), 501

    if result is None or (hasattr(result, "empty") and result.empty):
        return jsonify({"error": "No cities found for your budget and preferences."}), 404

    return jsonify(result.to_dict(orient="records"))


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    port  = int(os.getenv("PORT", 5000))
    app.run(debug=debug, port=port)
