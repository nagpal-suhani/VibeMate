from flask import Flask, request, jsonify
from matching_and_allocation_algorithm import get_top_matches_for_user

app = Flask(__name__)

@app.route("/match", methods=["GET"])
def match():
    user_id = request.args.get("user_id", type=int)
    if user_id is None:
        return jsonify({"error": "Missing user_id"}), 400
    matches = get_top_matches_for_user(user_id)
    # Convert objects to dicts for JSON serialization
    def serialize(match):
        return {
            "roommate_name": match["suggested_roommate"].name,
            "room_id": match["suggested_room"].room_id,
            "compatibility_score": match["compatibility_score"],
            "hotspots": match["hotspots"],
            "coldspots": match["coldspots"],
            "proactive_measures": match["proactive_measures"]
        }
    return jsonify([serialize(m) for m in matches])

if __name__ == "__main__":
    app.run(port=5000, debug=True)