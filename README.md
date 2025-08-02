## VibeMate — AI-Powered Roommate & Room Matching
VibeMate is a self-improving, explainable roommate and room matching system designed for women’s co-living spaces. It combines voice-driven micro-surveys, behavioral traits, and feedback loops to dynamically match roommates and suggest rooms while continuously tuning its own weighting logic via feedback.

# Project Overview
Traditional roommate matching uses static forms and generic filters. VibeMate leverages:

Voice AI micro-surveys to infer traits like cleanliness, sleep habits, communication style, and social behavior.

Compatibility scoring with dynamic weights that adapt based on real feedback.

Room preference alignment (natural light, ventilation, quiet zones, floor level).

Explainability via hotspot/coldspot analysis and proactive recommendations.

Self-learning feedback loop that adjusts trait weightings using logistic regression when poor matches exceed a threshold.

 # Core Features
Trait-based roommate compatibility scoring (cleanliness, sleep habits, social behavior, communication).

Room preference scoring (quiet zone, natural light, ventilation, floor level).

Self-learning feedback manager that suggests new weights when bad-match ratio is high.

Explainable match suggestions: hotspots, coldspots, and proactive tips.

MongoDB integration for profiles, feedback, and room state.

Persistent latest weights stored to drive future matching decisions.

# Requirements
Python 3.10+

MongoDB access (Atlas URI used in code by default)

Packages:

pandas

scikit-learn

joblib

pymongo

(standard library: os, json, typing)

# INSTALLATION
git clone <your-repo-url>
cd vibemate

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Linux/macOS
.\venv\Scripts\activate         # Windows

# Install dependencies
pip install pandas scikit-learn joblib pymongo

# Key Components
GuestProfile – Stores guest traits and exposes a feature vector for matching.
Room – Stores room attributes and occupants, with twin-availability check.
FeedbackManager – Handles self-learning weight adjustment via logistic regression when bad match ratio exceeds threshold.
find_best_match_weighted – Uses weights + preferences to score compatibility and generate explainable matches.

# Usage
Run the script:

bash
Copy
Edit
python main.py
