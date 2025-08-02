import pandas as pd
from typing import List, Dict, Any, Optional
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import StandardScaler
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
import json


# Load the latest weights if they exist
weights_file = "latest_weights.json"
if os.path.exists(weights_file):
    with open(weights_file, "r") as f:
        new_weights = json.load(f)


# --- 1. Define Data Structures ---
class GuestProfile:
    """Represents a guest with the new detailed compatibility and preference traits."""
    def __init__(self, user_id: int, name: str, traits: Dict[str, Any]):
        self.user_id = user_id
        self.name = name
        self.cleanliness = traits.get('cleanliness', 3)
        self.sleeping_habits = traits.get('sleeping_habits', 0)
        self.social_behaviour = traits.get('social_behaviour', 3)
        self.communication_frequency = traits.get('communication_frequency_with_roommate', 3)
        self.communication_style_directness = traits.get('communication_style_directness', 3)
        self.communication_style_expressiveness = traits.get('communication_style_expressiveness', 3)
        self.noise_level_preference = traits.get('noise_level_preference', 0)
        self.room_preference_natural_light = traits.get('room_preference_natural_light', 0)
        self.room_preference_ventilation = traits.get('room_preference_ventilation', 0)
        self.room_preference_floor_level = traits.get('room_preference_floor_level', 0)

    def get_feature_vector(self) -> List[int]:
        """Returns the core compatibility traits as a numerical vector."""
        return [
            self.cleanliness,
            self.sleeping_habits,
            self.social_behaviour,
            self.communication_frequency,
            self.communication_style_directness,
            self.communication_style_expressiveness
        ]

class Room:
    """Represents a physical room with its own attributes."""
    def __init__(self, room_id: str, attributes: Dict[str, Any], occupants: List[GuestProfile] = []):
        self.room_id = room_id
        self.occupants = occupants
        self.has_natural_light = attributes.get('has_natural_light', False)
        self.has_good_ventilation = attributes.get('has_good_ventilation', False)
        self.floor_level = attributes.get('floor_level', 0)
        self.is_quiet_zone = attributes.get('is_quiet_zone', False)

    def is_available_for_twin(self) -> bool:
        return len(self.occupants) == 1

# --- 2. Self-Learning Feedback Manager (MODIFIED) ---

class FeedbackManager:
    """
    Handles the self-learning aspect of the matching algorithm by training
    a model to propose new weights based on feedback.
    """
    def __init__(self, profiles_df: pd.DataFrame, feedback_df: pd.DataFrame, bad_match_threshold: float = 0.4):
        self.profiles_df = profiles_df
        self.feedback_df = feedback_df
        self.bad_match_threshold = bad_match_threshold
        self.model = LogisticRegression(solver='liblinear')
        self.scaler = StandardScaler()
        self.original_weights = {
            'cleanliness': 10,
            'sleeping_habits': 15,
            'social_behaviour': 8,
            'communication_frequency': 5,
            'communication_style_directness': 7,
            'communication_style_expressiveness': 8
        }
        self.feature_names = [
            'cleanliness', 'sleeping_habits', 'social_behaviour',
            'communication_frequency_with_roommate', 'communication_style_directness',
            'communication_style_expressiveness'
        ]

    def _prepare_training_data(self) -> tuple[pd.DataFrame, pd.Series]:
        """
        Creates features based on the absolute difference between guest traits
        and prepares the training data from the provided dataframes.
        """
        X = []
        y = []

        for _, row in self.feedback_df.iterrows():
            guest1_id = row['guest1_id']
            guest2_id = row['guest2_id']
            feedback = row['feedback']

            profile1 = self.profiles_df[self.profiles_df['user_id'] == guest1_id].iloc[0]
            profile2 = self.profiles_df[self.profiles_df['user_id'] == guest2_id].iloc[0]

            features = [abs(profile1[col] - profile2[col]) for col in self.feature_names]
            X.append(features)
            y.append(feedback)

        X_df = pd.DataFrame(X, columns=self.feature_names)
        y_series = pd.Series(y)
        return X_df, y_series

    def suggest_new_weights(self) -> Dict[str, Dict[str, float]]:
        """
        Main method to check the threshold and suggest new weights.
        """
        X_train, y_train = self._prepare_training_data()

        bad_matches = y_train[y_train == 0].count()
        total_matches = len(y_train)
        bad_match_ratio = bad_matches / total_matches if total_matches > 0 else 0

        print(f"Total feedback entries: {total_matches}")
        print(f"Number of bad matches: {bad_matches}")
        print(f"Bad match ratio: {bad_match_ratio:.2%}")

        new_weights = self.original_weights.copy()
        model_was_trained = False

        if bad_match_ratio > self.bad_match_threshold:
            print("\n🚨 Bad match threshold exceeded! Retraining model to suggest new weights...")
            
            X_scaled = self.scaler.fit_transform(X_train)
            self.model.fit(X_scaled, y_train)

            coefficients = self.model.coef_[0]

            abs_coeffs = [abs(c) for c in coefficients]

            if sum(abs_coeffs) == 0:
                scaled_weights = {key: 1 for key in self.original_weights}
            else:
                max_orig_weight = max(self.original_weights.values())
                max_abs_coeff = max(abs_coeffs)
                scaling_factor = max_orig_weight / max_abs_coeff if max_abs_coeff > 0 else 1

                new_weights = {
                    self.feature_names[i]: (abs_coeffs[i] * scaling_factor)
                    for i in range(len(self.feature_names))
                }
            print(f"✅ New weights suggested based on feedback analysis.")
            model_was_trained = True
        else:
            print("\n✅ Bad match ratio is below threshold. No weight adjustment needed.")

        updated_weights_output = {
            'cleanliness': new_weights.get('cleanliness', 0),
            'sleeping_habits': new_weights.get('sleeping_habits', 0),
            'social_behaviour': new_weights.get('social_behaviour', 0),
            'communication_frequency': new_weights.get('communication_frequency_with_roommate', 0),
            'communication_style_directness': self.original_weights.get('communication_style_directness', 0),
            'communication_style_expressiveness': self.original_weights.get('communication_style_expressiveness', 0)
        }

        if model_was_trained:
            with open("latest_weights.json", "w") as f:
                json.dump(new_weights, f)

        return {
            "old_weights": self.original_weights,
            "new_weights": updated_weights_output,
            "model_was_trained": model_was_trained
        }

# --- 3. The Matching Algorithm (MODIFIED to accept and use dynamic weights) ---

def find_best_match_weighted(new_guest: GuestProfile, available_rooms: List[Room], weights: Dict[str, float], top_n: int = 3) -> List[Dict[str, Any]]:
    all_possible_matches = []

    for room in available_rooms:
        if not room.is_available_for_twin(): continue
        existing_occupant = room.occupants[0]

        # --- A. Calculate Rule-Based Score using dynamic weights ---
        diffs = {
            'cleanliness': abs(new_guest.cleanliness - existing_occupant.cleanliness),
            'sleeping_habits': abs(new_guest.sleeping_habits - existing_occupant.sleeping_habits),
            'social_behaviour': abs(new_guest.social_behaviour - existing_occupant.social_behaviour),
            'communication_frequency': abs(new_guest.communication_frequency - existing_occupant.communication_frequency),
        }
        weighted_diff_sum = sum(weights[key] * diffs.get(key, 0) for key in diffs)
        
        # Calculate max possible diff score using the new weights for normalization
        max_possible_diff_score = (weights['cleanliness'] * 4) + (weights['sleeping_habits'] * 1) + (weights['social_behaviour'] * 4) + (weights['communication_frequency'] * 4)
        compatibility_score = 80 - (80 * weighted_diff_sum / max_possible_diff_score) if max_possible_diff_score else 80

        penalty = 0
        comm_penalty_directness = False
        comm_penalty_expressiveness = False
        if new_guest.communication_style_directness <= 2 and existing_occupant.communication_style_directness <= 2:
            penalty += weights.get('communication_style_directness', 7)
            comm_penalty_directness = True
        if new_guest.communication_style_expressiveness <= 2 and existing_occupant.communication_style_expressiveness <= 2:
            penalty += weights.get('communication_style_expressiveness', 8)
            comm_penalty_expressiveness = True
        compatibility_score -= penalty

        preference_score = 0
        if new_guest.noise_level_preference == 1 and room.is_quiet_zone: preference_score += 7
        if new_guest.room_preference_natural_light == 1 and room.has_natural_light: preference_score += 4
        if new_guest.room_preference_ventilation == 1 and room.has_good_ventilation: preference_score += 4
        if new_guest.room_preference_floor_level == room.floor_level: preference_score += 5

        rule_based_score = compatibility_score + preference_score
        
        # In this modified version, we'll use a rule-based score for simplicity,
        # since the AI model is now used for weights and not direct scoring.
        final_score = rule_based_score

        # --- B. FEATURE-LEVEL HOTSPOT/COLDSPOT & PROACTIVE MEASURES LOGIC ---
        hotspots = []
        coldspots = []

        if diffs['sleeping_habits'] == 0: hotspots.append("You share the same sleep schedule.")
        else: coldspots.append("Opposite sleep schedules (Night Owl vs. Early Bird).")
        if diffs['cleanliness'] <= 1: hotspots.append("Similar cleanliness standards.")
        elif diffs['cleanliness'] >= 3: coldspots.append("Different cleanliness standards.")
        if diffs['social_behaviour'] <= 1: hotspots.append("Similar social energy.")
        elif diffs['social_behaviour'] >= 3: coldspots.append("Different social energy (introvert vs. extrovert).")
        if comm_penalty_directness: coldspots.append("Both tend toward indirect communication.")
        if comm_penalty_expressiveness: coldspots.append("Both may be less expressive communicators.")
        if new_guest.noise_level_preference == 1 and room.is_quiet_zone: hotspots.append("Room is in a quiet zone, which you prefer.")
        if new_guest.room_preference_natural_light == 1 and room.has_natural_light: hotspots.append("Room has great natural light.")
        if new_guest.room_preference_ventilation == 1 and room.has_good_ventilation: hotspots.append("Room has good ventilation.")
        if new_guest.room_preference_floor_level == room.floor_level: hotspots.append("Room is on your preferred floor level.")

        proactive_measures = ""
        if 50 <= final_score <= 80 and coldspots:
            proactive_measures = f"This is a workable match! To ensure a smooth experience, we recommend proactively discussing expectations around: {', '.join([c.split('(')[0].strip() for c in coldspots])}."

        all_possible_matches.append({
            "suggested_roommate": existing_occupant,
            "suggested_room": room,
            "compatibility_score": round(final_score, 2),
            "hotspots": hotspots,
            "coldspots": coldspots,
            "proactive_measures": proactive_measures
        })

    sorted_matches = sorted(all_possible_matches, key=lambda x: x['compatibility_score'], reverse=True)
    return sorted_matches[:top_n]



def get_top_matches_for_user(user_id: int, top_n: int = 3):
    """
    Given a user_id, runs the matching algorithm and returns the top matches.
    """
    try:
        client = MongoClient("mongodb+srv://matching_algo:matching_algo@vibey-cluster.rojse0m.mongodb.net/?retryWrites=true&w=majority&appName=Vibey-cluster") 
        client.admin.command('ping')
        db = client['VibeMate'] 
    except ConnectionFailure as e:
        print(f"Error: Could not connect to MongoDB. Details: {e}")
        return []

    profiles_collection = db['profile']
    feedback_collection = db['feedback']
    rooms_collection = db['room']

    profiles_df = pd.DataFrame(list(profiles_collection.find({})))
    feedback_df = pd.DataFrame(list(feedback_collection.find({})))
    if '_id' in profiles_df.columns:
        profiles_df = profiles_df.drop(columns=['_id'])
    if '_id' in feedback_df.columns:
        feedback_df = feedback_df.drop(columns=['_id'])

    feedback_manager = FeedbackManager(profiles_df, feedback_df)
    weight_report = feedback_manager.suggest_new_weights()
    new_weights = weight_report["new_weights"]

    user_doc = profiles_collection.find_one({'user_id': user_id})
    if not user_doc:
        print(f"Error: Could not find user_id {user_id} in the database.")
        return []

    new_guest = GuestProfile(user_id=user_doc['user_id'], name=user_doc['name'], traits=user_doc)
    all_profiles = {doc['user_id']: GuestProfile(user_id=doc['user_id'], name=doc['name'], traits=doc) for doc in profiles_collection.find({})}

    all_rooms = []
    for room_doc in rooms_collection.find({}):
        occupant_profile = all_profiles.get(room_doc.get('occupant_user_id'))
        occupants_list = [occupant_profile] if occupant_profile else []
        all_rooms.append(Room(room_id=room_doc['room_id'], attributes=room_doc, occupants=occupants_list))

    available_rooms_for_new_guest = [room for room in all_rooms if new_guest.user_id not in [occ.user_id for occ in room.occupants]]

    top_matches = find_best_match_weighted(new_guest, available_rooms_for_new_guest, weights=new_weights, top_n=top_n)
    return top_matches


# --- 4. Main Execution Block (MODIFIED to read from MongoDB) ---

if __name__ == "__main__":
    
    try:
        client = MongoClient("mongodb+srv://matching_algo:matching_algo@vibey-cluster.rojse0m.mongodb.net/?retryWrites=true&w=majority&appName=Vibey-cluster") 
        client.admin.command('ping')
        db = client['VibeMate'] 
        print("MongoDB connection successful.")
    except ConnectionFailure as e:
        print(f"Error: Could not connect to MongoDB. Make sure the server is running. \nDetails: {e}")
        exit()

    # Get collections
    profiles_collection = db['profile']
    feedback_collection = db['feedback']
    rooms_collection = db['room']

    # --- Load Data from MongoDB into Pandas DataFrames ---
    profiles_df = pd.DataFrame(list(profiles_collection.find({})))
    feedback_df = pd.DataFrame(list(feedback_collection.find({})))
    
    # Handle the '_id' column if present
    if '_id' in profiles_df.columns:
        profiles_df = profiles_df.drop(columns=['_id'])
    if '_id' in feedback_df.columns:
        feedback_df = feedback_df.drop(columns=['_id'])
        
    print(f"\nLoaded {len(profiles_df)} profiles and {len(feedback_df)} feedback entries from MongoDB.")

    # Initialize the FeedbackManager
    feedback_manager = FeedbackManager(profiles_df, feedback_df)
    
    # Check the threshold and get the suggested weights
    weight_report = feedback_manager.suggest_new_weights()
    
    # Use the suggested new weights for the matching algorithm
    new_weights = weight_report["new_weights"]
    
    print("\n--- Weight Adjustment Report ---")
    print("Old Weights:", {k: round(v, 2) for k, v in weight_report["old_weights"].items()})
    
    if weight_report["model_was_trained"]:
      print("New Suggested Weights:", {k: round(v, 2) for k, v in weight_report["new_weights"].items()})
    else:
      print("New Suggested Weights: Weights were not adjusted as the bad match ratio was below the threshold.")
    
    # --- Example Matching Scenario with a Guest from the DB ---
    # Find a specific guest (e.g., Anika, user_id=2) to act as the 'new guest'
    anika_doc = profiles_collection.find_one({'user_id': 2})
    if anika_doc:
        # Create a GuestProfile object from the MongoDB document
        new_guest = GuestProfile(user_id=anika_doc['user_id'], name=anika_doc['name'], traits=anika_doc)
        print(f"\n--- SCENARIO: Finding matches for {new_guest.name}. ---")
    else:
        print("Error: Could not find 'Anika' (user_id 2) in the database. Exiting.")
        exit()

    # Convert all other profiles into GuestProfile objects
    all_profiles = {doc['user_id']: GuestProfile(user_id=doc['user_id'], name=doc['name'], traits=doc) for doc in profiles_collection.find({})}

    # Load rooms and their occupants from MongoDB
    all_rooms = []
    for room_doc in rooms_collection.find({}):
        occupant_profile = all_profiles.get(room_doc.get('occupant_user_id'))
        occupants_list = [occupant_profile] if occupant_profile else []
        all_rooms.append(Room(room_id=room_doc['room_id'], attributes=room_doc, occupants=occupants_list))

    available_rooms_for_new_guest = [room for room in all_rooms if new_guest.user_id not in [occ.user_id for occ in room.occupants]]
    
    # Use the NEW weights for the matching calculation
    top_matches = find_best_match_weighted(new_guest, available_rooms_for_new_guest, weights=new_weights, top_n=3)

    if top_matches:
        print("\n--- Top 3 Match Recommendations (Using NEW Weights) ---")
        for i, match in enumerate(top_matches):
            print(f"\n--- Match #{i+1} with {match['suggested_roommate'].name} in Room {match['suggested_room'].room_id} ---")
            print(f"📊 Final Score: {match['compatibility_score']}/100")
            
            if match['hotspots']:
                print("🔥 Hotspots (Strengths):")
                for h in match['hotspots']:
                    print(f"   - {h}")
            
            if match['coldspots']:
                print("❄️ Coldspots (Potential Friction):")
                for c in match['coldspots']:
                    print(f"   - {c}")
            
            if match['proactive_measures']:
                print(f"💡 Proactive Tip: {match['proactive_measures']}")
    else:
        print("\nNo suitable matches found.")



