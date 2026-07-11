import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# ==========================================
# 1. MATHEMATICAL FUZZY CONTROL SYSTEM
# ==========================================
def run_tourist_fuzzy_inference(budget, weather, duration):
    in_budget = ctrl.Antecedent(np.arange(1, 11, 1), 'budget')
    in_weather = ctrl.Antecedent(np.arange(1, 11, 1), 'weather')
    in_duration = ctrl.Antecedent(np.arange(1, 11, 1), 'duration')
    suitability = ctrl.Consequent(np.arange(0, 101, 1), 'suitability')

    for obj in [in_budget, in_weather, in_duration]:
        obj['low'] = fuzz.trimf(obj.universe, [1, 1, 5])
        obj['medium'] = fuzz.trimf(obj.universe, [3, 5, 8])
        obj['high'] = fuzz.trimf(obj.universe, [6, 10, 10])

    suitability['low'] = fuzz.trimf(suitability.universe, [0, 0, 50])
    suitability['medium'] = fuzz.trimf(suitability.universe, [30, 50, 80])
    suitability['high'] = fuzz.trimf(suitability.universe, [60, 100, 100])

    rule1 = ctrl.Rule(in_budget['high'] & in_duration['high'], suitability['high'])
    rule2 = ctrl.Rule(in_budget['low'] | in_duration['low'], suitability['low'])
    rule3 = ctrl.Rule(in_budget['medium'] & in_weather['medium'], suitability['medium'])

    suitability_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
    simulation = ctrl.ControlSystemSimulation(suitability_ctrl)

    simulation.input['budget'] = budget
    simulation.input['weather'] = weather
    simulation.input['duration'] = duration

    try:
        simulation.compute()
        score = simulation.output['suitability']
    except:
        score = np.mean([budget, weather, duration]) * 10

    if score < 40:
        cat = "Low Preference Match"
    elif 40 <= score < 75:
        cat = "Medium Preference Match"
    else:
        cat = "High / Excellent Match"

    return round(score, 2), cat


# ==========================================
# 2. ADDITIONAL AI LOGIC MODULES
# ==========================================
def suggest_accommodation_and_hotels(budget_level, members_count):
    """
    Fuzzy-driven logic to dynamically recommend hotel tiers and room setups
    """
    # Dynamic Hotel Tier Based on Budget Input
    if budget_level <= 3:
        hotel_tier = "Economy Guest House / Backpacker Hostels"
        estimated_cost = "Low Cost (PKR 3,000 - 6,000 per night)"
    elif 3 < budget_level <= 7:
        hotel_tier = "3-Star Standard Hotel / Executive Residency"
        estimated_cost = "Moderate Cost (PKR 8,000 - 15,000 per night)"
    else:
        hotel_tier = "Premium 5-Star Luxury Resort"
        estimated_cost = "Premium Cost (PKR 25,000+ per night)"
        
    # Dynamic Room Configuration Based on Group/Family Size
    if members_count == 1:
        room_type = "Single Standard Room"
    elif members_count == 2:
        room_type = "Double Sharing / King Size Bed Room"
    elif 3 <= members_count <= 5:
        room_type = "Family Suite (Interconnected Rooms)"
    else:
        room_type = "Multiple Deluxe Rooms / Independent Vacation Villa"
        
    return hotel_tier, estimated_cost, room_type


def knn_tourist_engine(user_profile, df, features, n_neighbors=2):
    X = df[features].values
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric='minkowski', p=2)
    knn.fit(X)
    distances, indices = knn.kneighbors([user_profile])
    return df.iloc[indices[0]]


# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    print("\n=============================================")
    print(" 🌴 ADVANCED SMART TOURISM RECOMMENDATION SYSTEM 🌴")
    print("=============================================\n")
    
    # User Inputs
    budget = float(input("Enter Budget Level (1-10): "))
    weather = float(input("Enter Weather Preference (1-10 [Cold to Hot]): "))
    duration = float(input("Enter Trip Duration Level (1-10): "))
    members = int(input("Enter Total Number of Family Members/Travelers: "))
    
    # 1. Fuzzy Logic Suitability
    score, status = run_tourist_fuzzy_inference(budget, weather, duration)
    
    # 2. Additional Modules Execution
    hotel_tier, cost_range, room_setup = suggest_accommodation_and_hotels(budget, members)
    
    print("\n" + "="*45)
    print("              💥 SYSTEM RESULTS 💥             ")
    print("="*45)
    print(f"[Fuzzy Logic Index] Suitability Score: {score}% -> ({status})")
    print(f"[Hotel Suggestion]  Tier: {hotel_tier}")
    print(f"[Cost Estimate]     Range: {cost_range}")
    print(f"[Room Allocation]   Required Setup: {room_setup}")
    print("="*45)
    
    # 3. KNN Dataset Initialization
    data = {
        'Place': ["Swat Valley", "Hunza Valley (Luxury)", "Karachi Coastline", "Lahore Historical Tour", "Murree Hills"],
        'Budget': [3.0, 9.0, 4.0, 7.0, 5.0],
        'Weather': [2.0, 1.0, 8.0, 7.0, 3.0],
        'Duration': [4.0, 8.0, 3.0, 6.0, 4.0]
    }
    df = pd.DataFrame(data)
    
    # 4. KNN Execution
    results = knn_tourist_engine([budget, weather, duration], df, ['Budget', 'Weather', 'Duration'])
    
    print("\n🎯 KNN System Top Destination Matches:")
    for idx, row in results.iterrows():
        print(f" * {row['Place']} (Optimal Match)")
    print("=============================================\n")