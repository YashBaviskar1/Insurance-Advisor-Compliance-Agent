import json

LOCATION_RISKS = {
    "Mumbai": "flood-prone coastal area with seismic activity (Zone III)",
    "Delhi": "high air pollution, urban flooding, and seismic risk (Zone IV)",
    "Chennai": "cyclone-prone coastal flood zone in seismic Zone III",
    "Kerala": "flood and landslide-prone high rainfall, seismic Zone III",
    "Kolkata": "cyclone and flood risk in low-elevation coastal area, seismic Zone III",
    "Bangalore": "low natural disaster risk but high urban stress and occasional flooding",
    "Hyderabad": "flash flooding from heavy monsoon rains due to poor drainage",
    "Pune": "urban flooding risks from monsoon rains and hill‑slope terrain",
    "Ahmedabad": "acute heatwaves, drought-prone semi‑arid region",
    "Guwahati": "very high vulnerability to earthquakes, floods, and storms",
    "Lucknow": "flood risk from monsoon and moderate seismic activity (Zone III)",
    "Patna": "recurrent flooding from Himalayan river systems"
}

def input_int(prompt, min_val=None, max_val=None):
    while True:
        try:
            val = int(input(prompt))
            if (min_val is not None and val < min_val) or (max_val is not None and val > max_val):
                print(f"Please enter a value between {min_val} and {max_val}.")
            else:
                return val
        except ValueError:
            print("Please enter a valid integer.")

def input_list(prompt):
    items = input(prompt)
    return [item.strip() for item in items.split(",") if item.strip()]

def collect_dependents():
    dependents = []
    while True:
        add = input("Add a dependent? (y/n): ").strip().lower()
        if add != 'y':
            break
        relation = input("  Relation (e.g. spouse, child, parent): ").strip()
        age = input_int("  Age: ", 0, 120)
        dependents.append({"relation": relation, "age": age})
    return dependents

def get_location():
    print("\nAvailable locations:")
    for loc in LOCATION_RISKS.keys():
        print(f" - {loc}")
    while True:
        loc = input("\nEnter your location from the list above: ").strip()
        if loc in LOCATION_RISKS:
            return loc
        else:
            print("Invalid location. Please choose from the list.")

def main():
    print("=== Insurance User Profile Collector ===\n")
    age = input_int("Your age: ", 18, 100)

    dependents = collect_dependents()

    income = input("Your income (e.g. ₹15L/year): ").strip()

    assets = input_list("List your assets separated by commas (e.g. home, car): ")

    health_conditions = input_list("List any health conditions separated by commas (or leave blank): ")

    location = get_location()

    occupation = input("Your occupation: ").strip()

    # For simplicity, derive risk factors from hardcoded location risk mapping
    risk_factors = {
        "flood_zone": "flood" in LOCATION_RISKS[location].lower(),
        "earthquake_zone": 4 if "Zone IV" in LOCATION_RISKS[location] else 3,
        "high_pollution": "pollution" in LOCATION_RISKS[location].lower()
    }

    user_profile = {
        "age": age,
        "dependents": dependents,
        "income": income,
        "assets": assets,
        "health_conditions": health_conditions,
        "location": location,
        "occupation": occupation,
        "risk_factors": risk_factors
    }

    filename = input("\nEnter filename to save (e.g. user_profile.json): ").strip() or "user_profile.json"

    with open(filename, 'w') as f:
        json.dump(user_profile, f, indent=4)

    print(f"\n✅ User profile saved to {filename}")

if __name__ == "__main__":
    main()
