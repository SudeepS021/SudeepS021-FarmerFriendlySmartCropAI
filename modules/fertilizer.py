def recommend_fertilizer(N, P, K, crop):

    crop = crop.lower()

    # Ideal NPK requirements (basic logic example)
    crop_requirements = {
        "rice": {"N": 100, "P": 50, "K": 50},
        "wheat": {"N": 120, "P": 60, "K": 40},
        "maize": {"N": 110, "P": 50, "K": 40},
        "cotton": {"N": 80, "P": 40, "K": 40},
        "sugarcane": {"N": 150, "P": 60, "K": 60}
    }

    if crop not in crop_requirements:
        return "Crop not found in database. Use balanced NPK fertilizer."

    ideal = crop_requirements[crop]

    recommendations = []

    if N < ideal["N"]:
        recommendations.append("Increase Nitrogen (Use Urea)")
    if P < ideal["P"]:
        recommendations.append("Increase Phosphorus (Use DAP)")
    if K < ideal["K"]:
        recommendations.append("Increase Potassium (Use MOP)")

    if not recommendations:
        return "NPK levels are sufficient for this crop. Use maintenance fertilizer."

    return ", ".join(recommendations)