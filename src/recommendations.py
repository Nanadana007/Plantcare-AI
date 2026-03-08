from __future__ import annotations


def _normalize(label: str) -> str:
    return label.replace("___", " ").replace("_", " ").lower().strip()


def get_recommendation(label: str) -> str:
    normalized = _normalize(label)

    if "healthy" in normalized:
        return "Plant appears healthy. Keep consistent watering, airflow, and periodic nutrient support."

    rule_book = {
        "blight": "Remove infected leaves, improve airflow, and apply a copper-based fungicide if spread continues.",
        "rust": "Prune affected parts, avoid overhead watering, and apply a sulfur or copper fungicide.",
        "mildew": "Reduce humidity around leaves, increase sunlight exposure, and use a fungicidal spray.",
        "spot": "Isolate the plant, remove spotted leaves, and sanitize tools after pruning.",
        "scab": "Collect fallen debris, prune crowded branches, and apply preventive fungicide in high-risk periods.",
        "mosaic": "Likely viral. Remove severely infected plants, control insect vectors, and disinfect tools.",
        "rot": "Improve drainage, reduce watering frequency, and remove rotten tissue promptly.",
        "yellow": "Check watering and nutrient levels; inspect for pests and adjust fertilization.",
    }

    for keyword, advice in rule_book.items():
        if keyword in normalized:
            return advice

    return "Disease detected. Isolate the plant, remove visibly infected areas, and consult local crop guidance for treatment."
