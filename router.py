def route_question(question):
    question_lower = question.lower()

    safety_words = [
        "safety",
        "safe",
        "danger",
        "hazard",
        "precaution",
        "warning",
        "protective",
        "ppe",
        "smoke",
        "leakage",
        "leak",
        "unusual odor",
        "overheating",
        "heating"
    ]

    maintenance_words = [
        "maintenance",
        "maintain",
        "repair",
        "service",
        "replace",
        "clean",
        "cleaning",
        "inspect",
        "inspection",
        "inspection procedure",
        "battery maintenance"
    ]

    equipment_words = [
        "equipment",
        "machine",
        "engine",
        "battery",
        "device",
        "system",
        "component",
        "start",
        "starts",
        "starting",
        "does not start",
        "fails to start",
        "not working",
        "malfunction",
        "troubleshoot",
        "troubleshooting"
    ]

    if any(word in question_lower for word in safety_words):
        return "Safety"

    if any(word in question_lower for word in maintenance_words):
        return "Maintenance"

    if any(word in question_lower for word in equipment_words):
        return "Equipment"

    return "General"