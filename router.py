def route_question(question):

    prompt = f"""
You are the intent-classification agent for DefenceOps AI.

Your task is to classify the user's question into exactly ONE
of these four categories:

1. Equipment
2. Maintenance
3. Safety
4. General

IMPORTANT CLASSIFICATION RULES:

EQUIPMENT:
Use Equipment when the question is about:
- starting or operating equipment
- equipment not starting
- equipment malfunction
- troubleshooting equipment operation
- equipment components
- how the equipment works
- equipment specifications

Examples:
"What should I check if the Falcon-1 does not start?" → Equipment
"How do I start the Falcon-1?" → Equipment
"What does the control selector do?" → Equipment

MAINTENANCE:
Use Maintenance when the question is specifically about:
- inspection
- cleaning
- servicing
- repair
- maintenance procedures
- battery maintenance
- replacing components

Examples:
"How should I inspect the battery?" → Maintenance
"How should I clean the equipment?" → Maintenance
"What should I inspect during routine maintenance?" → Maintenance

SAFETY:
Use Safety when the question is specifically about:
- hazards
- warnings
- dangerous conditions
- smoke
- unusual odor
- leakage
- overheating
- precautions
- safe operation
- PPE

Examples:
"What should I do if there is smoke?" → Safety
"What precautions should I take?" → Safety
"What should I do if the battery is leaking?" → Safety

GENERAL:
Use General when the question does not clearly belong
to Equipment, Maintenance, or Safety.

IMPORTANT:
A question about an equipment failure or equipment not starting
must be classified as Equipment, NOT Maintenance.

USER QUESTION:
{question}

Return ONLY ONE word:
Equipment
Maintenance
Safety
General
"""

    response = ollama.chat(
        model="llama3.2",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    result = response["message"]["content"].strip()

    valid_domains = [
        "Equipment",
        "Maintenance",
        "Safety",
        "General"
    ]

    for domain in valid_domains:
        if domain.lower() == result.lower():
            return domain

    # Fallback if Llama returns extra text
    for domain in valid_domains:
        if domain.lower() in result.lower():
            return domain

    return "General"