import re

# ── Mots clés par intention ───────────────────────────
INTENT_PATTERNS = {
    "SALUTATION": [
        "السلام", "bonjour", "salam", "صباح", "مساء",
        "أهلا", "مرحبا", "hello", "bonsoir"
    ],
    "DEMANDE_MEDICAMENT": [
        "بغيت دوا", "عطيني دوا", "شنو دوا", "أي دوا",
        "medicament", "médicament", "دواء", "حبة"
    ],
    "DEMANDE_PAR_SYMPTOME": [
        "عندي", "كنحس", "كيوجعني", "j'ai", "je souffre",
        "كيدير", "تعبت", "مريض", "ألم", "وجع"
    ],
    "DEMANDE_INFO": [
        "شنو هو", "واش", "كيفاش", "علاش", "شنو معنى",
        "c'est quoi", "qu'est ce que", "ما هو", "ما هي"
    ],
    "DEMANDE_SPECIALISTE": [
        "فين نمشي", "أي طبيب", "شنو الطبيب", "quel médecin",
        "où aller", "طبيب", "دكتور", "مستشفى"
    ]
}

def detect_intent(question):
    """Détecter l'intention de l'utilisateur"""
    question_lower = question.lower()
    
    for intent, patterns in INTENT_PATTERNS.items():
        for pattern in patterns:
            if pattern.lower() in question_lower:
                return intent
    
    # Par défaut
    return "DEMANDE_PAR_SYMPTOME"

# ── Test ─────────────────────────────────────────────
if __name__ == "__main__":
    tests = [
        "السلام عليكم",
        "عندي صداع شديد",
        "بغيت دوا ديال الضغط",
        "شنو هو الإيبوبروفين",
        "فين نمشي ديال القلب",
        "j'ai de la fièvre",
        "c'est quoi le paracetamol"
    ]
    
    for q in tests:
        intent = detect_intent(q)
        print(f"'{q}' → {intent}")