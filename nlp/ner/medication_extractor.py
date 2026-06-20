import re
import requests
import unicodedata
import json

# ── Médicaments disponibles au Maroc ─────────────────
MOROCCO_MEDICATIONS = {
    "headache":       ["paracetamol", "ibuprofen", "aspirin"],
    "fever":          ["paracetamol", "ibuprofen"],
    "infection":      ["amoxicillin", "azithromycin", "ciprofloxacin"],
    "allergy":        ["cetirizine", "loratadine"],
    "blood pressure": ["amlodipine", "ramipril", "atenolol"],
    "diabetes":       ["metformin", "glibenclamide"],
    "cough":          ["salbutamol", "ambroxol"],
    "pain":           ["ibuprofen", "diclofenac", "paracetamol"],
    "nausea":         ["metoclopramide", "domperidone"],
    "diarrhea":       ["loperamide"],
    "anxiety":        ["diazepam", "sertraline"],
    "depression":     ["sertraline", "fluoxetine"],
    "asthma":         ["salbutamol", "montelukast"],
    "cholesterol":    ["atorvastatin", "simvastatin"],
}

SYMPTOMS = {
    # ── Arabe classique ──────────────────────────────
    "صداع": "headache", "حمى": "fever", "سعال": "cough",
    "ألم": "pain", "غثيان": "nausea", "إسهال": "diarrhea",
    "ضغط": "blood pressure", "سكر": "diabetes", "حساسية": "allergy",
    "التهاب": "infection", "دوار": "dizziness", "قلق": "anxiety",
    "اكتئاب": "depression", "ربو": "asthma", "كحة": "cough",
    # ── Darija (marocain) ────────────────────────────
    "سخانة": "fever", "سخانا": "fever", "أسخانة": "fever",
    "سخونة": "fever", "تسخينة": "fever", "حرارة": "fever",
    "راسي كيدور": "headache", "داكشي فراسي": "headache",
    "كيولعني راسي": "headache", "فراسي": "headache",
    "ولع": "pain", "كيولع": "pain", "كتولعني": "pain",
    "كحا": "cough", "كاحة": "cough",
    "دايخ": "dizziness", "دايخة": "dizziness", "دوخة": "dizziness",
    "تقيأ": "nausea", "غاشية": "nausea", "قيا": "nausea",
    "رشة": "diarrhea",
    "ضغط الدم": "blood pressure", "ضغط دموي": "blood pressure",
    "السكري": "diabetes", "مرض السكر": "diabetes",
    "حساسية لدواء": "allergy", "الحساسية": "allergy",
    # ── Français (expressions complètes d'abord) ─────
    "j'ai mal à la tête": "headache", "mal à la tête": "headache",
    "maux de tête": "headache", "mal de tête": "headache",
    "migraine": "headache",
    "j'ai de la fièvre": "fever", "j'ai chaud": "fever",
    "je tousse": "cough", "j'ai la toux": "cough",
    "mal au ventre": "nausea", "douleur abdominale": "nausea",
    "tête qui tourne": "dizziness", "j'ai des vertiges": "dizziness",
    "j'ai la diarrhée": "diarrhea",
    # ── Français (mots simples) ───────────────────────
    "fièvre": "fever", "fievre": "fever", "température": "fever",
    "toux": "cough", "douleur": "pain", "mal": "pain",
    "nausée": "nausea", "nausee": "nausea", "vomissement": "nausea",
    "allergie": "allergy", "infection": "infection",
    "diabète": "diabetes", "diabete": "diabetes",
    "anxiété": "anxiety", "anxiete": "anxiety",
    "dépression": "depression", "depression": "depression",
    "asthme": "asthma", "vertige": "dizziness", "diarrhée": "diarrhea",
}

PREGNANCY_KEYWORDS = [
    "حامل", "حمل", "حوامل", "enceinte", "grossesse",
    "pregnant", "pregnancy", "حبلى"
]
BREASTFEEDING_KEYWORDS = [
    "مرضعة", "رضاعة", "مرضع", "allaitement", "allaitante",
    "breastfeeding", "breastfeed"
]


def normalize_text(text):
    return ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    ).lower()


def parse_age(age_str):
    if not age_str:
        return None
    match = re.search(r'\b(\d{1,3})\b', str(age_str))
    return int(match.group(1)) if match else None


def extract_symptom(question):
    q_lower = question.lower()
    q_normalized = normalize_text(question)

    # Patterns longs vérifiés en premier pour éviter "mal" avant "mal à la tête"
    for symptom_local, symptom_en in sorted(SYMPTOMS.items(), key=lambda x: len(x[0]), reverse=True):
        if (symptom_local.lower() in q_lower or
                normalize_text(symptom_local) in q_normalized):
            return symptom_en, symptom_local
    return None, None


def check_allergy_conflict(med_data, allergy_text, language="ar"):
    if not allergy_text:
        return None
    allergy_lower = allergy_text.lower()
    allergy_normalized = normalize_text(allergy_text)
    if any(kw in allergy_lower for kw in ["لا", "non", "no", "rien", "aucune", "makayn", "ma3ndish"]):
        return None
    for kw in med_data.get("allergie_keywords", []):
        if kw.lower() in allergy_lower or normalize_text(kw) in allergy_normalized:
            if language == "fr":
                return f"⚠️ Allergie : vous avez mentionné une allergie à '{kw}' — ce médicament pourrait ne pas vous convenir"
            else:
                return f"⚠️ تحذير حساسية: ذكرت حساسية من '{kw}' — هذا الدواء قد يكون غير مناسب لك"
    return None


def build_safety_warnings(med_data, age_int, allergy_text, other_symptoms, language="ar"):
    warnings = []
    age_min = med_data.get("age_min", 0)

    if age_int is not None and age_int < age_min:
        if language == "fr":
            warnings.append(f"⛔ Ce médicament est interdit aux moins de {age_min} ans (votre âge : {age_int} ans)")
        else:
            warnings.append(f"⛔ هذا الدواء ممنوع لمن هم دون {age_min} سنة (عمرك: {age_int} سنة)")

    # Alerte personnes âgées + AINS (ibuprofène, diclofénac, aspirine)
    if age_int is not None and age_int >= 65:
        if "ains" in [kw.lower() for kw in med_data.get("allergie_keywords", [])]:
            if language == "fr":
                warnings.append("⚠️ Après 65 ans, les anti-inflammatoires (AINS) sont à utiliser avec précaution — risque rénal et cardiovasculaire accru. Préférez le paracétamol si possible.")
            else:
                warnings.append("⚠️ بعد سن 65، تُستخدم مضادات الالتهاب بحذر شديد — خطر على الكلى والقلب. يُفضَّل الباراسيتامول إن أمكن.")

    combined_text = " ".join(filter(None, [allergy_text, other_symptoms, ""])).lower()

    if med_data.get("interdit_grossesse") and any(kw in combined_text for kw in PREGNANCY_KEYWORDS):
        if language == "fr":
            warnings.append("❌ Ce médicament est contre-indiqué pendant la grossesse")
        else:
            warnings.append("❌ هذا الدواء ممنوع للحوامل")

    if med_data.get("interdit_allaitement") and any(kw in combined_text for kw in BREASTFEEDING_KEYWORDS):
        if language == "fr":
            warnings.append("❌ Ce médicament est contre-indiqué pendant l'allaitement")
        else:
            warnings.append("❌ هذا الدواء ممنوع للمرضعات")

    allergy_warning = check_allergy_conflict(med_data, allergy_text, language=language)
    if allergy_warning:
        warnings.append(allergy_warning)

    return warnings


# ── Charger le dataset local ──────────────────────────
import os
_BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB_PATH = os.path.join(_BASE, "data", "morocco_medications.json")
with open(DB_PATH, 'r', encoding='utf-8') as f:
    MEDICATIONS_DB = json.load(f)


def get_fda_info(medication):
    return MEDICATIONS_DB.get(medication)


def get_medication_info(question, age=None, allergy=None, other_symptoms=None, language="ar"):
    symptom_en, symptom_local = extract_symptom(question)
    age_int = parse_age(age)
    if not symptom_en:
        return {
            "symptom_detected": None,
            "medications":      [],
            "message":          "Aucun symptôme détecté"
        }

    suggested = MOROCCO_MEDICATIONS.get(symptom_en, [])

    # Filtrer les médicaments interdits par âge
    if age_int is not None:
        valid = [m for m in suggested if MEDICATIONS_DB.get(m, {}).get("age_min", 0) <= age_int]
        filtered = valid if valid else suggested[:1]
    else:
        filtered = suggested

    medications_info = []

    for med in filtered[:3]:
        med_data = MEDICATIONS_DB.get(med)
        if not med_data:
            continue

        suffix = "fr" if language == "fr" else "ar"
        dosage_raw    = med_data.get(f"dosage_{suffix}", med_data.get("dosage_ar", ""))
        dosage_enfant = med_data.get(f"dosage_enfant_{suffix}", med_data.get("dosage_enfant_ar", ""))
        side_effects  = med_data.get(f"side_effects_{suffix}", med_data.get("side_effects_ar", ""))
        warnings_text = med_data.get(f"warnings_{suffix}", med_data.get("warnings_ar", ""))
        do_not_use    = med_data.get(f"do_not_use_{suffix}", med_data.get("do_not_use_ar", "-"))
        ask_doctor    = med_data.get(f"ask_doctor_{suffix}", med_data.get("ask_doctor_ar", "-"))

        # Choisir la bonne dose selon l'âge
        if age_int is not None and age_int < 18 and dosage_enfant:
            dosage_display = dosage_enfant
        else:
            dosage_display = dosage_raw

        safety_warnings = build_safety_warnings(med_data, age_int, allergy, other_symptoms, language=language)

        medications_info.append({
            "name":              med,
            "nom_ar":            med_data.get("nom_ar", med),
            "age_min":           med_data.get("age_min", 0),
            "interdit_grossesse":    med_data.get("interdit_grossesse", False),
            "interdit_allaitement":  med_data.get("interdit_allaitement", False),
            "dosage":            dosage_display,
            "dosage_ar":         dosage_display,
            "side_effects":      side_effects,
            "side_effects_ar":   side_effects,
            "warnings":          warnings_text,
            "warnings_ar":       warnings_text,
            "do_not_use":        do_not_use,
            "do_not_use_ar":     do_not_use,
            "ask_doctor":        ask_doctor,
            "ask_doctor_ar":     ask_doctor,
            "safety_warnings":   safety_warnings,
        })

    return {
        "symptom_detected": symptom_local,
        "symptom_en":       symptom_en,
        "medications":      medications_info
    }
