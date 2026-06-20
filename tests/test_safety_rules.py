"""
Tests pour les règles de sécurité médicale.
Couvre : build_safety_warnings(), check_allergy_conflict()
"""
import pytest
from nlp.ner.medication_extractor import build_safety_warnings, check_allergy_conflict

# ── Données de test (médicaments fictifs simplifiés) ─────────────────────────

PARACETAMOL = {
    "age_min": 0,
    "allergie_keywords": ["paracetamol", "acetaminophene"],
    "interdit_grossesse": False,
    "interdit_allaitement": False,
}

IBUPROFEN = {
    "age_min": 3,
    "allergie_keywords": ["ibuprofen", "ibuprofene", "ains"],
    "interdit_grossesse": True,
    "interdit_allaitement": False,
}

ASPIRIN = {
    "age_min": 16,
    "allergie_keywords": ["aspirine", "aspirin", "ains"],
    "interdit_grossesse": True,
    "interdit_allaitement": True,
}


# ── Tests restriction d'âge ──────────────────────────────────────────────────

class TestRestrictionAge:
    def test_enfant_trop_jeune_pour_aspirine(self):
        warnings = build_safety_warnings(ASPIRIN, age_int=10, allergy_text="", other_symptoms="")
        assert any("⛔" in w for w in warnings)

    def test_bebe_trop_jeune_pour_ibuprofen(self):
        warnings = build_safety_warnings(IBUPROFEN, age_int=1, allergy_text="", other_symptoms="")
        assert any("⛔" in w for w in warnings)

    def test_adulte_ok_pour_aspirine(self):
        warnings = build_safety_warnings(ASPIRIN, age_int=25, allergy_text="", other_symptoms="")
        assert not any("⛔" in w for w in warnings)

    def test_bebe_ok_pour_paracetamol(self):
        # paracetamol age_min = 0, donc autorisé pour tous
        warnings = build_safety_warnings(PARACETAMOL, age_int=1, allergy_text="", other_symptoms="")
        assert not any("⛔" in w for w in warnings)

    def test_message_en_francais(self):
        warnings = build_safety_warnings(ASPIRIN, age_int=10, allergy_text="", other_symptoms="", language="fr")
        assert any("interdit" in w.lower() for w in warnings)

    def test_message_en_arabe(self):
        warnings = build_safety_warnings(ASPIRIN, age_int=10, allergy_text="", other_symptoms="", language="ar")
        assert any("ممنوع" in w for w in warnings)


# ── Tests alerte personnes âgées + AINS ─────────────────────────────────────

class TestAINSPersonnesAgees:
    def test_65_ans_ibuprofen_alerte(self):
        warnings = build_safety_warnings(IBUPROFEN, age_int=70, allergy_text="", other_symptoms="")
        assert any("⚠️" in w for w in warnings)

    def test_65_ans_aspirine_alerte(self):
        warnings = build_safety_warnings(ASPIRIN, age_int=65, allergy_text="", other_symptoms="")
        assert any("⚠️" in w for w in warnings)

    def test_64_ans_pas_alerte_ains(self):
        warnings = build_safety_warnings(IBUPROFEN, age_int=64, allergy_text="", other_symptoms="")
        ains = [w for w in warnings if "مضادات" in w or "anti-inflam" in w.lower() or "AINS" in w]
        assert len(ains) == 0

    def test_personne_agee_paracetamol_pas_alerte_ains(self):
        # paracetamol n'a pas "ains" dans ses allergie_keywords
        warnings = build_safety_warnings(PARACETAMOL, age_int=70, allergy_text="", other_symptoms="")
        ains = [w for w in warnings if "مضادات" in w or "AINS" in w]
        assert len(ains) == 0


# ── Tests grossesse ──────────────────────────────────────────────────────────

class TestGrossesse:
    def test_mot_cle_arabe_حامل(self):
        warnings = build_safety_warnings(IBUPROFEN, age_int=28, allergy_text="حامل", other_symptoms="")
        assert any("❌" in w for w in warnings)

    def test_mot_cle_francais_enceinte(self):
        warnings = build_safety_warnings(IBUPROFEN, age_int=28, allergy_text="enceinte", other_symptoms="", language="fr")
        assert any("grossesse" in w.lower() for w in warnings)

    def test_mot_cle_dans_autres_symptomes(self):
        warnings = build_safety_warnings(IBUPROFEN, age_int=28, allergy_text="", other_symptoms="je suis enceinte")
        assert any("❌" in w for w in warnings)

    def test_sans_mot_cle_grossesse(self):
        warnings = build_safety_warnings(IBUPROFEN, age_int=28, allergy_text="لا", other_symptoms="")
        grossesse = [w for w in warnings if "حوامل" in w or "grossesse" in w]
        assert len(grossesse) == 0

    def test_paracetamol_autorise_grossesse(self):
        # paracetamol interdit_grossesse = False
        warnings = build_safety_warnings(PARACETAMOL, age_int=28, allergy_text="حامل", other_symptoms="")
        assert not any("حوامل" in w or "grossesse" in w for w in warnings)


# ── Tests allaitement ────────────────────────────────────────────────────────

class TestAllaitement:
    def test_mot_cle_arabe_مرضعة(self):
        warnings = build_safety_warnings(ASPIRIN, age_int=28, allergy_text="مرضعة", other_symptoms="")
        assert any("❌" in w for w in warnings)

    def test_mot_cle_francais_allaitement(self):
        warnings = build_safety_warnings(ASPIRIN, age_int=28, allergy_text="", other_symptoms="allaitement", language="fr")
        assert any("allaitement" in w.lower() for w in warnings)

    def test_ibuprofen_ok_allaitement(self):
        # ibuprofen interdit_allaitement = False
        warnings = build_safety_warnings(IBUPROFEN, age_int=28, allergy_text="مرضعة", other_symptoms="")
        allaitement = [w for w in warnings if "مرضعات" in w or "allaitement" in w]
        assert len(allaitement) == 0


# ── Tests conflit d'allergie ─────────────────────────────────────────────────

class TestAllergie:
    def test_allergie_detectee(self):
        warning = check_allergy_conflict(IBUPROFEN, "allergie à l'ibuprofene")
        assert warning is not None
        assert "⚠️" in warning

    def test_pas_allergie_laa(self):
        warning = check_allergy_conflict(IBUPROFEN, "لا")
        assert warning is None

    def test_pas_allergie_non(self):
        warning = check_allergy_conflict(IBUPROFEN, "non")
        assert warning is None

    def test_pas_allergie_rien(self):
        warning = check_allergy_conflict(IBUPROFEN, "rien")
        assert warning is None

    def test_pas_allergie_aucune(self):
        warning = check_allergy_conflict(IBUPROFEN, "aucune allergie")
        assert warning is None

    def test_message_francais(self):
        warning = check_allergy_conflict(IBUPROFEN, "ibuprofene", language="fr")
        assert warning is not None
        assert "Allergie" in warning

    def test_allergie_differente_pas_de_conflit(self):
        # allergie à la pénicilline, pas à l'ibuprofène
        warning = check_allergy_conflict(PARACETAMOL, "allergie pénicilline")
        assert warning is None

    def test_chaine_vide_pas_de_conflit(self):
        warning = check_allergy_conflict(IBUPROFEN, "")
        assert warning is None

    def test_none_pas_de_conflit(self):
        warning = check_allergy_conflict(IBUPROFEN, None)
        assert warning is None
