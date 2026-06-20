"""
Tests pour la détection de symptômes et le parsing d'âge.
Couvre : extract_symptom(), parse_age(), normalize_text()
"""
import pytest
from nlp.ner.medication_extractor import extract_symptom, parse_age, normalize_text


class TestExtractSymptomFrancais:
    def test_phrase_longue_mal_a_la_tete(self):
        symptom, _ = extract_symptom("j'ai mal à la tête")
        assert symptom == "headache"

    def test_maux_de_tete(self):
        symptom, _ = extract_symptom("maux de tête")
        assert symptom == "headache"

    def test_migraine(self):
        symptom, _ = extract_symptom("j'ai une migraine")
        assert symptom == "headache"

    def test_fievre_phrase(self):
        symptom, _ = extract_symptom("j'ai de la fièvre")
        assert symptom == "fever"

    def test_fievre_simple(self):
        symptom, _ = extract_symptom("fièvre")
        assert symptom == "fever"

    def test_toux(self):
        symptom, _ = extract_symptom("j'ai la toux")
        assert symptom == "cough"

    def test_douleur(self):
        symptom, _ = extract_symptom("j'ai une douleur")
        assert symptom == "pain"

    def test_diarrhee(self):
        symptom, _ = extract_symptom("j'ai la diarrhée")
        assert symptom == "diarrhea"

    def test_phrase_longue_avant_courte(self):
        # "mal à la tête" doit être détecté avant "mal" → headache, pas pain
        symptom, _ = extract_symptom("j'ai mal à la tête")
        assert symptom == "headache"

    def test_majuscules(self):
        symptom, _ = extract_symptom("FIÈVRE")
        assert symptom == "fever"


class TestExtractSymptomArabe:
    def test_sudaa_headache(self):
        symptom, _ = extract_symptom("عندي صداع")
        assert symptom == "headache"

    def test_huma_fever(self):
        symptom, _ = extract_symptom("عندي حمى")
        assert symptom == "fever"

    def test_saal_cough(self):
        symptom, _ = extract_symptom("عندي سعال")
        assert symptom == "cough"

    def test_ghathayan_nausea(self):
        symptom, _ = extract_symptom("أشعر بغثيان")
        assert symptom == "nausea"

    def test_retourne_le_mot_local(self):
        _, detected = extract_symptom("عندي صداع")
        assert detected == "صداع"


class TestExtractSymptomDarija:
    def test_skhana_fever(self):
        symptom, _ = extract_symptom("عندي سخانة")
        assert symptom == "fever"

    def test_rasi_kidour_headache(self):
        symptom, _ = extract_symptom("راسي كيدور")
        assert symptom == "headache"

    def test_kahha_cough(self):
        symptom, _ = extract_symptom("عندي كحا")
        assert symptom == "cough"

    def test_dukha_dizziness(self):
        symptom, _ = extract_symptom("عندي دوخة")
        assert symptom == "dizziness"


class TestExtractSymptomAucun:
    def test_bonjour_aucun(self):
        symptom, detected = extract_symptom("bonjour comment ça va")
        assert symptom is None
        assert detected is None

    def test_chaine_vide(self):
        symptom, detected = extract_symptom("")
        assert symptom is None

    def test_texte_non_medical(self):
        symptom, _ = extract_symptom("le temps est beau aujourd'hui")
        assert symptom is None


class TestParseAge:
    def test_nombre_seul(self):
        assert parse_age("35") == 35

    def test_age_avec_ans(self):
        assert parse_age("35 ans") == 35

    def test_age_dans_phrase(self):
        assert parse_age("j'ai 7 ans") == 7

    def test_chaine_vide(self):
        assert parse_age("") is None

    def test_none(self):
        assert parse_age(None) is None

    def test_enfant(self):
        assert parse_age("5") == 5

    def test_personne_agee(self):
        assert parse_age("70 ans") == 70

    def test_age_arabe(self):
        assert parse_age("عمري 30 سنة") == 30


class TestNormalizeText:
    def test_supprime_accents(self):
        assert normalize_text("fièvre") == "fievre"

    def test_minuscules(self):
        assert normalize_text("FIÈVRE") == "fievre"

    def test_e_accent_grave(self):
        assert normalize_text("nausée") == "nausee"

    def test_arabe_inchange(self):
        result = normalize_text("صداع")
        assert len(result) > 0
