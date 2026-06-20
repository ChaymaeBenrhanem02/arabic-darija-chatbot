"""
Tests pour l'endpoint FastAPI POST /predict.
Le modèle BERT est mocké pour éviter le chargement lent depuis HuggingFace.
"""
import sys
import os
import torch
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── Mock du modèle BERT (doit être fait AVANT l'import de api.main) ──────────

_mock_tokenizer = MagicMock()
_mock_tokenizer.return_value = {
    "input_ids":      torch.tensor([[1, 2, 3]]),
    "attention_mask": torch.tensor([[1, 1, 1]]),
}

_mock_logits = torch.zeros(1, 24)
_mock_logits[0][3] = 5.0  # forte confiance pour l'index 3 = "general practitioner"

_mock_output = MagicMock()
_mock_output.logits = _mock_logits

_mock_model = MagicMock()
_mock_model.return_value = _mock_output

with patch("transformers.AutoTokenizer.from_pretrained", return_value=_mock_tokenizer), \
     patch("transformers.AutoModelForSequenceClassification.from_pretrained", return_value=_mock_model):
    from fastapi.testclient import TestClient
    from api.main import app

client = TestClient(app)


# ── Tests endpoint GET / ─────────────────────────────────────────────────────

class TestEndpointHome:
    def test_retourne_200(self):
        response = client.get("/")
        assert response.status_code == 200

    def test_contient_message(self):
        response = client.get("/")
        assert "message" in response.json()


# ── Tests endpoint POST /predict ─────────────────────────────────────────────

class TestEndpointPredict:
    def test_symptome_arabe_retourne_200(self):
        response = client.post("/predict", json={
            "question":       "عندي صداع",
            "age":            "35",
            "allergy":        "لا",
            "other_symptoms": "",
            "duration":       "",
            "ui_language":    "ar"
        })
        assert response.status_code == 200

    def test_champs_obligatoires_presents(self):
        response = client.post("/predict", json={
            "question":       "عندي صداع",
            "age":            "30",
            "allergy":        "",
            "other_symptoms": "",
            "duration":       "",
            "ui_language":    "ar"
        })
        data = response.json()
        for champ in ["specialty", "confidence", "symptom_detected", "medications", "language"]:
            assert champ in data, f"Champ manquant : {champ}"

    def test_symptome_arabe_detecte(self):
        response = client.post("/predict", json={
            "question":       "عندي صداع",
            "age":            "35",
            "allergy":        "لا",
            "other_symptoms": "",
            "duration":       "",
            "ui_language":    "ar"
        })
        data = response.json()
        assert data["symptom_detected"] is not None
        assert len(data["medications"]) > 0

    def test_symptome_francais_fievre(self):
        response = client.post("/predict", json={
            "question":       "j'ai de la fièvre",
            "age":            "25",
            "allergy":        "non",
            "other_symptoms": "",
            "duration":       "2 jours",
            "ui_language":    "fr"
        })
        data = response.json()
        assert data["symptom_detected"] is not None
        assert len(data["medications"]) > 0

    def test_symptome_darija_skhana(self):
        response = client.post("/predict", json={
            "question":       "عندي سخانة",
            "age":            "30",
            "allergy":        "",
            "other_symptoms": "",
            "duration":       "",
            "ui_language":    "ar"
        })
        data = response.json()
        assert data["symptom_detected"] is not None

    def test_aucun_symptome_liste_vide(self):
        response = client.post("/predict", json={
            "question":       "bonjour comment allez vous",
            "age":            "",
            "allergy":        "",
            "other_symptoms": "",
            "duration":       "",
            "ui_language":    "fr"
        })
        assert response.status_code == 200
        data = response.json()
        assert data["medications"] == [] or data["symptom_detected"] is None

    def test_medicament_contient_champs_requis(self):
        response = client.post("/predict", json={
            "question":       "عندي صداع",
            "age":            "30",
            "allergy":        "لا",
            "other_symptoms": "",
            "duration":       "",
            "ui_language":    "ar"
        })
        data = response.json()
        if data["medications"]:
            med = data["medications"][0]
            for champ in ["name", "dosage", "side_effects", "warnings", "safety_warnings"]:
                assert champ in med, f"Champ médicament manquant : {champ}"

    def test_age_enfant_filtre_aspirine(self):
        # L'aspirine est interdite aux moins de 16 ans
        response = client.post("/predict", json={
            "question":       "عندي صداع",
            "age":            "5",
            "allergy":        "",
            "other_symptoms": "",
            "duration":       "",
            "ui_language":    "ar"
        })
        data = response.json()
        noms = [m["name"] for m in data["medications"]]
        assert "aspirin" not in noms

    def test_grossesse_declenche_avertissement(self):
        response = client.post("/predict", json={
            "question":       "عندي صداع",
            "age":            "28",
            "allergy":        "حامل",
            "other_symptoms": "",
            "duration":       "",
            "ui_language":    "ar"
        })
        data = response.json()
        # Au moins un médicament (ibuprofen) doit porter un avertissement ❌
        tous_warnings = [
            w for med in data["medications"]
            for w in med.get("safety_warnings", [])
        ]
        assert any("❌" in w for w in tous_warnings)

    def test_confidence_est_un_nombre(self):
        response = client.post("/predict", json={
            "question":       "صداع",
            "age":            "30",
            "allergy":        "",
            "other_symptoms": "",
            "duration":       "",
            "ui_language":    "ar"
        })
        data = response.json()
        assert isinstance(data["confidence"], float)
        assert 0.0 <= data["confidence"] <= 1.0

    def test_ui_language_fr_retourne_contenu_francais(self):
        response = client.post("/predict", json={
            "question":       "fièvre",
            "age":            "30",
            "allergy":        "non",
            "other_symptoms": "",
            "duration":       "",
            "ui_language":    "fr"
        })
        data = response.json()
        if data["medications"]:
            dosage = data["medications"][0].get("dosage", "")
            # Le contenu français contient des mots latins
            assert any(c.isascii() and c.isalpha() for c in dosage)
