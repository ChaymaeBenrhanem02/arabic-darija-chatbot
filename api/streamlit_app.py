import streamlit as st
import requests

# ── faster-whisper (pip install faster-whisper) ───────
try:
    from faster_whisper import WhisperModel as _FasterWhisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

# ── Configuration ─────────────────────────────────────
st.set_page_config(
    page_title="Chatbot Médical Darija",
    page_icon="🏥",
    layout="centered"
)

API_URL = "http://127.0.0.1:8000/predict"

# ── Style CSS ─────────────────────────────────────────
st.markdown("""
<style>
    .chat-message-user {
        background-color: #0084ff;
        color: white;
        padding: 10px 15px;
        border-radius: 15px 15px 0px 15px;
        margin: 5px 0;
        max-width: 70%;
        float: right;
        clear: both;
    }
    .chat-message-bot {
        background-color: #f0f0f0;
        color: black;
        padding: 10px 15px;
        border-radius: 15px 15px 15px 0px;
        margin: 5px 0;
        max-width: 70%;
        float: left;
        clear: both;
    }
    .clearfix { clear: both; }
    .lang-btn {
        font-size: 18px;
        padding: 20px;
        border-radius: 16px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ── Questions par langue ───────────────────────────────
QUESTIONS_LANGS = {
    "darija": [
        "السلام! أنا المساعد الطبي ديالك 🏥\nكيفاش نقدر نعاونك اليوم؟",
        "منذ متى وعندك هاد المشكل؟ (مثال: من يوم، من أسبوع...)",
        "شحال عمرك؟",
        "واش عندك حساسية لأي دواء؟ (إذا لا، كتب 'لا')",
        "واش عندك أعراض أخرى؟ (مثال: حمى، غثيان، دوار...)",
    ],
    "ar": [
        "مرحباً! أنا مساعدك الطبي 🏥\nكيف يمكنني مساعدتك اليوم؟",
        "منذ متى وأنت تعاني من هذه المشكلة؟",
        "كم عمرك؟",
        "هل لديك حساسية لأي دواء؟ (إذا لا، اكتب 'لا')",
        "هل لديك أعراض أخرى؟ (مثال: حمى، غثيان، دوار...)",
    ],
    "fr": [
        "Bonjour ! Je suis votre assistant médical 🏥\nComment puis-je vous aider aujourd'hui ?",
        "Depuis combien de temps avez-vous ce problème ?",
        "Quel est votre âge ?",
        "Avez-vous des allergies médicamenteuses ? (Si non, écrivez 'non')",
        "Avez-vous d'autres symptômes ? (ex: fièvre, nausée, vertiges...)",
    ],
}

# Langue Whisper : darija → "ar", fr → "fr", ar → "ar"
WHISPER_LANG = {"darija": "ar", "ar": "ar", "fr": "fr"}

# Labels d'interface par langue
UI = {
    "darija": {
        "placeholder": "مثال: عندي صداع شديد",
        "send":        "إرسال ➤",
        "record":      "🎤 سجل رسالتك",
        "recognized":  "🗣️ تم التعرف على:",
        "confirm":     "✅ إرسال",
        "retry":       "🔄 إعادة",
        "analyzing":   "⏳ كنحلل معلوماتك...",
        "new_chat":    "🔄 محادثة جديدة",
        "no_meds":     "ما كاين معلومات عن الأدوية لهاد العرض.",
        "disclaimer":  "هاد المعلومات ماشي بديل على الطبيب. استشر الطبيب دائماً قبل أخذ أي دواء.",
        "result":      "✅ نتيجة التشخيص",
    },
    "ar": {
        "placeholder": "مثال: عندي صداع شديد",
        "send":        "إرسال ➤",
        "record":      "🎤 سجل رسالتك",
        "recognized":  "🗣️ تم التعرف على:",
        "confirm":     "✅ إرسال",
        "retry":       "🔄 إعادة",
        "analyzing":   "⏳ جاري تحليل بياناتك...",
        "new_chat":    "🔄 محادثة جديدة",
        "no_meds":     "لا توجد معلومات عن الأدوية لهذا العرض.",
        "disclaimer":  "هذه المعلومات ليست بديلاً عن الطبيب. استشر طبيبك دائماً قبل تناول أي دواء.",
        "result":      "✅ نتيجة التشخيص",
    },
    "fr": {
        "placeholder": "Ex: j'ai mal à la tête",
        "send":        "Envoyer ➤",
        "record":      "🎤 Enregistrer",
        "recognized":  "🗣️ Reconnu :",
        "confirm":     "✅ Envoyer",
        "retry":       "🔄 Réessayer",
        "analyzing":   "⏳ Analyse en cours...",
        "new_chat":    "🔄 Nouvelle conversation",
        "no_meds":     "Aucune information médicament pour ce symptôme.",
        "disclaimer":  "Ces informations ne remplacent pas un médecin. Consultez toujours un professionnel de santé.",
        "result":      "✅ Résultat du diagnostic",
    },
}

KEYS = ["symptom", "duration", "age", "allergy", "other_symptoms"]

# ── Whisper helpers ────────────────────────────────────
if WHISPER_AVAILABLE:
    import tempfile, os

    @st.cache_resource(show_spinner="جاري تحميل نموذج التعرف على الصوت...")
    def load_asr():
        return _FasterWhisper("small", device="cpu", compute_type="int16")

    def transcribe(audio_bytes: bytes, lang: str = "ar") -> str:
        model = load_asr()
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.write(audio_bytes)
        tmp.close()
        try:
            segments, _ = model.transcribe(
                tmp.name,
                beam_size=3,
                language=lang,
                condition_on_previous_text=False,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 300},
            )
            return "".join(seg.text for seg in segments).strip()
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass

# ── Initialiser session ───────────────────────────────
for key, default in [
    ("language",          None),   # None = pas encore choisi
    ("messages",          []),
    ("step",              0),
    ("user_data",         {}),
    ("conversation_done", False),
    ("voice_transcript",  None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Titre + sélecteur de langue (popover) ─────────────
LANG_LABELS = {None: "🌐 Langue", "darija": "🇲🇦 Darija", "ar": "🌍 العربية", "fr": "🇫🇷 Français"}

col_title, col_lang = st.columns([4, 1])
with col_title:
    st.title("🏥 Chatbot Médical")
with col_lang:
    st.markdown("<br>", unsafe_allow_html=True)
    with st.popover(LANG_LABELS[st.session_state.language], use_container_width=True):
        st.markdown("**اختر لغتك | Choisissez :**")
        for _code, _label in [("darija", "🇲🇦 Darija"), ("ar", "🌍 العربية"), ("fr", "🇫🇷 Français")]:
            if st.button(_label, key=f"pop_{_code}", use_container_width=True):
                st.session_state.language          = _code
                st.session_state.messages          = []
                st.session_state.step              = 0
                st.session_state.user_data         = {}
                st.session_state.conversation_done = False
                st.rerun()

st.markdown("---")

if st.session_state.language is None:
    st.markdown(
        "<p style='text-align:center;font-size:20px;margin-top:50px'>"
        "👆 اضغط على <b>🌐 Langue</b> لاختيار لغتك"
        "<br><br>Cliquez sur <b>🌐 Langue</b> en haut à droite pour commencer"
        "</p>",
        unsafe_allow_html=True
    )
    st.stop()

# ── Conversation ──────────────────────────────────────
lang      = st.session_state.language
ui        = UI[lang]
QUESTIONS = QUESTIONS_LANGS[lang]
wlang     = WHISPER_LANG[lang]


def render_result(msg):
    data      = msg["data"]
    user_data = msg["user_data"]
    u         = UI[msg.get("lang", "darija")]

    st.markdown(f"### {u['result']}")

    col1, col2 = st.columns(2)
    with col1:
        st.info(f"🔍 **{'الأعراض' if lang != 'fr' else 'Symptômes'}**\n\n{user_data.get('symptom', '-')}")
        st.info(f"👤 **{'العمر' if lang != 'fr' else 'Âge'}**\n\n{user_data.get('age', '-')}")
    with col2:
        st.info(f"⏱️ **{'المدة' if lang != 'fr' else 'Durée'}**\n\n{user_data.get('duration', '-')}")
        st.info(f"⚠️ **{'الحساسية' if lang != 'fr' else 'Allergie'}**\n\n{user_data.get('allergy', '-')}")

    st.markdown("---")

    col3, col4 = st.columns([2, 1])
    with col3:
        st.metric(label="🏥 التخصص المقترح" if lang != "fr" else "🏥 Spécialité",
                  value=data.get("specialty", "-"))
    with col4:
        st.metric(label="🎯 نسبة الثقة" if lang != "fr" else "🎯 Confiance",
                  value=f"{round(data.get('confidence', 0) * 100, 1)}%")

    st.progress(float(data.get("confidence", 0)))

    if data.get("symptom_detected"):
        lbl = "العرض المكتشف" if lang != "fr" else "Symptôme détecté"
        st.markdown(f"**💊 {lbl}:** `{data.get('symptom_detected')}`")

    medications = data.get("medications", [])
    if medications:
        st.markdown("---")
        st.markdown("### 💊 " + ("الأدوية المقترحة" if lang != "fr" else "Médicaments suggérés"))
        for med in medications:
            with st.expander(f"🔹 {med['name']} — {med.get('nom_ar') or med['name']}", expanded=True):
                safety = med.get("safety_warnings", [])
                if safety:
                    for w in safety:
                        st.error(w)
                    st.divider()

                flags = []
                if med.get("age_min", 0) > 0:
                    flags.append(f"⛔ {'دون' if lang != 'fr' else 'Moins de'} {med['age_min']} {'سنة' if lang != 'fr' else 'ans'}")
                if med.get("interdit_grossesse"):
                    flags.append("❌ " + ("الحوامل" if lang != "fr" else "Femmes enceintes"))
                if med.get("interdit_allaitement"):
                    flags.append("❌ " + ("المرضعات" if lang != "fr" else "Allaitement"))
                if flags:
                    lbl = "ممنوع على" if lang != "fr" else "Interdit pour"
                    st.markdown(f"**{lbl}:** " + " · ".join(flags))
                    st.divider()

                _dosage   = med.get('dosage')       or med.get('dosage_ar',       '-')
                _side_eff = med.get('side_effects') or med.get('side_effects_ar', '')
                _no_use   = med.get('do_not_use')   or med.get('do_not_use_ar',   '-')
                _warnings = med.get('warnings')     or med.get('warnings_ar',     '-')
                _doctor   = med.get('ask_doctor')   or med.get('ask_doctor_ar',   '-')

                st.success(f"📋 **{'الجرعة' if lang != 'fr' else 'Posologie'}**\n\n{_dosage}")
                st.divider()
                if _side_eff:
                    st.info(f"💊 **{'الآثار الجانبية' if lang != 'fr' else 'Effets secondaires'}**\n\n{_side_eff}")
                    st.divider()
                st.error(f"❌ **{'لا تستخدم إذا' if lang != 'fr' else 'Ne pas utiliser si'}**\n\n{_no_use}")
                st.divider()
                st.warning(f"⚠️ **{'تحذيرات' if lang != 'fr' else 'Avertissements'}**\n\n{_warnings}")
                st.divider()
                st.info(f"👨‍⚕️ **{'استشر الطبيب' if lang != 'fr' else 'Consulter si'}**\n\n{_doctor}")
    else:
        st.info(u["no_meds"])

    st.warning(u["disclaimer"])


# ── Afficher historique ───────────────────────────────
for msg in st.session_state.messages:
    if msg["role"] == "bot":
        if msg.get("type") == "result":
            render_result(msg)
        else:
            st.markdown(
                f'<div class="chat-message-bot">🤖 {msg["content"]}</div>'
                f'<div class="clearfix"></div>',
                unsafe_allow_html=True
            )
    else:
        st.markdown(
            f'<div class="chat-message-user">{msg["content"]}</div>'
            f'<div class="clearfix"></div>',
            unsafe_allow_html=True
        )

# ── Logique conversationnelle ─────────────────────────
step = st.session_state.step

if step == 0 and not st.session_state.messages:
    st.session_state.messages.append({"role": "bot", "content": QUESTIONS[0]})
    st.rerun()

# ── Zone de saisie ────────────────────────────────────
if not st.session_state.conversation_done:

    with st.form(key=f"form_{step}", clear_on_submit=True):
        prefill    = st.session_state.pop("voice_transcript", None) or ""
        user_input = st.text_input("", value=prefill, placeholder=ui["placeholder"])
        submitted  = st.form_submit_button(ui["send"])

    if WHISPER_AVAILABLE:
        st.markdown("---")
        audio_data = st.audio_input(ui["record"], key=f"audio_{step}")
        if audio_data is not None:
            with st.spinner("⏳ ..."):
                transcript = transcribe(audio_data.getvalue(), lang=wlang)
            if transcript:
                st.info(f"{ui['recognized']} **{transcript}**")
                col_ok, col_retry = st.columns(2)
                with col_ok:
                    if st.button(ui["confirm"], key=f"ws_{step}", use_container_width=True):
                        user_input = transcript
                        submitted  = True
                with col_retry:
                    if st.button(ui["retry"], key=f"wr_{step}", use_container_width=True):
                        st.rerun()
    else:
        st.caption("🎤 `pip install faster-whisper`")

    if submitted and user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        if step <= len(KEYS) - 1:
            st.session_state.user_data[KEYS[step]] = user_input

        st.session_state.step += 1
        next_step = st.session_state.step

        if next_step < len(QUESTIONS):
            st.session_state.messages.append({
                "role": "bot", "content": QUESTIONS[next_step]
            })
        else:
            st.session_state.messages.append({
                "role": "bot", "content": ui["analyzing"]
            })
            try:
                question = st.session_state.user_data.get("symptom", "")
                response = requests.post(API_URL, json={
                    "question":       question,
                    "age":            st.session_state.user_data.get("age", ""),
                    "allergy":        st.session_state.user_data.get("allergy", ""),
                    "other_symptoms": st.session_state.user_data.get("other_symptoms", ""),
                    "duration":       st.session_state.user_data.get("duration", ""),
                    "ui_language":    "fr" if lang == "fr" else "ar",
                })
                data = response.json()
                st.session_state.messages.append({
                    "role":      "bot",
                    "type":      "result",
                    "lang":      lang,
                    "data":      data,
                    "user_data": st.session_state.user_data.copy()
                })
            except Exception as e:
                st.session_state.messages.append({
                    "role": "bot", "content": f"❌ {e}"
                })
            st.session_state.conversation_done = True

        st.rerun()

# ── Bouton recommencer ────────────────────────────────
if st.session_state.conversation_done:
    if st.button(ui["new_chat"]):
        st.session_state.messages          = []
        st.session_state.step              = 0
        st.session_state.user_data         = {}
        st.session_state.conversation_done = False
        st.session_state.voice_transcript  = None
        st.session_state.language          = None  # retour à la sélection de langue
        st.rerun()
