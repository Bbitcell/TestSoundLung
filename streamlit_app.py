"""SoundLung — voice age screening demo (Streamlit Community Cloud).

Runs entirely on NumPy + Praat (no PyTorch) so it fits a free hosting tier;
weights come from models.npz, exported with parity asserted against the
PyTorch originals.
"""
from pathlib import Path

import streamlit as st

from soundlung_model import SoundLungModels

HERE = Path(__file__).parent

st.set_page_config(page_title="SoundLung — Voice Age Screening",
                   page_icon="🎙️", layout="centered")


REQUIRED_FORMAT = 4      # keep in step with export_models.FORMAT_VERSION


@st.cache_resource
def load_models():
    """Loads the weights, failing with a readable message rather than a
    traceback when the three coupled files (streamlit_app.py,
    soundlung_model.py, models.npz) are not all at the same version."""
    have = getattr(SoundLungModels, "FORMAT_VERSION", 0)
    if have != REQUIRED_FORMAT:
        return None, (f"`soundlung_model.py` is version {have or 'pre-1'}, but this "
                      f"page needs version {REQUIRED_FORMAT}. Re-upload "
                      f"**soundlung_model.py** *and* **models.npz**.")
    try:
        m = SoundLungModels(HERE / "models.npz")
    except (ValueError, KeyError) as exc:
        detail = (f"`models.npz` has no `{exc.args[0]}` entry."
                  if isinstance(exc, KeyError) else str(exc))
        return None, (f"{detail} Re-upload **models.npz** (and "
                      f"**soundlung_model.py**) to the repository.")
    missing = [v for v in ("robust", "voice", "full") if v not in m.val_auc]
    if missing:
        return None, (f"`models.npz` is missing the {', '.join(missing)} model(s). "
                      f"Re-upload **models.npz**.")
    return m, None


M, load_error = load_models()
if load_error:
    st.error(f"**Model files are out of step.** {load_error}", icon="🧩")
    st.stop()

st.title("🎙️ SoundLung")
st.caption("Research demo — estimates whether a speaker is young (18–30) or "
           "old (60+) from a sustained vowel and a spoken sentence.")

st.info("**Not a clinical tool.** Models were trained on lab recordings from the "
        "Saarbrücken Voice Database. Consumer microphones shift the acoustic "
        "measurements enough to imitate ageing, so the app flags recordings that "
        "fall outside the range it was calibrated on.", icon="ℹ️")

# ------------------------------------------------------------- 1. recordings
st.subheader("1 · Sustained vowel")
st.markdown("Hold a steady **“aaaah”** for **4–5 seconds** (or as long as is "
            "comfortable), 10–15 cm from the microphone, in a quiet room. Only the "
            "**steadiest 1.5 seconds** are measured — matching the length of the "
            "clinical recordings the models were trained on — so the start, the end "
            "and anything your microphone does to the tail are discarded.")
with st.expander("Hear an example"):
    ca, cb = st.columns(2)
    ca.caption("female speaker"); ca.audio(str(HERE / "ref_vowel_F.wav"))
    cb.caption("male speaker"); cb.audio(str(HERE / "ref_vowel_M.wav"))
vowel = st.audio_input("Record the vowel", key="vowel")
vowel_up = st.file_uploader("…or upload a .wav", type=["wav"], key="vowel_up")

st.subheader("2 · Sentence — in German")
st.markdown("The models were trained on this exact sentence, so saying it in "
            "German gives the most reliable result:")
st.markdown("### „Guten Morgen, wie geht es Ihnen?“")
st.markdown("*(Good morning, how are you?)* — pronounced roughly "
            "**GOO-ten MOR-gen, vee gayt es EE-nen?** \n"
            "Say it in one breath at a natural pace (about 2 seconds).")
with st.expander("Hear the sentence", expanded=True):
    ca, cb = st.columns(2)
    ca.caption("female speaker"); ca.audio(str(HERE / "ref_phrase_F.wav"))
    cb.caption("male speaker"); cb.audio(str(HERE / "ref_phrase_M.wav"))
phrase = st.audio_input("Record the sentence", key="phrase")
phrase_up = st.file_uploader("…or upload a .wav", type=["wav"], key="phrase_up")

# ------------------------------------------------------------------ examples
st.subheader("3 · Analyse")
example = st.selectbox(
    "Use your own recordings, or run a known-age example from the training database",
    ["My recordings", "Example: woman aged 21", "Example: woman aged 76"])

col1, col2 = st.columns([1, 2])
go = col1.button("Analyse voice", type="primary", use_container_width=True)


def get_audio():
    """Returns (vowel, phrase, true_age). Sex is always detected from the audio."""
    if example.startswith("Example: woman aged 21"):
        return ((HERE / "example_young_vowel.wav").read_bytes(),
                (HERE / "example_young_phrase.wav").read_bytes(), 21)
    if example.startswith("Example: woman aged 76"):
        return ((HERE / "example_old_vowel.wav").read_bytes(),
                (HERE / "example_old_phrase.wav").read_bytes(), 76)
    v = vowel_up or vowel
    p = phrase_up or phrase
    if v is None or p is None:
        return None
    return v.getvalue(), p.getvalue(), None


if go:
    got = get_audio()
    if got is None:
        st.warning("Record (or upload) both the vowel and the sentence first.")
    else:
        v_bytes, p_bytes, true_age = got
        with st.spinner("Measuring the voice…"):
            try:
                res = M.analyse(None, v_bytes, p_bytes)   # sex detected from audio
            except Exception as exc:
                st.error(str(exc))
                st.stop()

        sex_info = res.get("sex")
        if sex_info:
            name = "male" if sex_info["detected"] == "M" else "female"
            conf = sex_info["prob_male"] if sex_info["detected"] == "M" else 1 - sex_info["prob_male"]
            if sex_info["confident"]:
                st.caption(f"Detected sex: **{name}** ({100 * conf:.0f}% confident) — "
                           f"from pitch, 97% accurate on validation.")
            else:
                st.warning(f"**Sex detected as {name}, but only {100 * conf:.0f}% "
                           f"confident** — this voice sits near the male/female pitch "
                           f"boundary, so the age result below may be affected.",
                           icon="⚧")

        if not res["reliable"]:
            st.warning(
                "**Outside calibrated range — treat this result as unreliable.** "
                "These measurements are unlike the lab recordings the models were "
                "fitted on, usually because of microphone and room differences "
                "rather than the speaker's age:\n\n"
                + "\n".join(f"- {o}" for o in res["out_of_range"]), icon="⚠️")

        verdict = res["label"]
        if true_age is not None:
            correct = (true_age >= 60) == (res["prob_old"] > 0.5)
            st.success(f"**{verdict}** — p(old) = {res['prob_old']:.3f}  \n"
                       f"True age: {true_age} — {'correct ✓' if correct else 'incorrect ✗'}")
        elif res["prob_old"] > 0.5:
            st.error(f"**{verdict}** — p(old) = {res['prob_old']:.3f}")
        else:
            st.success(f"**{verdict}** — p(old) = {res['prob_old']:.3f}")

        label_rate = res.get("label_rate")
        c1, c2, c3 = st.columns(3)
        c1.metric("Verdict — pitch only", verdict, f"p(old) = {res['prob_old']:.2f}",
                  delta_color="off",
                  help="F0, jitter and sex. Unaffected by microphone quality and by "
                       "how fast you speak. Validation AUC 0.70.")
        if label_rate:
            c2.metric("Including speaking rate", label_rate,
                      f"p(old) = {res['prob_old_rate']:.2f}", delta_color="off",
                      help="Adds the sentence. Best on the database (AUC 0.93) but "
                           "partly measures how fast you talk.")
        c3.metric("CNN age estimate", f"≈ {res['cnn_age']:.0f} years")

        if "probs" in res:
            st.caption("All three classifiers — accuracy on the database rises with "
                       "each addition, but so does sensitivity to the recording setup:")
            st.dataframe(
                {"model": ["pitch only (verdict)", "+ shimmer & HNR", "+ sentence"],
                 "immune to": ["microphone and speaking rate", "speaking rate", "—"],
                 "validation AUC": [f"{M.val_auc[v]:.2f}" for v in ("robust", "voice", "full")],
                 "p(old)": [f"{res['probs'][v]:.2f}" for v in ("robust", "voice", "full")],
                 "reads as": [res["labels"][v] for v in ("robust", "voice", "full")]},
                hide_index=True, use_container_width=True)

        if label_rate and res["label"] != label_rate:
            st.info(
                f"**The two models disagree, and that is informative.** The verdict "
                f"uses voice quality only. The rate-sensitive model reads "
                f"**{label_rate}**, driven largely by how long the sentence took "
                f"({res['features']['Phrase_Duration_s']:.2f} s — the database median is "
                f"1.7 s for young speakers and 2.5 s for those over 60). Speech does "
                f"slow with age, but rate is voluntary, so talking slowly reads as older "
                f"whatever your age.", icon="⏱️")

        drift = res.get("vowel_drift_db", 0.0)
        if abs(drift) > 6:
            st.warning(
                f"**Your microphone is processing the sound.** The vowel is "
                f"{abs(drift):.0f} dB {'louder' if drift > 0 else 'quieter'} at the end "
                f"than at the start — that is automatic gain control or noise "
                f"suppression, which mistakes a steady vowel for background noise and "
                f"ducks it after a couple of seconds. Only the steadiest 1.5 seconds were "
                f"measured, so the result is still usable, but for clean data turn the "
                f"processing off (Windows: Settings ▸ System ▸ Sound ▸ your microphone ▸ "
                f"turn **Audio enhancements** off).", icon="🎛️")

        hnr = res["features"]["HNR_dB"]
        if hnr < 20:
            st.warning(
                f"**Recording quality is limiting this result.** Vowel HNR is "
                f"{hnr:.1f} dB; clean recordings of young voices are typically "
                f"22–30 dB. Low HNR and the high shimmer that comes with it are "
                f"the signature the models read as an aged voice, so a distant or "
                f"noisy microphone can push a young speaker to “old”. Move closer, "
                f"silence fans, use a headset mic, and try again.", icon="🎤")

        with st.expander("Measured features", expanded=True):
            st.dataframe(
                {"feature": list(res["features"].keys()),
                 "value": [round(float(v), 3) for v in res["features"].values()]},
                hide_index=True, use_container_width=True)

st.divider()
st.caption(
    f"Classifiers: 10-model ensembles trained on the Saarbrücken Voice Database "
    f"(speaker-grouped split, decision thresholds tuned on validation). The verdict "
    f"uses pitch features only (AUC {M.val_auc['robust']:.2f}) because the more "
    f"accurate models lean on measurements that a consumer microphone distorts "
    f"(shimmer, HNR) or on sentence duration (AUC {M.val_auc['full']:.2f}), which "
    f"correlates +0.59 with age but is under the speaker's control. Age estimate: "
    f"small CNN over the sentence spectrogram. Example recordings © Saarbrücken Voice "
    f"Database, CC-BY 4.0. Recordings are analysed in memory and are not stored."
)
