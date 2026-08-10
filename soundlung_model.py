"""NumPy inference + feature extraction for the SoundLung cloud app.

Self-contained: no PyTorch. Loads models.npz (exported by export_models.py,
with parity against PyTorch asserted there) and reproduces the local app's
pipeline exactly - the same Praat measurements, the same silence trimming, the
same out-of-distribution guard.
"""
import io

import numpy as np
import parselmouth
import soundfile as sf
from parselmouth.praat import call

F0_MIN, F0_MAX = 75, 600
SR_CNN = 16000
CLIP_S, N_FFT, HOP, N_MELS = 4.5, 512, 256, 64


# ---------------------------------------------------------------- audio utils
def read_audio(data):
    samples, sr = sf.read(io.BytesIO(data), dtype="float64")
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    return samples - samples.mean(), sr


def trim_silence(samples, sr, drop_db=32.0, keep_s=0.04):
    """Training clips are cropped tight around the utterance; a browser
    recording is not. Untrimmed silence inflates Phrase_Duration_s - the
    strongest age feature - enough to make a young voice read as old."""
    win = max(int(0.02 * sr), 1)
    n = len(samples) // win
    if n < 3:
        return samples
    env = np.sqrt((samples[:n * win].reshape(n, win) ** 2).mean(axis=1))
    if env.max() <= 0:
        return samples
    loud = np.where(env > env.max() * 10 ** (-drop_db / 20))[0]
    if len(loud) == 0:
        return samples
    pad = int(keep_s * sr)
    start, end = max(0, loud[0] * win - pad), min(len(samples), (loud[-1] + 1) * win + pad)
    return samples[start:end] if end - start > win else samples


def stable_window(samples, sr, want_s=1.5):
    """Steadiest `want_s` of a sustained vowel. Clinical protocols measure the
    stable middle of a phonation anyway, and it side-steps microphone
    processing: noise suppression mistakes a steady tone for background noise
    and ducks it after a couple of seconds, and auto-gain rides the level.

    1.5 s matches the training clips (SVD /a/ is 1.36 s median) and perturbs the
    features least - see stable_window in ../app.py for the measurements."""
    n = int(want_s * sr)
    if len(samples) <= n:
        return samples
    win = max(int(0.02 * sr), 1)
    nf = len(samples) // win
    env = np.sqrt((samples[:nf * win].reshape(nf, win) ** 2).mean(axis=1))
    fpw = max(int(want_s / 0.02), 1)
    best, best_score = 0, np.inf
    for s0 in range(0, nf - fpw + 1, 2):
        seg = env[s0:s0 + fpw]
        if seg.mean() <= 0:
            continue
        score = seg.std() / seg.mean()
        if score < best_score:
            best_score, best = score, s0
    return samples[best * win: best * win + n]


def level_drift_db(samples, sr, edge_s=0.7):
    """Loudness change from the start to the end of a sustained vowel; a large
    value means the microphone or its driver is processing the signal."""
    n = int(edge_s * sr)
    if len(samples) < 3 * n:
        return 0.0
    first, last = np.sqrt(np.mean(samples[:n] ** 2)), np.sqrt(np.mean(samples[-n:] ** 2))
    if first <= 0 or last <= 0:
        return 0.0
    return float(20 * np.log10(last / first))


def to_sound(data, clip_transients=False, steady=False):
    samples, sr = read_audio(data)
    if clip_transients:
        ref = np.percentile(np.abs(samples), 99.0)
        if ref > 0:
            samples = np.clip(samples, -ref, ref)
    samples = trim_silence(samples, sr)
    if steady:
        samples = stable_window(samples, sr)
    peak = np.abs(samples).max()
    if peak > 0:
        samples = samples / peak
    return parselmouth.Sound(samples, sampling_frequency=sr)


def audio_stats(data):
    samples, sr = read_audio(data)
    rms = float(np.sqrt(np.mean(samples ** 2))) if len(samples) else 0.0
    return f"{len(samples) / sr:.1f}s, peak {np.abs(samples).max():.3f}, rms {rms:.4f}"


# ------------------------------------------------------------- Praat features
def vowel_features(snd, gender):
    if snd.duration < 0.3:
        raise ValueError(f"vowel too short ({snd.duration:.2f}s)")
    pitch = call(snd, "To Pitch", 0.0, F0_MIN, F0_MAX)
    mean_f0 = call(pitch, "Get mean", 0, 0, "Hertz")
    if np.isnan(mean_f0):
        raise ValueError("no pitch detected in the vowel")
    pp = call(snd, "To PointProcess (periodic, cc)", F0_MIN, F0_MAX)
    if call(pp, "Get number of points") < 10:
        raise ValueError("vowel: too few glottal pulses")
    jitter = call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3) * 100
    shimmer = call([snd, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6) * 100
    hnr = call(call(snd, "To Harmonicity (cc)", 0.01, F0_MIN, 0.1, 1.0), "Get mean", 0, 0)
    if any(np.isnan(v) for v in (jitter, shimmer, hnr)):
        raise ValueError("vowel: NaN in jitter/shimmer/HNR")
    return {"Mean_F0_Hz": mean_f0, "Jitter_percent": jitter,
            "Shimmer_percent": shimmer, "HNR_dB": hnr}


def phrase_features(snd):
    pitch = call(snd, "To Pitch", 0.0, F0_MIN, F0_MAX)
    f0 = pitch.selected_array["frequency"]
    voiced = f0[f0 > 0]
    if len(voiced) < 10:
        raise ValueError("phrase: too few voiced frames")
    hnr = call(call(snd, "To Harmonicity (cc)", 0.01, F0_MIN, 0.1, 1.0), "Get mean", 0, 0)
    return {"Phrase_F0_Mean_Hz": float(np.mean(voiced)),
            "Phrase_F0_SD_Hz": float(np.std(voiced)),
            "Phrase_HNR_dB": hnr,
            "Phrase_Duration_s": snd.duration,
            "Phrase_Voiced_Fraction": len(voiced) / len(f0)}


# ------------------------------------------------------------- log-mel (CNN)
def _mel_filterbank(sr=SR_CNN, n_fft=N_FFT, n_mels=N_MELS, fmin=50.0, fmax=8000.0):
    hz2mel = lambda f: 2595.0 * np.log10(1.0 + f / 700.0)
    mel2hz = lambda m: 700.0 * (10.0 ** (m / 2595.0) - 1.0)
    pts = mel2hz(np.linspace(hz2mel(fmin), hz2mel(fmax), n_mels + 2))
    bins = np.floor((n_fft + 1) * pts / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for i in range(n_mels):
        l, c, r = bins[i], bins[i + 1], bins[i + 2]
        if c > l:
            fb[i, l:c] = (np.arange(l, c) - l) / (c - l)
        if r > c:
            fb[i, c:r] = (r - np.arange(c, r)) / (r - c)
    return fb.astype(np.float32)


MEL_FB = _mel_filterbank()
WINDOW = np.hanning(N_FFT).astype(np.float32)


def log_mel(samples, sr):
    if sr != SR_CNN:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(int(sr), SR_CNN)
        samples = resample_poly(samples, SR_CNN // g, int(sr) // g)
    audio = np.asarray(samples, dtype=np.float32)
    peak = np.abs(audio).max()
    if peak > 0:
        audio = audio / peak
    target = int(CLIP_S * SR_CNN)
    audio = np.pad(audio[:target], (0, max(0, target - len(audio))))
    n_frames = 1 + (len(audio) - N_FFT) // HOP
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(n_frames)[:, None]
    power = np.abs(np.fft.rfft(audio[idx] * WINDOW, axis=1)) ** 2
    return np.log(power @ MEL_FB.T + 1e-8).T.astype(np.float32)


# ------------------------------------------------------------ numpy inference
def _conv2d(x, w, b):
    """3x3, stride 1, padding 1 — the only conv geometry SpecCNN uses."""
    c_out, c_in, kh, kw = w.shape
    _, h, ww = x.shape
    xp = np.pad(x, ((0, 0), (1, 1), (1, 1)))
    patches = np.empty((c_in, kh, kw, h, ww), dtype=np.float32)
    for i in range(kh):
        for j in range(kw):
            patches[:, i, j] = xp[:, i:i + h, j:j + ww]
    return np.einsum("cijhw,ocij->ohw", patches, w, optimize=True) + b[:, None, None]


def _batchnorm(x, w, b, mean, var, eps=1e-5):
    return (x - mean[:, None, None]) / np.sqrt(var[:, None, None] + eps) * \
        w[:, None, None] + b[:, None, None]


def _maxpool2(x):
    c, h, w = x.shape
    h2, w2 = h // 2, w // 2
    return x[:, :h2 * 2, :w2 * 2].reshape(c, h2, 2, w2, 2).max(axis=(2, 4))


class SoundLungModels:
    """Two classifiers: "voice" (vowel + sex; immune to speaking rate) and
    "full" (adds the sentence, including its duration — more accurate on the
    database but responds to how fast the speaker talks)."""

    FORMAT_VERSION = 3      # must match the stamp written by export_models.py
    VARIANTS = ("robust", "voice", "full")
    VERDICT = "robust"      # immune to both microphone quality and speaking rate

    def __init__(self, npz_path):
        z = np.load(npz_path, allow_pickle=False)
        self.z = {k: z[k] for k in z.files}
        found = int(self.z["format_version"]) if "format_version" in self.z else 0
        if found != self.FORMAT_VERSION:
            raise ValueError(
                f"models.npz is format v{found}, but this soundlung_model.py expects "
                f"v{self.FORMAT_VERSION}. Re-upload models.npz.")
        self.features = {v: [str(f) for f in self.z[f"{v}_features"]] for v in self.VARIANTS}
        self.val_auc = {v: float(self.z[f"{v}_val_auc"]) for v in self.VARIANTS}
        # per-variant thresholds tuned on validation; class-weighted training
        # inflates probabilities so a fixed 0.5 over-calls "old"
        self.thresholds = {v: float(self.z[f"{v}_threshold"]) for v in self.VARIANTS}
        self.threshold = self.thresholds[self.VERDICT]
        self.spec_mu = float(self.z["I_spec_mu"])
        self.spec_sd = float(self.z["I_spec_sd"])

    def standardise(self, feats, variant):
        x = np.array([feats[f] for f in self.features[variant]], dtype=np.float64)
        return (x - self.z[f"{variant}_scaler_mean"]) / self.z[f"{variant}_scaler_scale"]

    # --- young/old ensemble
    def prob_old_standardised(self, x_std, variant="voice"):
        outs = []
        for i in range(int(self.z[f"{variant}_n_models"])):
            h = x_std @ self.z[f"{variant}_{i}_net_0_weight"].T + \
                self.z[f"{variant}_{i}_net_0_bias"]
            h = np.maximum(h, 0)
            logit = h @ self.z[f"{variant}_{i}_net_3_weight"].T + \
                self.z[f"{variant}_{i}_net_3_bias"]
            outs.append(1.0 / (1.0 + np.exp(-logit)))
        return float(np.mean(outs))

    # --- CNN age regressor
    def cnn_age_from_standardised_spec(self, spec_std):
        x = np.asarray(spec_std, dtype=np.float32)[None]
        for blk in (0, 4, 8):  # conv, batchnorm, relu, maxpool
            x = _conv2d(x, self.z[f"I_features_{blk}_weight"], self.z[f"I_features_{blk}_bias"])
            x = _batchnorm(x, self.z[f"I_features_{blk + 1}_weight"],
                           self.z[f"I_features_{blk + 1}_bias"],
                           self.z[f"I_features_{blk + 1}_running_mean"],
                           self.z[f"I_features_{blk + 1}_running_var"])
            x = _maxpool2(np.maximum(x, 0))
        pooled = x.mean(axis=(1, 2))                       # AdaptiveAvgPool2d(1)
        return float(pooled @ self.z["I_head_3_weight"][0] + self.z["I_head_3_bias"][0])

    # --- full pipeline
    def analyse(self, gender, vowel_bytes, phrase_bytes):
        # vowel: steadiest stretch only (the phrase stays whole - its duration
        # is itself a feature)
        feats = dict(vowel_features(to_sound(vowel_bytes, steady=True), gender))
        v_samples, v_sr = read_audio(vowel_bytes)
        drift = level_drift_db(trim_silence(v_samples, v_sr), v_sr)
        try:
            feats.update(phrase_features(to_sound(phrase_bytes)))
        except Exception:
            try:
                feats.update(phrase_features(to_sound(phrase_bytes, clip_transients=True)))
            except Exception as exc:
                raise ValueError(
                    f"Could not measure the sentence ({exc}). Recorded: "
                    f"{audio_stats(phrase_bytes)}. Speak closer to the microphone, "
                    f"a little louder, and start ~1 s after pressing record.") from None
        feats["Gender_num"] = 1.0 if gender == "M" else 0.0

        probs = {v: self.prob_old_standardised(self.standardise(feats, v), v)
                 for v in self.VARIANTS}
        prob = probs[self.VERDICT]

        xf = self.standardise(feats, "full")
        dist = float(np.sqrt(xf @ self.z["full_inv_cov"] @ xf.T))
        reliable = dist <= float(self.z["full_ood_threshold"])
        odd = []
        if not reliable:
            for i in np.argsort(-np.abs(xf)):
                f = self.features["full"][i]
                if f == "Gender_num" or abs(xf[i]) < 2:
                    continue
                odd.append(f"{f} = {feats[f]:.2f} "
                           f"({abs(xf[i]):.1f} SD {'high' if xf[i] > 0 else 'low'})")

        samples, sr = read_audio(phrase_bytes)
        spec = log_mel(trim_silence(samples, sr), sr)
        age = self.cnn_age_from_standardised_spec((spec - self.spec_mu) / self.spec_sd)

        def label(p, variant):
            return "old (60+)" if p > self.thresholds[variant] else "young (18-30)"

        return {"features": feats, "prob_old": prob, "label": label(prob, self.VERDICT),
                "probs": probs,
                "labels": {v: label(probs[v], v) for v in self.VARIANTS},
                "prob_old_rate": probs["full"], "label_rate": label(probs["full"], "full"),
                "reliable": reliable, "ood_distance": dist, "out_of_range": odd,
                "cnn_age": age, "vowel_drift_db": drift}
