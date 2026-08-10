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


def to_sound(data, clip_transients=False):
    samples, sr = read_audio(data)
    if clip_transients:
        ref = np.percentile(np.abs(samples), 99.0)
        if ref > 0:
            samples = np.clip(samples, -ref, ref)
    samples = trim_silence(samples, sr)
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
    def __init__(self, npz_path):
        z = np.load(npz_path, allow_pickle=False)
        self.z = {k: z[k] for k in z.files}
        self.features = [str(f) for f in self.z["G_features"]]
        self.n_models = int(self.z["G_n_models"])
        self.scaler_mean = self.z["G_scaler_mean"]
        self.scaler_scale = self.z["G_scaler_scale"]
        self.inv_cov = self.z["G_inv_cov"]
        self.ood_threshold = float(self.z["G_ood_threshold"])
        self.threshold = float(self.z["G_threshold"])
        self.val_auc = float(self.z["G_val_auc"])
        self.spec_mu = float(self.z["I_spec_mu"])
        self.spec_sd = float(self.z["I_spec_sd"])

    # --- young/old ensemble
    def prob_old_standardised(self, x_std):
        outs = []
        for i in range(self.n_models):
            h = x_std @ self.z[f"G{i}_net_0_weight"].T + self.z[f"G{i}_net_0_bias"]
            h = np.maximum(h, 0)
            logit = h @ self.z[f"G{i}_net_3_weight"].T + self.z[f"G{i}_net_3_bias"]
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
        feats = dict(vowel_features(to_sound(vowel_bytes), gender))
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

        x = np.array([feats[f] for f in self.features], dtype=np.float64)
        x_std = (x - self.scaler_mean) / self.scaler_scale
        prob = self.prob_old_standardised(x_std)

        dist = float(np.sqrt(x_std @ self.inv_cov @ x_std.T))
        reliable = dist <= self.ood_threshold
        odd = []
        if not reliable:
            for i in np.argsort(-np.abs(x_std)):
                f = self.features[i]
                if f == "Gender_num" or abs(x_std[i]) < 2:
                    continue
                odd.append(f"{f} = {feats[f]:.2f} "
                           f"({abs(x_std[i]):.1f} SD {'high' if x_std[i] > 0 else 'low'})")

        samples, sr = read_audio(phrase_bytes)
        spec = log_mel(trim_silence(samples, sr), sr)
        age = self.cnn_age_from_standardised_spec((spec - self.spec_mu) / self.spec_sd)

        return {"features": feats, "prob_old": prob,
                "label": "old (60+)" if prob > self.threshold else "young (18-30)",
                "reliable": reliable, "ood_distance": dist, "out_of_range": odd,
                "cnn_age": age}
