# SoundLung — cloud demo

Shareable version of the voice age-screening demo, for Streamlit Community
Cloud (free, HTTPS, works on phones — so a supervisor can open a link and try
it without any local setup).

## Deploying — all in the browser, no command line (about 5 minutes)

### Step 1 · Create the GitHub repository

1. Sign in at [github.com](https://github.com) and click **+** (top right) →
   **New repository**.
2. Name it `soundlung-demo`, keep it **Public**, and **do not** tick
   "Add a README file" (an empty repo makes the next step simpler).
3. Click **Create repository**.

### Step 2 · Upload the files by drag-and-drop

1. On the new empty repo page, click the **uploading an existing file** link
   (under "…or upload an existing file").
2. Open this `cloud` folder in Explorer, select **all 14 files**
   (`Ctrl+A`), and drag them onto the GitHub page.
   **Upload the files themselves, not the folder** — `streamlit_app.py` must
   end up at the top level of the repo.
3. Scroll down, click **Commit changes**.

You should now see `streamlit_app.py`, `soundlung_model.py`, `models.npz`,
`requirements.txt`, `packages.txt` and the `.wav` files listed in the repo.

### Step 3 · Deploy on Streamlit

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with
   **GitHub** (authorise it when asked).
2. Click **Create app** → **Deploy a public app from GitHub**.
3. Fill in:
   - **Repository**: `<your-username>/soundlung-demo`
   - **Branch**: `main`
   - **Main file path**: `streamlit_app.py`
4. Click **Deploy**. The first build takes 2–5 minutes (it installs Praat and
   friends). Watch the log; it is done when the app appears.

### Step 4 · Share

Copy the `https://<name>.streamlit.app` link and send it to your supervisor.
It works on phones, microphone included (Streamlit serves HTTPS, which browsers
require for recording).

> If the app sleeps after a period of inactivity, the next visitor simply clicks
> **Yes, get this app back up** and it restarts in under a minute.

Streamlit reads `requirements.txt` (Python packages) and `packages.txt`
(`libsndfile1`, needed by soundfile) automatically — nothing else to configure.

### Updating it later

Open the file on GitHub → pencil icon → edit → **Commit changes**, or use
**Add file ▸ Upload files** to replace `models.npz` after retraining. Streamlit
redeploys automatically within a minute.

## Why no PyTorch

`torch` on Linux pulls in ~2 GB of CUDA packages, which does not fit the free
tier. These models are a few hundred KB of weights, so `export_models.py`
(project root) exports them to `models.npz` and `soundlung_model.py` runs the
inference in NumPy — the MLP ensemble and the CNN's conv/batch-norm/pool stack.

Parity with the PyTorch originals is asserted during export (max abs difference
2.7e-08 for the classifier, 0.0 for the CNN) and re-checked end-to-end against
the local app: identical features, p(old) and age estimates.

## What ships here

| File | Purpose |
|---|---|
| `streamlit_app.py` | the app |
| `soundlung_model.py` | NumPy inference + Praat feature extraction |
| `models.npz` | exported weights (110 KB) |
| `requirements.txt`, `packages.txt` | dependencies |
| `ref_*.wav` | the German sentence and vowel to copy (SVD, CC-BY 4.0) |
| `example_*.wav` | known-age examples (ages 21 and 76) for demoing |

## Regenerating the weights

After retraining, from the project root:

```bash
python train_deploy_model.py   # refits the ensemble + OOD guard
python export_models.py        # rewrites cloud/models.npz, asserts parity
```

## Privacy note

Recordings are analysed in memory and never written to disk, but they *are*
uploaded to a third-party host to be processed. Before using this with study
participants rather than colleagues, check it against your ethics approval —
self-hosting `app.py` (project root) keeps audio on your own machine.
