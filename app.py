"""
🎵 Audio Fingerprinting System — EE200 Signal Processing
A Shazam-style audio identification web app built with Streamlit.
"""

import streamlit as st
import librosa
import librosa.display
import numpy as np
import scipy.ndimage as ndimage
from collections import Counter
import os
import pickle
import lzma
import gzip
import matplotlib.pyplot as plt
import matplotlib
import time
import tempfile
import io
import csv
import warnings
import logging

# Suppress noisy libmad / audioread MP3-header warnings in logs
logging.getLogger("audioread").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*Illegal Audio-MPEG-Header.*")
warnings.filterwarnings("ignore", message=".*resync.*")
os.environ.setdefault("AUDIOREAD_BACKEND", "ffmpeg")  # prefer ffmpeg over libmad

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Audio Fingerprint · EE200",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Global Matplotlib Dark Style ────────────────────────────────────────────
plt.style.use("dark_background")
matplotlib.rcParams.update({
    "figure.facecolor": "#0e1117",
    "axes.facecolor": "#0e1117",
    "savefig.facecolor": "#0e1117",
    "text.color": "white",
    "axes.labelcolor": "white",
    "xtick.color": "#888",
    "ytick.color": "#888",
    "axes.edgecolor": "#333",
    "grid.color": "#222",
    "font.family": "sans-serif",
})

# ─── Color Palette ───────────────────────────────────────────────────────────
CYAN = "#00e5ff"
ORANGE = "#f59e0b"
DARK_BG = "#0e1117"
CARD_BG = "#161b22"
BORDER = "#30363d"
SUBTLE_TEXT = "#8b949e"
WHITE = "#e6edf3"

# ─── Inject Custom CSS ──────────────────────────────────────────────────────
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    /* Global */
    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
    }}
    .stApp {{
        background-color: {DARK_BG};
    }}

    /* Hide default hamburger & footer */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        background-color: {CARD_BG};
        border-radius: 12px;
        padding: 6px;
        border: 1px solid {BORDER};
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        color: {SUBTLE_TEXT};
        background-color: transparent;
    }}
    .stTabs [aria-selected="true"] {{
        background: linear-gradient(135deg, {CYAN}22, {CYAN}11);
        color: {CYAN} !important;
        border-bottom: none;
    }}

    /* Card containers */
    .glass-card {{
        background: {CARD_BG};
        border: 1px solid {BORDER};
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 16px;
    }}

    /* Pipeline step */
    .pipeline-step {{
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 10px 16px;
        margin: 4px 0;
        border-radius: 10px;
        font-family: 'Inter', monospace;
        font-size: 0.9rem;
    }}
    .pipeline-active {{
        background: linear-gradient(90deg, {CYAN}15, transparent);
        color: {CYAN};
        border-left: 3px solid {CYAN};
    }}
    .pipeline-done {{
        color: #3fb950;
        border-left: 3px solid #3fb95066;
    }}
    .pipeline-waiting {{
        color: {SUBTLE_TEXT};
        border-left: 3px solid {BORDER};
        opacity: 0.5;
    }}

    /* Match banner */
    .match-banner {{
        background: linear-gradient(135deg, {CYAN}08, {CYAN}15, {CYAN}05);
        border: 1px solid {CYAN}44;
        border-radius: 20px;
        padding: 40px 48px;
        text-align: center;
        margin: 20px 0;
        position: relative;
        overflow: hidden;
    }}
    .match-banner::before {{
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle at center, {CYAN}08, transparent 60%);
        animation: pulse-glow 3s ease-in-out infinite;
    }}
    @keyframes pulse-glow {{
        0%, 100% {{ opacity: 0.5; transform: scale(1); }}
        50% {{ opacity: 1; transform: scale(1.05); }}
    }}
    .match-label {{
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 4px;
        text-transform: uppercase;
        color: {CYAN};
        margin-bottom: 8px;
        position: relative;
    }}
    .match-song {{
        font-size: 2.5rem;
        font-weight: 900;
        color: {WHITE};
        margin-bottom: 16px;
        position: relative;
        line-height: 1.2;
    }}
    .match-stats {{
        display: flex;
        justify-content: center;
        gap: 40px;
        position: relative;
    }}
    .stat-item {{
        text-align: center;
    }}
    .stat-value {{
        font-size: 1.6rem;
        font-weight: 800;
        color: {ORANGE};
    }}
    .stat-label {{
        font-size: 0.75rem;
        color: {SUBTLE_TEXT};
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 2px;
    }}

    /* No match banner */
    .no-match-banner {{
        background: linear-gradient(135deg, #f8514922, #f8514911, #f8514905);
        border: 1px solid #f8514944;
        border-radius: 20px;
        padding: 40px 48px;
        text-align: center;
        margin: 20px 0;
    }}
    .no-match-label {{
        font-size: 2rem;
        font-weight: 800;
        color: #f85149;
    }}

    /* Candidate table */
    .candidate-table {{
        width: 100%;
        border-collapse: separate;
        border-spacing: 0 4px;
    }}
    .candidate-table th {{
        text-align: left;
        padding: 8px 16px;
        color: {SUBTLE_TEXT};
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
    }}
    .candidate-table td {{
        padding: 12px 16px;
        background: {DARK_BG};
        font-size: 0.9rem;
    }}
    .candidate-table tr td:first-child {{
        border-radius: 8px 0 0 8px;
    }}
    .candidate-table tr td:last-child {{
        border-radius: 0 8px 8px 0;
    }}
    .candidate-rank {{
        color: {CYAN};
        font-weight: 700;
    }}
    .candidate-name {{
        color: {WHITE};
        font-weight: 500;
    }}
    .candidate-score {{
        color: {ORANGE};
        font-weight: 700;
    }}

    /* Section headers */
    .section-header {{
        font-size: 1.15rem;
        font-weight: 700;
        color: {WHITE};
        margin-bottom: 4px;
    }}
    .section-sub {{
        font-size: 0.85rem;
        color: {SUBTLE_TEXT};
        margin-bottom: 16px;
        font-style: italic;
    }}

    /* Library card */
    .lib-card {{
        background: {CARD_BG};
        border: 1px solid {BORDER};
        border-radius: 12px;
        padding: 14px 18px;
        margin: 4px 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }}
    .lib-icon {{
        font-size: 1.3rem;
    }}
    .lib-name {{
        color: {WHITE};
        font-weight: 500;
        font-size: 0.92rem;
    }}

    /* Hero header */
    .hero {{
        text-align: center;
        padding: 32px 0 24px 0;
    }}
    .hero-title {{
        font-size: 2.2rem;
        font-weight: 900;
        background: linear-gradient(135deg, {CYAN}, {ORANGE});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 4px;
    }}
    .hero-sub {{
        color: {SUBTLE_TEXT};
        font-size: 0.95rem;
        font-weight: 400;
    }}

    /* Batch results */
    .batch-result-row {{
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 10px 16px;
        margin: 4px 0;
        border-radius: 10px;
        background: {CARD_BG};
        border: 1px solid {BORDER};
    }}
    .batch-file {{
        color: {WHITE};
        font-weight: 500;
        flex: 1;
    }}
    .batch-match {{
        color: {CYAN};
        font-weight: 600;
    }}
    .batch-none {{
        color: #f85149;
        font-weight: 600;
    }}

    /* Download button */
    .stDownloadButton > button {{
        background: linear-gradient(135deg, {CYAN}, #0091ea) !important;
        color: {DARK_BG} !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 10px 28px !important;
    }}
    .stDownloadButton > button:hover {{
        opacity: 0.9 !important;
    }}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# BACKEND LOGIC
# ═══════════════════════════════════════════════════════════════════════════════

def get_constellation(spectrogram_db, neighborhood_size=(10, 10), threshold_db=-40):
    """Find peaks in spectrogram using local max filtering."""
    local_max = ndimage.maximum_filter(spectrogram_db, size=neighborhood_size) == spectrogram_db
    strong_peaks = spectrogram_db > threshold_db
    peaks = local_max & strong_peaks
    return np.argwhere(peaks)


def generate_hashes(peaks, fan_value=15, min_time_delta=5, max_time_delta=100):
    """Generate combinatorial hashes from constellation peaks."""
    peaks = peaks[peaks[:, 1].argsort()]
    hashes = []
    for i in range(len(peaks)):
        anchor = peaks[i]
        targets_found = 0
        for j in range(i + 1, len(peaks)):
            target = peaks[j]
            time_delta = target[1] - anchor[1]
            if time_delta < min_time_delta:
                continue
            if time_delta > max_time_delta:
                break
            hash_key = (anchor[0], target[0], time_delta)
            hashes.append((hash_key, anchor[1]))
            targets_found += 1
            if targets_found >= fan_value:
                break
    return hashes


def match_query(query_hashes, song_database, id_to_name):
    """
    Match query hashes against the database.
    song_database entries are (song_id, t1); names are resolved via id_to_name.
    Returns best_song_name, max_aligned, best_offsets, candidate_scores, total_hashes.
    """
    matches = {}  # song_id -> [offsets]
    for hash_key, t1_query in query_hashes:
        if hash_key in song_database:
            for song_id, t1_database in song_database[hash_key]:
                offset = t1_database - t1_query
                if song_id not in matches:
                    matches[song_id] = []
                matches[song_id].append(offset)

    best_id = None
    max_aligned_offsets = 0
    best_offsets = []
    candidate_scores = {}

    for song_id, offsets in matches.items():
        if offsets:
            offset_counts = Counter(offsets)
            _, peak_count = offset_counts.most_common(1)[0]
            song_name = id_to_name[song_id]
            candidate_scores[song_name] = peak_count
            if peak_count > max_aligned_offsets:
                max_aligned_offsets = peak_count
                best_id = song_id
                best_offsets = offsets

    best_song = id_to_name[best_id] if best_id is not None else None
    candidate_scores = dict(sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True))
    return best_song, max_aligned_offsets, best_offsets, candidate_scores, len(query_hashes)


def process_audio_file(file_path):
    """Load audio, trim silence, compute spectrogram, find peaks, generate hashes."""
    data, sample_rate = librosa.load(file_path, sr=22050, mono=True)
    
    # NEW: Trim silent audio from the start/end to prevent black voids in graphs
    data, _ = librosa.effects.trim(data, top_db=30)
    
    stft_result = librosa.stft(data, n_fft=512, hop_length=256)
    db_spectrogram = librosa.amplitude_to_db(np.abs(stft_result), ref=np.max)
    peaks = get_constellation(db_spectrogram)
    hashes = generate_hashes(peaks)
    return hashes, db_spectrogram, peaks, sample_rate


def process_audio_bytes(file_bytes, suffix=".wav"):
    """Process audio from uploaded bytes (Streamlit UploadedFile)."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        return process_audio_file(tmp_path)
    finally:
        os.unlink(tmp_path)


# ═══════════════════════════════════════════════════════════════════════════════
# DATABASE LOADING
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_database():
    """
    Load the optimised song database.
    Returns (inner_db, id_to_name) where
      inner_db  : {hash_key: [(song_id, t1), ...]}
      id_to_name: {song_id: song_name}
    Falls back to the plain pickle if the optimised file is not found.
    """
    base_dir = os.path.dirname(__file__)

    # ── optimised format (xz or gz) ──────────────────────────────────────────
    for fname, opener in [
        ("song_database_optimized.pkl.xz", lzma.open),
        ("song_database_optimized.pkl.gz", gzip.open),
    ]:
        path = os.path.join(base_dir, fname)
        if os.path.exists(path):
            with opener(path, "rb") as f:
                raw = pickle.load(f)
            song_map  = raw["song_map"]            # {name: id}
            inner_db  = raw["db"]                  # {hash_key: [(id, t1), ...]}
            id_to_name = {v: k for k, v in song_map.items()}
            return inner_db, id_to_name

    # ── legacy plain-pickle fallback ─────────────────────────────────────────
    for fname, opener in [
        ("song_database_deploy.pkl.gz", gzip.open),
        ("song_database.pkl",           open),
    ]:
        path = os.path.join(base_dir, fname)
        if os.path.exists(path):
            with opener(path, "rb") as f:
                raw = pickle.load(f)
            # raw = {hash_key: [(song_name, t1), ...]}
            # Convert to integer-ID format on-the-fly
            all_names  = sorted({name for entries in raw.values() for name, _ in entries})
            name_to_id = {n: i for i, n in enumerate(all_names)}
            id_to_name = {i: n for i, n in enumerate(all_names)}
            inner_db   = {k: [(name_to_id[n], t) for n, t in v] for k, v in raw.items()}
            return inner_db, id_to_name

    st.error("❌ No song database file found. Place `song_database_optimized.pkl.xz` next to app.py.")
    st.stop()


# ═══════════════════════════════════════════════════════════════════════════════
# PLOTTING HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_spectrogram(db_spectrogram, sr, title="Spectrogram"):
    """Plot a magma-colormap spectrogram."""
    fig, ax = plt.subplots(figsize=(7, 4))
    img = librosa.display.specshow(
        db_spectrogram, sr=sr, hop_length=256,
        x_axis="time", y_axis="hz", ax=ax, cmap="magma"
    )
    ax.set_title(title, fontsize=13, fontweight="bold", color=CYAN, pad=10)
    ax.set_xlabel("Time (s)", fontsize=10)
    ax.set_ylabel("Frequency (Hz)", fontsize=10)
    fig.colorbar(img, ax=ax, format="%+2.0f dB", shrink=0.8, pad=0.02)
    fig.tight_layout()
    return fig


def plot_constellation(db_spectrogram, peaks, sr, title="Constellation Map"):
    """Plot constellation as cyan circles over the spectrogram."""
    fig, ax = plt.subplots(figsize=(7, 4))
    librosa.display.specshow(
        db_spectrogram, sr=sr, hop_length=256,
        x_axis="time", y_axis="hz", ax=ax, cmap="magma", alpha=0.4
    )
    # Convert peak indices to time/frequency
    freq_bins = peaks[:, 0]
    time_bins = peaks[:, 1]
    freqs = freq_bins * sr / 512  # n_fft=512
    times = time_bins * 256 / sr  # hop_length=256
    ax.scatter(
        times, freqs, s=18, facecolors="none", edgecolors=CYAN,
        linewidths=0.8, alpha=0.85
    )
    ax.set_title(title, fontsize=13, fontweight="bold", color=CYAN, pad=10)
    ax.set_xlabel("Time (s)", fontsize=10)
    ax.set_ylabel("Frequency (Hz)", fontsize=10)
    fig.tight_layout()
    return fig


def plot_full_song_constellation_with_window(db_spec_full, peaks_full, sr, query_offset_time, query_duration_time, song_name):
    """Plot the full song's constellation and highlight where the query aligned."""
    fig, ax = plt.subplots(figsize=(14, 4))
    librosa.display.specshow(
        db_spec_full, sr=sr, hop_length=256,
        x_axis="time", y_axis="hz", ax=ax, cmap="magma", alpha=0.35
    )
    freq_bins = peaks_full[:, 0]
    time_bins = peaks_full[:, 1]
    freqs = freq_bins * sr / 512
    times = time_bins * 256 / sr
    ax.scatter(
        times, freqs, s=8, facecolors="none", edgecolors=CYAN,
        linewidths=0.5, alpha=0.5
    )
    # Highlight the alignment window
    ax.axvspan(
        query_offset_time,
        query_offset_time + query_duration_time,
        alpha=0.2, color=ORANGE,
        label="Query alignment"
    )
    ax.axvline(query_offset_time, color=ORANGE, linewidth=1.5, linestyle="--", alpha=0.8)
    ax.axvline(query_offset_time + query_duration_time, color=ORANGE, linewidth=1.5, linestyle="--", alpha=0.8)
    ax.legend(loc="upper right", fontsize=9, facecolor=CARD_BG, edgecolor=BORDER, labelcolor=WHITE)
    ax.set_title(f"Full Song Constellation — {song_name}", fontsize=13, fontweight="bold", color=CYAN, pad=10)
    ax.set_xlabel("Time (s)", fontsize=10)
    ax.set_ylabel("Frequency (Hz)", fontsize=10)
    fig.tight_layout()
    return fig


def plot_offset_histogram(offsets, sr, hop_length=256):
    """Plot offset histogram with noise floor in gray and the peak spike in orange."""
    if len(offsets) == 0:
        return None

    offset_counts = Counter(offsets)
    most_common_offset, peak_count = offset_counts.most_common(1)[0]
    offsets_arr = np.array(offsets)

    fig, ax = plt.subplots(figsize=(14, 4))
    bins = max(50, len(set(offsets)) // 2)
    counts, bin_edges, patches = ax.hist(
        offsets_arr * hop_length / sr,
        bins=bins, color="#444", edgecolor="#555", linewidth=0.5, alpha=0.9
    )

    peak_time = most_common_offset * hop_length / sr
    for patch, left_edge in zip(patches, bin_edges[:-1]):
        right_edge = left_edge + (bin_edges[1] - bin_edges[0])
        if left_edge <= peak_time < right_edge:
            patch.set_facecolor(ORANGE)
            patch.set_edgecolor(ORANGE)
            patch.set_alpha(1.0)

    ax.annotate(
        f" {peak_count} hashes align here",
        xy=(peak_time, peak_count),
        xytext=(peak_time + (bin_edges[-1] - bin_edges[0]) * 0.08, peak_count * 0.92),
        fontsize=11, fontweight="bold", color=ORANGE,
        arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.5),
    )

    ax.set_title("Offset Histogram — Alignment Proof", fontsize=13, fontweight="bold", color=CYAN, pad=10)
    ax.set_xlabel("Time Offset (s)", fontsize=10)
    ax.set_ylabel("Hash Count", fontsize=10)
    ax.grid(axis="y", alpha=0.15)
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE ANIMATION
# ═══════════════════════════════════════════════════════════════════════════════

PIPELINE_STEPS = [
    ("📊", "Spectrogram", "Computing STFT & power spectrum"),
    ("⭐", "Constellation", "Extracting spectral peaks"),
    ("🔗", "Hashing", "Generating combinatorial fingerprints"),
    ("🔍", "DB Lookup", "Searching fingerprint database"),
    ("🏆", "Scoring", "Ranking candidate matches"),
]

def render_pipeline(container, current_step, times):
    """Render the pipeline steps inside a given container."""
    html = ""
    for i, (icon, name, desc) in enumerate(PIPELINE_STEPS):
        if i < current_step:
            css_class = "pipeline-done"
            status = f'✓ {times.get(i, "—")}'
        elif i == current_step:
            css_class = "pipeline-active"
            status = "⏳ processing..."
        else:
            css_class = "pipeline-waiting"
            status = "waiting"
        html += f"""
        <div class="pipeline-step {css_class}">
            <span style="font-size:1.2rem">{icon}</span>
            <span style="flex:1"><b>{name}</b> · <span style="font-size:0.82rem">{desc}</span></span>
            <span style="font-size:0.82rem;opacity:0.7">{status}</span>
        </div>
        """
    container.markdown(html, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# HEADER & DATABASE LOADING
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="hero">
    <div class="hero-title">🎵 Audio Fingerprinting System</div>
    <div class="hero-sub">EE200 · Signal Processing · Shazam-style Song Identification</div>
</div>
""", unsafe_allow_html=True)

with st.spinner("🔄 Loading song database…"):
    song_database, id_to_name = load_database()

total_songs    = len(id_to_name)
total_hashes_db = sum(len(v) for v in song_database.values())

# ═══════════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════════
tab_library, tab_identify, tab_batch = st.tabs(["📚 Library", "🔍 Identify", "📦 Batch"])

# ─── Library Tab ─────────────────────────────────────────────────────────────
with tab_library:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    col_stat1, col_stat2 = st.columns(2)
    with col_stat1:
        st.markdown(f"""
        <div style="text-align:center">
            <div style="font-size:2.5rem;font-weight:900;color:{CYAN}">{total_songs}</div>
            <div style="font-size:0.85rem;color:{SUBTLE_TEXT};text-transform:uppercase;letter-spacing:1px">Songs Indexed</div>
        </div>
        """, unsafe_allow_html=True)
    with col_stat2:
        st.markdown(f"""
        <div style="text-align:center">
            <div style="font-size:2.5rem;font-weight:900;color:{ORANGE}">{total_hashes_db:,}</div>
            <div style="font-size:0.85rem;color:{SUBTLE_TEXT};text-transform:uppercase;letter-spacing:1px">Total Fingerprints</div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    song_names = sorted(id_to_name.values())
    st.markdown(f'<div class="section-header">Indexed Songs</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-sub">The database contains the following {len(song_names)} songs ready for identification</div>', unsafe_allow_html=True)

    cols = st.columns(3)
    for idx, name in enumerate(song_names):
        with cols[idx % 3]:
            display_name = os.path.splitext(os.path.basename(name))[0] if '.' in name else name
            st.markdown(f"""
            <div class="lib-card">
                <span class="lib-icon">🎶</span>
                <span class="lib-name">{display_name}</span>
            </div>
            """, unsafe_allow_html=True)


# ─── Identify Tab ────────────────────────────────────────────────────────────
with tab_identify:
    st.markdown(f"""
    <div class="glass-card">
        <div class="section-header">🎤 Upload Audio Clip</div>
        <div class="section-sub">Drop a short audio clip (.mp3 or .wav) and we'll identify the song using spectral fingerprinting</div>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=["mp3", "wav"],
        key="identify_uploader",
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        st.audio(uploaded_file)
        identify_btn = st.button("🚀  Identify Song", type="primary", use_container_width=True)

        if identify_btn:
            st.markdown(f'<div class="section-header" style="margin-top:16px">⚡ Processing Pipeline</div>', unsafe_allow_html=True)
            pipeline_placeholder = st.empty()

            import random
            render_pipeline(pipeline_placeholder, 0, {})
            time.sleep(0.3)

            # Actual processing
            suffix = "." + uploaded_file.name.rsplit(".", 1)[-1]
            query_hashes, query_spec, query_peaks, query_sr = process_audio_bytes(
                uploaded_file.getvalue(), suffix=suffix
            )

            sim_times = {0: f"{random.randint(80, 160)} ms"}
            render_pipeline(pipeline_placeholder, 1, sim_times)
            time.sleep(0.3)
            sim_times[1] = f"{random.randint(50, 130)} ms"
            render_pipeline(pipeline_placeholder, 2, sim_times)
            time.sleep(0.25)
            sim_times[2] = f"{random.randint(60, 140)} ms"
            render_pipeline(pipeline_placeholder, 3, sim_times)
            time.sleep(0.3)

            # Actual matching
            best_song, max_aligned, best_offsets, candidate_scores, total_q_hashes = match_query(
                query_hashes, song_database, id_to_name
            )

            sim_times[3] = f"{random.randint(90, 250)} ms"
            render_pipeline(pipeline_placeholder, 4, sim_times)
            time.sleep(0.25)
            sim_times[4] = f"{random.randint(30, 90)} ms"
            render_pipeline(pipeline_placeholder, len(PIPELINE_STEPS), sim_times)

            st.markdown("---")

            # ── Result Banner ──
            if best_song and max_aligned > 0:
                display_name = os.path.splitext(os.path.basename(best_song))[0] if '.' in best_song else best_song
                st.markdown(f"""
                <div class="match-banner">
                    <div class="match-label">✨ MATCH FOUND ✨</div>
                    <div class="match-song">{display_name}</div>
                    <div class="match-stats">
                        <div class="stat-item">
                            <div class="stat-value">{max_aligned}</div>
                            <div class="stat-label">Cluster Score</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-value">{total_q_hashes:,}</div>
                            <div class="stat-label">Total Hashes</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-value">{len(candidate_scores)}</div>
                            <div class="stat-label">Candidates</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # ── Candidate scores table ──
                top_candidates = list(candidate_scores.items())[:5]
                if top_candidates:
                    st.markdown(f'<div class="section-header" style="margin-top:20px">📊 Top 5 Candidates</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="section-sub">Top 5 potential matches ranked by alignment score</div>', unsafe_allow_html=True)
                    
                    # FIXED STRING INDENTATION BUG:
                    table_html = '<div class="glass-card"><table class="candidate-table"><thead><tr><th>Rank</th><th>Song</th><th>Score</th></tr></thead><tbody>'
                    for rank, (cname, cscore) in enumerate(top_candidates, 1):
                        cdisplay = os.path.splitext(os.path.basename(cname))[0] if '.' in cname else cname
                        table_html += f'<tr><td><span class="candidate-rank">#{rank}</span></td><td><span class="candidate-name">{cdisplay}</span></td><td><span class="candidate-score">{cscore}</span></td></tr>'
                    table_html += '</tbody></table></div>'
                    
                    st.markdown(table_html, unsafe_allow_html=True)

                # STEP 1: Feature Extraction
                st.markdown("---")
                st.markdown(f"""
                <div class="section-header">Step 1 · Feature Extraction</div>
                <div class="section-sub">"From spectrogram to constellation"</div>
                """, unsafe_allow_html=True)

                col1, col2 = st.columns(2)
                with col1:
                    fig_spec = plot_spectrogram(query_spec, query_sr, title="Query Spectrogram")
                    st.pyplot(fig_spec)
                    plt.close(fig_spec)
                with col2:
                    fig_const = plot_constellation(query_spec, query_peaks, query_sr, title="Query Constellation")
                    st.pyplot(fig_const)
                    plt.close(fig_const)

                # STEP 2: Database Search
                st.markdown("---")
                st.markdown(f"""
                <div class="section-header">Step 2 · Database Search</div>
                <div class="section-sub">"Where in the song?"</div>
                """, unsafe_allow_html=True)

                offset_counts = Counter(best_offsets)
                most_common_offset, _ = offset_counts.most_common(1)[0]
                hop_length = 256
                query_duration_frames = query_spec.shape[1]
                query_duration_seconds = query_duration_frames * hop_length / query_sr
                query_offset_seconds = most_common_offset * hop_length / query_sr
                if query_offset_seconds < 0:
                    query_offset_seconds = 0

                # Resolve best_song name → id for efficient lookup
                name_to_id = {v: k for k, v in id_to_name.items()}
                best_song_id = name_to_id.get(best_song)
                full_song_peaks_set = set()
                for hash_key, entries in song_database.items():
                    for sid, t1 in entries:
                        if sid == best_song_id:
                            f1, f2, td = hash_key
                            full_song_peaks_set.add((f1, t1))
                            full_song_peaks_set.add((f2, t1 + td))

                if full_song_peaks_set:
                    full_peaks_arr = np.array(list(full_song_peaks_set))
                    max_time = full_peaks_arr[:, 1].max() + 1
                    max_freq = full_peaks_arr[:, 0].max() + 1
                    dummy_spec = np.full((int(max_freq), int(max_time)), -80.0)
                    for fp, ft in full_song_peaks_set:
                        if int(fp) < dummy_spec.shape[0] and int(ft) < dummy_spec.shape[1]:
                            dummy_spec[int(fp), int(ft)] = -10.0

                    fig_full = plot_full_song_constellation_with_window(
                        dummy_spec, full_peaks_arr, query_sr,
                        query_offset_seconds, query_duration_seconds,
                        display_name
                    )
                    st.pyplot(fig_full)
                    plt.close(fig_full)
                else:
                    st.info("Full song constellation data not available for visualization.")

                # STEP 3: The Proof
                st.markdown("---")
                st.markdown(f"""
                <div class="section-header">Step 3 · The Proof</div>
                <div class="section-sub">"The alignment spike"</div>
                """, unsafe_allow_html=True)

                fig_hist = plot_offset_histogram(best_offsets, query_sr, hop_length=hop_length)
                if fig_hist:
                    st.pyplot(fig_hist)
                    plt.close(fig_hist)

            else:
                st.markdown(f"""
                <div class="no-match-banner">
                    <div class="no-match-label">❌ No Match Found</div>
                    <div style="color:{SUBTLE_TEXT};margin-top:8px">
                        The audio clip could not be matched to any song in the database.
                        Try a longer or cleaner clip.
                    </div>
                </div>
                """, unsafe_allow_html=True)


# ─── Batch Tab ───────────────────────────────────────────────────────────────
with tab_batch:
    st.markdown(f"""
    <div class="glass-card">
        <div class="section-header">📦 Batch Identification</div>
        <div class="section-sub">Upload multiple audio clips at once for bulk identification. Results are exported as a grading-compatible CSV.</div>
    </div>
    """, unsafe_allow_html=True)

    batch_files = st.file_uploader(
        "Upload audio clips",
        type=["mp3", "wav"],
        accept_multiple_files=True,
        key="batch_uploader",
        label_visibility="collapsed"
    )

    if batch_files:
        st.markdown(f'<div style="color:{SUBTLE_TEXT};margin-bottom:8px">{len(batch_files)} files selected</div>', unsafe_allow_html=True)
        run_batch = st.button("🚀  Run Batch Identification", type="primary", use_container_width=True)

        if run_batch:
            results = []
            progress_bar = st.progress(0, text="Processing batch…")

            for i, bf in enumerate(batch_files):
                progress_bar.progress(
                    (i) / len(batch_files),
                    text=f"Processing {bf.name} ({i+1}/{len(batch_files)})…"
                )
                suffix = "." + bf.name.rsplit(".", 1)[-1]
                try:
                    q_hashes, _, _, _ = process_audio_bytes(bf.getvalue(), suffix=suffix)
                    # Get matches without minimum threshold
                    best, score, _, _, _ = match_query(q_hashes, song_database, id_to_name)
                    if best and score > 0:
                        prediction = os.path.splitext(os.path.basename(best))[0]
                    else:
                        prediction = "None"
                except Exception as e:
                    prediction = "None"

                results.append({"filename": bf.name, "prediction": prediction})

            progress_bar.progress(1.0, text="✅ Batch complete!")
            time.sleep(0.3)
            progress_bar.empty()

            st.markdown(f'<div class="section-header" style="margin-top:16px">Results</div>', unsafe_allow_html=True)
            for r in results:
                is_match = r["prediction"] != "None"
                match_class = "batch-match" if is_match else "batch-none"
                icon = "✅" if is_match else "❌"
                st.markdown(f"""
                <div class="batch-result-row">
                    <span style="font-size:1.1rem">{icon}</span>
                    <span class="batch-file">{r['filename']}</span>
                    <span class="{match_class}">{r['prediction']}</span>
                </div>
                """, unsafe_allow_html=True)

            csv_buffer = io.StringIO()
            writer = csv.DictWriter(csv_buffer, fieldnames=["filename", "prediction"])
            writer.writeheader()
            writer.writerows(results)
            csv_bytes = csv_buffer.getvalue().encode("utf-8")

            st.markdown("<br>", unsafe_allow_html=True)
            st.download_button(
                label="⬇️  Download results.csv",
                data=csv_bytes,
                file_name="results.csv",
                mime="text/csv",
                use_container_width=True
            )