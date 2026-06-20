"""
🎵 Audio Fingerprinting System — EE200 Signal Processing
A Shazam-style audio identification web app built with Streamlit.
"""

import gradio as gr
import pandas as pd
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
import subprocess

# Suppress noisy libmad / audioread MP3-header warnings in logs
logging.getLogger("audioread").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*Illegal Audio-MPEG-Header.*")
warnings.filterwarnings("ignore", message=".*resync.*")
os.environ.setdefault("AUDIOREAD_BACKEND", "ffmpeg")  # prefer ffmpeg over libmad
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
    """
    Process audio from uploaded bytes.
    MP3 files are first converted to WAV via ffmpeg to avoid
    libmad C-level stderr noise (Illegal Audio-MPEG-Header warnings).
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    wav_path = None
    try:
        if suffix.lower() == ".mp3":
            # Convert MP3 → WAV with ffmpeg so librosa uses soundfile (no libmad)
            wav_path = tmp_path[:-4] + "_converted.wav"
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-i", tmp_path,
                    "-ar", "22050",
                    "-ac", "1",
                    "-loglevel", "error",
                    wav_path,
                ],
                check=True,
            )
            return process_audio_file(wav_path)
        else:
            return process_audio_file(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        if wav_path and os.path.exists(wav_path):
            os.unlink(wav_path)


# ═══════════════════════════════════════════════════════════════════════════════
# DATABASE LOADING
# ═══════════════════════════════════════════════════════════════════════════════

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

    raise Exception("❌ No song database file found. Place `song_database_optimized.pkl.xz` next to app.py.")


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


def plot_full_song_constellation_with_window(peaks_full, sr, query_offset_time, query_duration_time, song_name, max_time_frames):
    """Plot the full song's constellation and highlight where the query aligned, without using memory-heavy arrays."""
    fig, ax = plt.subplots(figsize=(14, 4))
    
    # Simulate the dark background of a spectrogram without creating a 50MB array
    ax.set_facecolor("#111116")
    
    # Calculate limits
    max_time_sec = max_time_frames * 256 / sr
    ax.set_xlim(0, max_time_sec)
    ax.set_ylim(0, sr / 2)
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

# ═══════════════════════════════════════════════════════════════════════════════
# GRADIO UI
# ═══════════════════════════════════════════════════════════════════════════════

print("Loading database...")
try:
    song_database, id_to_name = load_database()
    total_songs = len(id_to_name)
    total_hashes_db = sum(len(v) for v in song_database.values())
    song_names = sorted(id_to_name.values())
    print(f"Loaded {total_songs} songs with {total_hashes_db} hashes.")
except Exception as e:
    print(e)
    song_database, id_to_name = {}, {}
    total_songs, total_hashes_db, song_names = 0, 0, []

def identify_audio(audio_path):
    if not audio_path:
        return "Please upload an audio file.", None, None, None, None, None
        
    try:
        if audio_path.lower().endswith(".mp3"):
            wav_path = audio_path[:-4] + "_converted.wav"
            import subprocess
            subprocess.run(["ffmpeg", "-y", "-i", audio_path, "-ar", "22050", "-ac", "1", "-loglevel", "error", wav_path], check=True)
            process_path = wav_path
        else:
            process_path = audio_path
            
        query_hashes, query_spec, query_peaks, query_sr = process_audio_file(process_path)
        best_song, max_aligned, best_offsets, candidate_scores, total_q_hashes = match_query(query_hashes, song_database, id_to_name)
        
        if best_song and max_aligned > 0:
            display_name = os.path.splitext(os.path.basename(best_song))[0] if '.' in best_song else best_song
            
            top_candidates = list(candidate_scores.items())[:5]
            df_candidates = pd.DataFrame(top_candidates, columns=["Song", "Score"])
            df_candidates["Rank"] = range(1, len(df_candidates) + 1)
            df_candidates = df_candidates[["Rank", "Song", "Score"]]
            
            fig_spec = plot_spectrogram(query_spec, query_sr, title="Query Spectrogram")
            fig_const = plot_constellation(query_spec, query_peaks, query_sr, title="Query Constellation")
            
            from collections import Counter
            offset_counts = Counter(best_offsets)
            most_common_offset, _ = offset_counts.most_common(1)[0]
            hop_length = 256
            query_duration_frames = query_spec.shape[1]
            query_duration_seconds = query_duration_frames * hop_length / query_sr
            query_offset_seconds = most_common_offset * hop_length / query_sr
            if query_offset_seconds < 0:
                query_offset_seconds = 0
                
            offset_min = int(query_offset_seconds // 60)
            offset_sec = query_offset_seconds % 60
            end_sec = query_offset_seconds + query_duration_seconds
            end_min = int(end_sec // 60)
            end_sec_r = end_sec % 60
            
            match_text = f"### ✨ MATCH FOUND: {display_name} ✨\n"
            match_text += f"**Cluster Score:** {max_aligned} | **Total Hashes:** {total_q_hashes:,} | **Candidates:** {len(candidate_scores)}\n"
            match_text += f"\n**📍 Query Aligned At:** {offset_min:02d}:{offset_sec:05.2f} → {end_min:02d}:{end_sec_r:05.2f}"
            
            name_to_id = {v: k for k, v in id_to_name.items()}
            best_song_id = name_to_id.get(best_song)
            full_song_peaks_set = set()
            for hash_key, entries in song_database.items():
                for sid, t1 in entries:
                    if sid == best_song_id:
                        f1, f2, td = hash_key
                        full_song_peaks_set.add((f1, t1))
                        full_song_peaks_set.add((f2, t1 + td))
                        
            fig_full = None
            if full_song_peaks_set:
                import numpy as np
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
                
            fig_hist = plot_offset_histogram(best_offsets, query_sr, hop_length=hop_length)
            
            return match_text, df_candidates, fig_spec, fig_const, fig_full, fig_hist
            
        else:
            return "❌ No Match Found.", None, None, None, None, None
            
    except Exception as e:
        import traceback
        return f"Error: {str(e)}\n{traceback.format_exc()}", None, None, None, None, None

def batch_process(files):
    if not files:
        return pd.DataFrame()
        
    results = []
    for f in files:
        try:
            audio_path = f.name
            if audio_path.lower().endswith(".mp3"):
                wav_path = audio_path[:-4] + "_converted.wav"
                import subprocess
                subprocess.run(["ffmpeg", "-y", "-i", audio_path, "-ar", "22050", "-ac", "1", "-loglevel", "error", wav_path], check=True)
                process_path = wav_path
            else:
                process_path = audio_path
                
            q_hashes, _, _, _ = process_audio_file(process_path)
            best, score, _, _, _ = match_query(q_hashes, song_database, id_to_name)
            
            if best and score > 0:
                prediction = os.path.splitext(os.path.basename(best))[0]
                status = "✅"
            else:
                prediction = "None"
                status = "❌"
        except Exception as e:
            prediction = f"Error: {str(e)}"
            status = "❌"
            
        results.append({"Status": status, "File": os.path.basename(f.name), "Prediction": prediction})
        
    return pd.DataFrame(results)

css = """
.gradio-container { font-family: 'Inter', sans-serif; }
"""

with gr.Blocks(title="Audio Fingerprint | EE200") as demo:
    gr.Markdown("# 🎵 Audio Fingerprinting System\n*EE200 · Signal Processing · Shazam-style Song Identification*")
    
    with gr.Tabs():
        with gr.Tab("📚 Library"):
            with gr.Row():
                gr.Number(value=total_songs, label="Songs Indexed", interactive=False)
                gr.Number(value=total_hashes_db, label="Total Fingerprints", interactive=False)
            
            df_songs = pd.DataFrame({"Indexed Songs": [os.path.splitext(os.path.basename(n))[0] if '.' in n else n for n in song_names]})
            gr.Dataframe(value=df_songs, interactive=False)
            
        with gr.Tab("🔍 Identify"):
            gr.Markdown("Drop a short audio clip (.mp3 or .wav) and we'll identify the song using spectral fingerprinting.")
            with gr.Row():
                audio_input = gr.Audio(type="filepath", label="Upload Audio Clip")
            
            identify_btn = gr.Button("🚀 Identify Song", variant="primary")
            match_output = gr.Markdown()
            candidates_output = gr.Dataframe(label="Top 5 Candidates")
            
            with gr.Row():
                plot_spec = gr.Plot(label="Query Spectrogram")
                plot_const = gr.Plot(label="Query Constellation")
                
            plot_full = gr.Plot(label="Full Song Constellation (Alignment Window)")
            plot_hist = gr.Plot(label="Alignment Spike (Offset Histogram)")
            
            identify_btn.click(
                fn=identify_audio,
                inputs=audio_input,
                outputs=[match_output, candidates_output, plot_spec, plot_const, plot_full, plot_hist]
            )
            
        with gr.Tab("📦 Batch"):
            gr.Markdown("Upload multiple clips to identify them all at once.")
            batch_files = gr.File(file_count="multiple", label="Upload Audio Files")
            batch_btn = gr.Button("Process Batch", variant="primary")
            batch_output = gr.Dataframe(label="Batch Results")
            
            batch_btn.click(fn=batch_process, inputs=batch_files, outputs=batch_output)

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Monochrome(), css=css)
