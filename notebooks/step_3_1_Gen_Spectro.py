# L 4-23-25
# notebooks/step_3_1_Gen_Spectro.py

import os
import gc
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib

matplotlib.use("Agg")  # headless image rendering

# Default constants
IMG_SIZE = 128
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV_PATH = os.path.join(
    BASE_DIR, "../reports/2_MFCC_RF_Classifier/used_tracks.csv")
DEFAULT_AUDIO_DIR = os.path.join(BASE_DIR, "../data/fma_small")
DEFAULT_OUTPUT_DIR = os.path.join(BASE_DIR, "../spectrograms")


def generate_spectrogram(input_path, output_path, img_size=IMG_SIZE):
    """Generate and save mel-spectrogram for a given audio file."""
    try:
        y, sr = librosa.load(input_path, sr=None, duration=30)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S_dB = librosa.power_to_db(S, ref=np.max)

        fig, ax = plt.subplots(figsize=(1.28, 1.28), dpi=100)
        librosa.display.specshow(S_dB, sr=sr, cmap='viridis', ax=ax)
        ax.axis("off")
        plt.tight_layout(pad=0)
        fig.savefig(output_path, bbox_inches="tight", pad_inches=0)
        plt.close(fig)

        # Resize to square RGB
        with Image.open(output_path) as img:
            img = img.convert("RGB")
            img = img.resize((img_size, img_size))
            img.save(output_path)

        return True
    except Exception as e:
        print(f"[ERROR] Spectrogram failed for {input_path}: {e}")
        gc.collect()
        return False


def generate_spectrograms_from_csv(csv_path, audio_dir, output_dir, img_size=IMG_SIZE):
    """Generate spectrograms from track CSV metadata and audio folder."""
    if not os.path.isfile(csv_path):
        print(f"[ERROR] CSV not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    print(f"[INFO] Generating spectrograms for {len(df)} tracks from CSV")

    for _, row in df.iterrows():
        track_id = str(row["track_id"]).zfill(6)
        genre = row["genre"]
        subdir = track_id[:3]
        audio_path = os.path.join(audio_dir, subdir, f"{track_id}.mp3")
        output_genre_path = os.path.join(output_dir, genre)
        os.makedirs(output_genre_path, exist_ok=True)
        output_path = os.path.join(output_genre_path, f"{track_id}.png")

        if not os.path.exists(output_path):
            generate_spectrogram(audio_path, output_path, img_size)


def generate_spectrograms_from_folder(song_dir, output_dir, img_size=IMG_SIZE):
    """Generate spectrograms from a folder structured by genre with MP3 files."""
    print(f"[INFO] Generating spectrograms from folder: {song_dir}")

    for genre in os.listdir(song_dir):
        genre_dir = os.path.join(song_dir, genre)
        if not os.path.isdir(genre_dir):
            continue

        output_genre_path = os.path.join(output_dir, genre)
        os.makedirs(output_genre_path, exist_ok=True)

        for fname in os.listdir(genre_dir):
            if not fname.endswith(".mp3"):
                continue
            audio_path = os.path.join(genre_dir, fname)
            output_path = os.path.join(
                output_genre_path, f"{os.path.splitext(fname)[0]}.png")

            if not os.path.exists(output_path):
                generate_spectrogram(audio_path, output_path, img_size)


if __name__ == "__main__":
    if os.path.isfile(DEFAULT_CSV_PATH):
        generate_spectrograms_from_csv(
            DEFAULT_CSV_PATH, DEFAULT_AUDIO_DIR, DEFAULT_OUTPUT_DIR, IMG_SIZE)
    else:
        print(
            f"[WARNING] Default CSV not found: {DEFAULT_CSV_PATH}. No spectrograms generated.")
