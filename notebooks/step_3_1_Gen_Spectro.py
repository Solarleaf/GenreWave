# step_3_1_Gen_Spectro.py
# L 4-24-25

import os
import gc
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib

# Use non-interactive backend for image rendering
matplotlib.use("Agg")

# Default constants
IMG_SIZE = 128
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_CSV_PATH = os.path.join(
    PROJECT_ROOT, "reports/step_2_MFCC_RF/used_tracks.csv")
DEFAULT_AUDIO_DIR = os.path.join(PROJECT_ROOT, "data/fma_small")
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "spectrograms")


def generate_spectrogram(input_path, output_path, img_size=IMG_SIZE):
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
    print(f"[INFO] Verifying spectrograms for {len(df)} tracks")

    for _, row in df.iterrows():
        track_id = str(row["track_id"]).zfill(6)
        genre = row["genre"]
        subdir = track_id[:3]
        audio_path = os.path.join(audio_dir, subdir, f"{track_id}.mp3")
        output_genre_path = os.path.join(output_dir, genre)
        os.makedirs(output_genre_path, exist_ok=True)
        output_path = os.path.join(output_genre_path, f"{track_id}.png")

        if not os.path.isfile(output_path):
            generate_spectrogram(audio_path, output_path, img_size)


def generate_spectrograms_from_folder(song_dir, output_dir, img_size=IMG_SIZE):
    """
    Generate spectrograms from a folder structure:
    song_dir/
        genre1/
            song1.mp3
            song2.mp3
        genre2/
            ...
    """
    print(f"[INFO] Generating spectrograms from: {song_dir}")

    for genre_name in os.listdir(song_dir):
        genre_path = os.path.join(song_dir, genre_name)
        if not os.path.isdir(genre_path):
            continue

        output_genre_path = os.path.join(output_dir, genre_name)
        os.makedirs(output_genre_path, exist_ok=True)

        for file_name in os.listdir(genre_path):
            if not file_name.lower().endswith(".mp3"):
                continue

            base_name = os.path.splitext(file_name)[0]
            input_path = os.path.join(genre_path, file_name)
            output_path = os.path.join(output_genre_path, f"{base_name}.png")

            if not os.path.exists(output_path):
                generate_spectrogram(input_path, output_path, img_size)


def generate_all(csv_path=DEFAULT_CSV_PATH, audio_dir=DEFAULT_AUDIO_DIR, output_dir=DEFAULT_OUTPUT_DIR, img_size=IMG_SIZE):
    """Entry function to generate all spectrograms with optional custom paths."""
    if os.path.isfile(csv_path):
        generate_spectrograms_from_csv(
            csv_path, audio_dir, output_dir, img_size)
    else:
        print(
            f"[WARNING] CSV not found: {csv_path}. No spectrograms generated.")


if __name__ == "__main__":
    generate_all()  # Uses default paths
