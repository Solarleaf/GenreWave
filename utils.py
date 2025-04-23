# utils.py
# L 4-22-25

import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt


def get_track_path(track_id, base_dir="data/fma_small"):
    tid = f"{int(track_id):06d}"
    return os.path.join(base_dir, tid[:3], f"{tid}.mp3")


def extract_mfcc(audio_path, n_mfcc=20):
    y, sr = librosa.load(audio_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc, axis=1)  # shape: (n_mfcc,)


def save_mel_spectrogram(audio_path, out_path):
    y, sr = librosa.load(audio_path, sr=None)
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_dB = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(4, 4))
    librosa.display.specshow(S_dB, sr=sr, cmap='magma')
    plt.axis('off')
    plt.tight_layout(pad=0)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.close()
