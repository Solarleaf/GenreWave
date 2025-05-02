# notebooks/step_4_1_Gen_vggish_embed.py

import os
import torchaudio
import numpy as np
from tqdm import tqdm
from torchvggish import vggish, vggish_input
import torch

#  Constants
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_MP3_ROOT = os.path.join(PROJECT_ROOT, "data/fma_small")
DEFAULT_OUTPUT_DIR = os.path.join(
    PROJECT_ROOT, "reports/step_4_1/vggish_embeddings")


def generate_vggish_embeddings(mp3_root=DEFAULT_MP3_ROOT, output_dir=DEFAULT_OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)

    model = vggish()
    model.eval()

    success_count = 0
    error_count = 0

    for folder in sorted(os.listdir(mp3_root)):
        folder_path = os.path.join(mp3_root, folder)
        if not os.path.isdir(folder_path):
            continue

        for file in tqdm(os.listdir(folder_path), desc=f"Processing {folder}"):
            if not file.endswith(".mp3"):
                continue

            track_id = file.split(".")[0]
            mp3_path = os.path.join(folder_path, file)
            out_path = os.path.join(output_dir, f"{track_id}.npy")

            if os.path.exists(out_path):
                continue

            try:
                waveform, sr = torchaudio.load(mp3_path)
                if sr != 16000:
                    resampler = torchaudio.transforms.Resample(
                        orig_freq=sr, new_freq=16000)
                    waveform = resampler(waveform)

                mel = vggish_input.waveform_to_examples(
                    waveform.numpy()[0], 16000)

                with torch.no_grad():
                    features = model(torch.tensor(mel).float())
                    song_embedding = features.mean(dim=0).numpy()

                np.save(out_path, song_embedding)
                success_count += 1

            except Exception as e:
                print(f"[ERROR] {track_id}: {e}")
                error_count += 1

    print(f"\nSuccessfully processed: {success_count}")
    print(f"Failed to process: {error_count}")


if __name__ == "__main__":
    generate_vggish_embeddings()
