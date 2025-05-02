import os
import torchaudio
import numpy as np
import pandas as pd
from tqdm import tqdm
from torchvggish import vggish, vggish_input
import torch

# Constants
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_MP3_ROOT = os.path.join(PROJECT_ROOT, "data/fma_small")
DEFAULT_OUTPUT_DIR = os.path.join(
    PROJECT_ROOT, "reports/step_4_1_vggish_embeddings")
DEFAULT_METADATA_PATH = os.path.join(PROJECT_ROOT, "data", "metadata.csv")


def generate_vggish_embeddings(mp3_root=DEFAULT_MP3_ROOT, output_dir=DEFAULT_OUTPUT_DIR, metadata_path=DEFAULT_METADATA_PATH):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)

    model = vggish()
    model.eval()

    metadata_records = []
    success_count = 0
    error_count = 0

    for genre_folder in sorted(os.listdir(mp3_root)):
        genre_path = os.path.join(mp3_root, genre_folder)
        if not os.path.isdir(genre_path):
            continue

        for file in tqdm(os.listdir(genre_path), desc=f"Processing {genre_folder}"):
            if not file.endswith(".mp3"):
                continue

            track_id = os.path.splitext(file)[0]
            mp3_path = os.path.join(genre_path, file)
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
                metadata_records.append(
                    {"track_id": int(track_id), "genre": genre_folder})
                success_count += 1

            except Exception as e:
                print(f"[ERROR] {track_id}: {e}")
                error_count += 1

    # Load existing metadata if present
    if os.path.exists(metadata_path):
        existing_df = pd.read_csv(metadata_path)
        metadata_df = pd.DataFrame(metadata_records)
        combined_df = pd.concat([existing_df, metadata_df], ignore_index=True)
        combined_df.drop_duplicates(subset="track_id", inplace=True)
    else:
        combined_df = pd.DataFrame(metadata_records)

    combined_df.to_csv(metadata_path, index=False)
    print(f"\nSuccessfully processed: {success_count}")
    print(f"Failed to process: {error_count}")
    print(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    generate_vggish_embeddings()
