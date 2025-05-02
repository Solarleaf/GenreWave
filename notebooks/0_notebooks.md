# Testing grounds

# classify_new_song.ipynb

1. Setup

   - Import libraries
   - Define paths (e.g. SONG_DIR, MODEL paths)
   - Load models (RF, CNN, CNNv2)

2. Spectrogram Generation

   - Call `generate_spectrograms_from_folder()` from 3.1
   - Save new spectrograms to `spectrograms/<genre>/<file>.png`

3. Inference

   - For each `.mp3` in `data/Songs/<genre>/`:
     - Extract MFCCs → Predict with RF
     - Load spectrogram image:
       - Predict with CNN
       - Predict with CNNv2
     - Store predictions from all models

4. Evaluation + Graphs

   - Classification report (per model)
   - Confusion matrix (per model)
   - Per-genre bar charts: precision, recall, F1-score

5. Output
   - Save all results to:
     - `reports/4_Classify_New_Song/RF/`
     - `.../CNN/`
     - `.../CNNv2/`
