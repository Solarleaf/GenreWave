# Project Log – Music Genre Classification

## ✅ What Was Done

### April 30, 2025

- Updated `classify_new_song.ipynb` to:

  - Only use **Random Forest** and **spectrograms**
  - Automatically generate spectrograms via `%run ./3.1_Generate_Spectrograms.ipynb`
  - Use RF model trained in `2_MFCC_RF_Classifier.ipynb`
  - Save new spectrograms to `reports/4_Classify_New_Song/spectrograms/<genre>/<file>.png`
  - Append spectrogram paths to results in `rf_model_predictions.csv`

- `3.1_Generate_Spectrograms.ipynb`:

  - Modularized with dual behavior:
    - If run directly: processes `fma_small` + `used_tracks.csv`
    - If run via `%run` (e.g., from `classify_new_song`): generates spectrograms from `data/Songs/` to `reports/4_Classify_New_Song/spectrograms/`

- Confirmed that `classify_new_song` never directly imports or depends on `2_MFCC_RF_Classifier.ipynb`; it only uses the trained RF model file.

## 🔜 Next Steps

- Re-integrate **CNN** and **CNNv2** into `classify_new_song.ipynb`
- Enable visual inline plots of spectrograms and prediction confidence
- Consider extending the output CSV with confidence scores (from CNN)

## ⚠️ Known Issues / Blockers

- `%run` execution works but does not share variables across notebook cells
- `3.1_Generate_Spectrograms.ipynb` must remain a notebook (not a .py script) to be compatible with `%run`

## 📝 TODOs (as prompts for LLMs or teammates)

- Add CNN/CNNv2 support back into `classify_new_song.ipynb`
  - Load each inference bundle
  - Generate CNN predictions from spectrograms in `reports/4_Classify_New_Song/spectrograms/`
  - Save results to CSV
- Visualize spectrograms + prediction confidence inline in `classify_new_song.ipynb`
- Update `rf_model_predictions.csv` to include top-N RF confidence scores (optional)
- Confirm that all paths in notebooks resolve when running from root via `jupyter lab`
- Modularize confusion matrix + metric plotting into reusable function or cell

## 💡 Design Assumptions

- Spectrograms are always saved to a consistent directory based on calling context
- RF, CNN, CNNv2 all take genre-labeled directories under `data/Songs/` for inference
- Jupyter notebooks are the primary interface; scripts are avoided
