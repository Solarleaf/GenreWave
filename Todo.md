## ✅ Completed Tasks

- [x] Loaded and explored metadata (`tracks.csv`, `genres.csv`)
- [x] Created `overall_tracks.csv` (all usable top-level + subgenre metadata)
- [x] Created `valid_track_genres.csv` (subset with corresponding `.mp3` files)
- [x] Visualized genre and subgenre distributions (both datasets)
- [x] Extracted MFCCs for valid tracks (`mfcc_features.npz`)
- [x] Trained and evaluated Random Forest classifier on MFCCs
- [x] Generated spectrogram images from `used_tracks.csv`
- [x] Trained CNN (SimpleCNN) on spectrograms
- [x] Evaluated CNN: classification report + confusion matrix
- [x] Generated per-genre metrics (precision, recall, F1, support) for CNN and RF
- [x] Cleaned project file structure and updated all relevant scripts

## 🛠️ Backlog

- [ ] Evaluate CNN with:
  - More epochs
  - Alternative learning rates
  - Weighted or random sampling for class imbalance
- [ ] Plot training loss curves during CNN training
- [ ] Add chroma-based features (e.g. `chroma_stft`) for MFCC or CNN input
- [ ] Try additional scikit-learn models (e.g., SVM, Gradient Boosting)
- [ ] Use `tqdm` for progress feedback during audio preprocessing
- [ ] Log per-run metadata (duration, memory, failure stats)

## ❌ Planned Features / Future Steps

- [ ] Add validation split + early stopping to CNN
- [ ] Replace SimpleCNN with pretrained models (ResNet18, VGG16, etc.)
- [ ] Save full inference-ready CNN model (with transforms + class mapping)
- [ ] Enhance `classify_new_song.py` to:
  - Accept either MFCC or spectrogram input
  - Route to RF or CNN models automatically
- [ ] Create `docs/` folder for:
  - API usage examples (e.g. FastAPI)
  - Preprocessing pipeline documentation
  - Model explanations and usage demos
