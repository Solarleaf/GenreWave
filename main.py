from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm
import numpy as np
import librosa
import pandas as pd
import os

DATA_DIR = 'fma_small'
METADATA_DIR = 'fma_metadata'

# CSV's
tracks = pd.read_csv(os.path.join(
    METADATA_DIR, 'tracks.csv'), index_col=0, header=[0, 1])
tracks.head()


# Extract MFCC

AUDIO_DIR = DATA_DIR
TRACK_IDS = tracks['set', 'subset'] == 'small'
SELECTED_TRACKS = tracks[TRACK_IDS]


def extract_features(track_id):
    file_path = f'{AUDIO_DIR}/{str(track_id).zfill(6)[:3]}/{str(track_id).zfill(6)}.mp3'
    try:
        y, sr = librosa.load(file_path, duration=30)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        return np.mean(mfcc.T, axis=0)
    except Exception as e:
        print(f'Error loading {file_path}: {e}')
        return None


features = []
labels = []

print("Extracting features...")
for idx, row in tqdm(SELECTED_TRACKS.iterrows(), total=SELECTED_TRACKS.shape[0]):
    genre = row['track', 'genre_top']
    if pd.isna(genre):
        continue  # Skip if no genre
    feat = extract_features(idx)
    if feat is not None:
        features.append(feat)
        labels.append(genre)

X = np.array(features)
y = np.array(labels)


# Training Simple Classifier

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))
