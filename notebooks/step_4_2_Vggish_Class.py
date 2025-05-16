# notebooks/step_4_2_Vggish_Class.py
# L 5-15-20 & Aditya
import os
import numpy as np
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt

# Constants
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_EMBEDDING_DIR = os.path.join(
    PROJECT_ROOT, "reports/step_4_1_vggish_embeddings")
DEFAULT_METADATA_FILE = os.path.join(PROJECT_ROOT, "data", "metadata.csv")
DEFAULT_REPORT_DIR = os.path.join(
    PROJECT_ROOT, "reports/step_4_2_vggish_class")
DEFAULT_MODEL_SAVE_PATH = os.path.join(
    PROJECT_ROOT, "models/vggish_rf_model.pkl")
DEFAULT_CLASS_NAMES_PATH = os.path.join(
    DEFAULT_REPORT_DIR, "vggish_rf_class_names.npy")

os.makedirs(DEFAULT_REPORT_DIR, exist_ok=True)


def load_metadata(metadata_file, embedding_dir):
    df = pd.read_csv(metadata_file, dtype={"track_id": str})
    available_ids = {f.split(".")[0] for f in os.listdir(
        embedding_dir) if f.endswith(".npy")}
    df = df[df["track_id"].isin(available_ids)]
    df['label'] = LabelEncoder().fit_transform(df['genre'])
    df['track_id_str'] = df['track_id']
    return df


def load_embeddings(df, embedding_dir):
    X, y, missing = [], [], []
    for _, row in df.iterrows():
        track_id = row["track_id_str"]
        label = row["label"]
        path = os.path.join(embedding_dir, f"{track_id}.npy")

        if os.path.exists(path):
            embedding = np.load(path)
            if embedding.ndim == 2:
                mean_embedding = embedding.mean(axis=0)
            elif embedding.ndim == 1:
                mean_embedding = embedding
            else:
                print(f"[SKIP] {track_id}: Unexpected shape {embedding.shape}")
                missing.append(track_id)
                continue

            # Expected VGGish embedding size
            if mean_embedding.shape != (128,):
                print(
                    f"[SKIP] {track_id}: Invalid embedding shape {mean_embedding.shape}")
                missing.append(track_id)
                continue

            X.append(mean_embedding)
            y.append(label)
        else:
            missing.append(track_id)

    return np.array(X), np.array(y), missing


def train_rf_model(X_train, y_train):
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    return clf


def evaluate_rf_model(clf, X_test, y_test, class_names, report_dir):
    os.makedirs(report_dir, exist_ok=True)
    y_pred = clf.predict(X_test)

    report = classification_report(y_test, y_pred, target_names=class_names)
    print(report)
    with open(os.path.join(report_dir, "vggish_rf_classification_report.txt"), "w") as f:
        f.write(report)

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title("Confusion Matrix - VGGish + RF")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha='right')
    plt.yticks(tick_marks, class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.tight_layout()
    plt.savefig(os.path.join(report_dir, "vggish_rf_confusion_matrix.png"))
    plt.close()

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, labels=range(len(class_names)), zero_division=0
    )
    for metric_name, values in zip(["Precision", "Recall", "F1-Score"], [precision, recall, f1]):
        plt.figure(figsize=(10, 5))
        plt.bar(class_names, values)
        plt.ylabel(metric_name)
        plt.title(f"Per-Genre {metric_name} - VGGish + RF")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        filename = f"vggish_rf_per_genre_{metric_name.lower().replace('-', '_')}.png"
        plt.savefig(os.path.join(report_dir, filename))
        plt.close()


def train_vggish_rf(
    embedding_dir=DEFAULT_EMBEDDING_DIR,
    metadata_file=DEFAULT_METADATA_FILE,
    report_dir=DEFAULT_REPORT_DIR,
    model_save_path=DEFAULT_MODEL_SAVE_PATH,
    class_names_path=DEFAULT_CLASS_NAMES_PATH
):
    print("[INFO] Loading metadata and embeddings...")
    df = load_metadata(metadata_file, embedding_dir)
    label_encoder = LabelEncoder().fit(df["genre"])
    df['label'] = label_encoder.transform(df["genre"])
    class_names = label_encoder.classes_
    X, y, _ = load_embeddings(df, embedding_dir)
    print(f"[INFO] Loaded {len(X)} embeddings")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42)
    clf = train_rf_model(X_train, y_train)
    evaluate_rf_model(clf, X_test, y_test, class_names, report_dir)
    joblib.dump(clf, model_save_path)
    np.save(class_names_path, class_names)
    print(f"[INFO] Saved trained model to {model_save_path}")


def evaluate_on_new_embeddings(model_path, embedding_dir, metadata_file, report_dir, class_names_path=DEFAULT_CLASS_NAMES_PATH):
    print("[INFO] Evaluating on new embeddings...")
    df = load_metadata(metadata_file, embedding_dir)
    df['label'] = LabelEncoder().fit_transform(df["genre"])
    class_names = np.load(class_names_path, allow_pickle=True)
    X, y, _ = load_embeddings(df, embedding_dir)
    clf = joblib.load(model_path)
    evaluate_rf_model(clf, X, y, class_names, report_dir)


def predict_vggish_for_files(model_path, embedding_dir, metadata_file, class_names_path=DEFAULT_CLASS_NAMES_PATH):
    df = load_metadata(metadata_file, embedding_dir)
    class_names = np.load(class_names_path, allow_pickle=True)
    clf = joblib.load(model_path)
    X, _, missing = load_embeddings(df, embedding_dir)

    preds = clf.predict(X)
    track_ids = df["track_id"].values
    genres = df["genre"].values
    pred_labels = [class_names[p] for p in preds]

    return pd.DataFrame({
        "file": [f"{tid}.mp3" for tid in track_ids],
        "true_genre": genres,
        "VGGISH": pred_labels
    })


if __name__ == "__main__":
    train_vggish_rf()
