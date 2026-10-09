import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


API_URL = "https://clinicaltables.nlm.nih.gov/api/icd10cm/v3/search"
ARTIFACT_DIR = Path(__file__).parent / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "condition_organ_system.keras"
LABELS_PATH = ARTIFACT_DIR / "condition_organ_system_labels.json"
METRICS_PATH = ARTIFACT_DIR / "condition_organ_system_metrics.json"

SYSTEM_CHAPTERS = {
    "G": "Nervous system",
    "I": "Cardiovascular system",
    "J": "Respiratory system",
    "K": "Digestive system",
    "M": "Musculoskeletal system",
    "N": "Genitourinary system",
}
PAGE_SIZE = 500
MAX_RESULTS_PER_QUERY = 7500
MAX_EXAMPLES_PER_CLASS = 350
MAX_TOKENS = 8000
SEQUENCE_LENGTH = 40
SEED = 42


def _fetch_chapter(chapter: str) -> tuple[list[dict[str, str]], int]:
    rows = []
    offset = 0
    total_count = 0

    while offset < MAX_RESULTS_PER_QUERY:
        response = requests.get(
            API_URL,
            params={
                "terms": chapter,
                "sf": "code",
                "df": "code,name",
                "count": PAGE_SIZE,
                "offset": offset,
            },
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        total_count = payload[0]
        page_rows = payload[3]

        if not page_rows:
            break

        rows.extend(
            {"code": code, "description": name, "organ_system": SYSTEM_CHAPTERS[chapter]}
            for code, name in page_rows
            if code.startswith(chapter)
        )
        offset += len(page_rows)

        if offset >= min(total_count, MAX_RESULTS_PER_QUERY):
            break

    return rows, total_count


def _fetch_training_data() -> pd.DataFrame:
    records = []
    for chapter, system_name in SYSTEM_CHAPTERS.items():
        chapter_records, total_matches = _fetch_chapter(chapter)
        records.extend(chapter_records)
        print(f"{system_name}: fetched {len(chapter_records):,} descriptions (API matched {total_matches:,})")

    conditions = pd.DataFrame(records).drop_duplicates(subset="code").reset_index(drop=True)
    conditions = conditions.dropna(subset=["description", "organ_system"])
    conditions["description"] = conditions["description"].str.strip()
    conditions = conditions[conditions["description"].ne("")]

    balanced_parts = []
    for _, group in conditions.groupby("organ_system"):
        sample_size = min(MAX_EXAMPLES_PER_CLASS, len(group))
        if sample_size:
            balanced_parts.append(group.sample(n=sample_size, random_state=SEED))

    if len(balanced_parts) != len(SYSTEM_CHAPTERS):
        raise RuntimeError("The API did not return examples for every configured organ system.")

    return (
        pd.concat(balanced_parts, ignore_index=True)
        .sample(frac=1, random_state=SEED)
        .reset_index(drop=True)
    )


def _train_and_save_model() -> tuple[tf.keras.Model, list[str]]:
    print("Model artifacts are missing. Fetching ICD-10-CM data and training the classifier...")
    conditions = _fetch_training_data()
    X_train, X_test, y_train_text, y_test_text = train_test_split(
        conditions["description"],
        conditions["organ_system"],
        test_size=0.2,
        random_state=SEED,
        stratify=conditions["organ_system"],
    )

    label_encoder = LabelEncoder().fit(y_train_text)
    y_train = label_encoder.transform(y_train_text)
    train_text = tf.constant(X_train.tolist(), dtype=tf.string)

    vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=MAX_TOKENS,
        output_mode="int",
        output_sequence_length=SEQUENCE_LENGTH,
    )
    vectorizer.adapt(train_text)

    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(), dtype=tf.string),
            vectorizer,
            tf.keras.layers.Embedding(MAX_TOKENS, 64, mask_zero=True),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Dense(len(label_encoder.classes_), activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True,
    )
    model.fit(
        train_text,
        y_train,
        validation_split=0.15,
        epochs=15,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1,
    )

    class_names = label_encoder.classes_.tolist()
    metrics = _evaluate_model(model, X_test, y_test_text, class_names)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    with LABELS_PATH.open("w", encoding="utf-8") as labels_file:
        json.dump(class_names, labels_file, indent=2)
    with METRICS_PATH.open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)
    return model, class_names, metrics


def _evaluate_model(
    model: tf.keras.Model,
    descriptions: pd.Series,
    labels: pd.Series,
    class_names: list[str],
) -> dict[str, float]:
    class_indices = {name: index for index, name in enumerate(class_names)}
    y_true = np.array([class_indices[label] for label in labels])
    probabilities = model.predict(
        tf.constant(descriptions.tolist(), dtype=tf.string),
        verbose=0,
    )
    y_pred = probabilities.argmax(axis=1)
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1_score),
    }


@lru_cache(maxsize=1)
def _load_classifier() -> tuple[tf.keras.Model, list[str], dict[str, float]]:
    if MODEL_PATH.exists() and LABELS_PATH.exists():
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        with LABELS_PATH.open(encoding="utf-8") as labels_file:
            class_names = json.load(labels_file)
        if METRICS_PATH.exists():
            with METRICS_PATH.open(encoding="utf-8") as metrics_file:
                metrics = json.load(metrics_file)
        else:
            conditions = _fetch_training_data()
            _, X_test, _, y_test_text = train_test_split(
                conditions["description"],
                conditions["organ_system"],
                test_size=0.2,
                random_state=SEED,
                stratify=conditions["organ_system"],
            )
            metrics = _evaluate_model(model, X_test, y_test_text, class_names)
            with METRICS_PATH.open("w", encoding="utf-8") as metrics_file:
                json.dump(metrics, metrics_file, indent=2)
        return model, class_names, metrics

    return _train_and_save_model()


def classify_condition(description: str) -> dict[str, str | float]:
    """Return the predicted organ system and its model confidence."""
    model, class_names, metrics = _load_classifier()
    probabilities = model.predict(
        tf.constant([description], dtype=tf.string),
        verbose=0,
    )[0]
    best_index = int(np.argmax(probabilities))
    return {
        "predicted_system": class_names[best_index],
        "confidence": float(probabilities[best_index]),
        "metrics": metrics,
    }