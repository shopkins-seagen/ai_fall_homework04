import json
import re
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
MODEL_REVISION_PATH = ARTIFACT_DIR / "condition_organ_system_revision.json"
MODEL_REVISION = 3

SYSTEM_CHAPTERS = {
    "D": "Blood, blood-forming, and immune disorders",
    "E": "Endocrine, nutritional, and metabolic diseases",
    "G": "Nervous system",
    "I": "Cardiovascular system",
    "J": "Respiratory system",
    "K": "Digestive system",
    "M": "Musculoskeletal system",
    "N": "Genitourinary system",
}
PAGE_SIZE = 500
MAX_RESULTS_PER_QUERY = 7500
MAX_TOKENS = 8000
SEQUENCE_LENGTH = 40
SEED = 42
TRAINING_ALIASES = {
    "Nervous system": [
        "ALS",
        "amyotrophic lateral sclerosis",
        "Alzheimer's disease",
        "Parkinson's disease",
        "epilepsy",
        "seizure disorder",
        "multiple sclerosis",
        "migraine",
        "Guillain-Barre syndrome",
        "peripheral neuropathy",
        "polyneuropathy",
        "Huntington's disease",
        "essential tremor",
    ],
    "Cardiovascular system": [
        "AFib",
        "atrial fibrillation",
        "CAD",
        "coronary artery disease",
        "CHF",
        "congestive heart failure",
        "heart failure",
        "heart attack",
        "myocardial infarction",
        "high blood pressure",
        "hypertension",
        "DVT",
        "deep vein thrombosis",
        "angina",
        "peripheral artery disease",
    ],
    "Blood, blood-forming, and immune disorders": [
        "anemia",
        "iron deficiency anemia",
        "sickle cell disease",
        "thalassemia",
        "aplastic anemia",
        "hemophilia",
        "von Willebrand disease",
        "thrombocytopenia",
        "immune thrombocytopenia",
        "neutropenia",
        "coagulopathy",
        "clotting disorder",
    ],
    "Endocrine, nutritional, and metabolic diseases": [
        "diabetes",
        "diabetes mellitus",
        "type 1 diabetes",
        "type 2 diabetes",
        "hypothyroidism",
        "hyperthyroidism",
        "Hashimoto's thyroiditis",
        "Graves' disease",
        "obesity",
        "metabolic syndrome",
        "hyperlipidemia",
        "malnutrition",
        "Cushing syndrome",
        "Addison disease",
        "polycystic ovary syndrome",
    ],
    "Respiratory system": [
        "COPD",
        "chronic obstructive pulmonary disease",
        "asthma",
        "pneumonia",
        "bronchitis",
        "emphysema",
        "interstitial lung disease",
        "pulmonary fibrosis",
        "bronchiectasis",
        "pleural effusion",
    ],
    "Digestive system": [
        "IBD",
        "inflammatory bowel disease",
        "Crohn's disease",
        "Crohn disease",
        "ulcerative colitis",
        "IBS",
        "irritable bowel syndrome",
        "GERD",
        "gastroesophageal reflux disease",
        "acid reflux",
        "celiac disease",
        "diverticulitis",
        "pancreatitis",
        "gallstones",
    ],
    "Musculoskeletal system": [
        "RA",
        "rheumatoid arthritis",
        "OA",
        "osteoarthritis",
        "gout",
        "osteoporosis",
        "fibromyalgia",
        "lupus",
        "systemic lupus erythematosus",
        "ankylosing spondylitis",
        "scoliosis",
    ],
    "Genitourinary system": [
        "CKD",
        "chronic kidney disease",
        "UTI",
        "urinary tract infection",
        "kidney stones",
        "nephrolithiasis",
        "cystitis",
        "bladder infection",
        "endometriosis",
        "benign prostatic hyperplasia",
        "BPH",
        "ovarian cyst",
        "urinary incontinence",
        "kidney failure",
    ],
}
ABBREVIATION_EXPANSIONS = {
    "ALS": "amyotrophic lateral sclerosis",
    "AFib": "atrial fibrillation",
    "CAD": "coronary artery disease",
    "CHF": "congestive heart failure",
    "DVT": "deep vein thrombosis",
    "COPD": "chronic obstructive pulmonary disease",
    "IBD": "inflammatory bowel disease",
    "IBS": "irritable bowel syndrome",
    "GERD": "gastroesophageal reflux disease",
    "RA": "rheumatoid arthritis",
    "OA": "osteoarthritis",
    "CKD": "chronic kidney disease",
    "UTI": "urinary tract infection",
    "BPH": "benign prostatic hyperplasia",
}


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

    if set(conditions["organ_system"].unique()) != set(SYSTEM_CHAPTERS.values()):
        raise RuntimeError("The API did not return examples for every configured organ system.")

    return conditions.sample(frac=1, random_state=SEED).reset_index(drop=True)


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

    aliases = pd.DataFrame(
        [
            {"description": alias, "organ_system": system_name}
            for system_name, alias_list in TRAINING_ALIASES.items()
            for alias in alias_list
        ]
    )
    training_examples = pd.concat(
        [
            pd.DataFrame({"description": X_train, "organ_system": y_train_text}),
            aliases,
        ],
        ignore_index=True,
    ).sample(frac=1, random_state=SEED).reset_index(drop=True)

    X_train = training_examples["description"]
    y_train_text = training_examples["organ_system"]
    label_encoder = LabelEncoder().fit(y_train_text)
    class_names = label_encoder.classes_.tolist()
    y_train = label_encoder.transform(y_train_text)
    train_text = tf.constant(X_train.tolist(), dtype=tf.string)
    class_counts = np.bincount(y_train, minlength=len(class_names))
    class_weights = {
        index: len(y_train) / (len(class_names) * count)
        for index, count in enumerate(class_counts)
    }

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
        class_weight=class_weights,
        verbose=1,
    )

    metrics = _evaluate_model(model, X_test, y_test_text, class_names)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    with LABELS_PATH.open("w", encoding="utf-8") as labels_file:
        json.dump(class_names, labels_file, indent=2)
    with METRICS_PATH.open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)
    with MODEL_REVISION_PATH.open("w", encoding="utf-8") as revision_file:
        json.dump({"revision": MODEL_REVISION}, revision_file, indent=2)
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
    if MODEL_PATH.exists() and LABELS_PATH.exists() and MODEL_REVISION_PATH.exists():
        with MODEL_REVISION_PATH.open(encoding="utf-8") as revision_file:
            revision = json.load(revision_file).get("revision")
        if revision != MODEL_REVISION:
            return _train_and_save_model()

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
    normalized_description = description
    for abbreviation, expansion in ABBREVIATION_EXPANSIONS.items():
        normalized_description = re.sub(
            rf"\b{re.escape(abbreviation)}\b",
            expansion,
            normalized_description,
            flags=re.IGNORECASE,
        )
    probabilities = model.predict(
        tf.constant([normalized_description], dtype=tf.string),
        verbose=0,
    )[0]
    best_index = int(np.argmax(probabilities))
    return {
        "predicted_system": class_names[best_index],
        "confidence": float(probabilities[best_index]),
        "metrics": metrics,
    }