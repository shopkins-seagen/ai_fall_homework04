import json
from pathlib import Path

from htmltools import head_content
from shiny import App, reactive, render, ui
from classifier_backend import classify_condition


ARTIFACT_DIR = Path(__file__).parent / "artifacts"
METRICS_PATH = ARTIFACT_DIR / "condition_organ_system_metrics.json"


def read_saved_metrics():
    if not METRICS_PATH.exists():
        return None
    with METRICS_PATH.open(encoding="utf-8") as metrics_file:
        return json.load(metrics_file)


def metric_grid(metrics):
    return ui.div(
        ui.div("Accuracy", ui.span(f"{metrics['accuracy']:.1%}", class_="metric-value"), class_="metric-item"),
        ui.div("Precision", ui.span(f"{metrics['precision']:.1%}", class_="metric-value"), class_="metric-item"),
        ui.div("Recall", ui.span(f"{metrics['recall']:.1%}", class_="metric-value"), class_="metric-item"),
        ui.div("F1-score", ui.span(f"{metrics['f1_score']:.1%}", class_="metric-value"), class_="metric-item"),
        class_="metrics-grid",
    )


classifier_page = ui.div(
    ui.div(
        ui.div(
            ui.h1("Organ-system classifier"),
            ui.p(
                "Classify a standardized diagnosis description into one of eight "
                "ICD-10-CM chapter-based categories."
            ),
            class_="intro",
        ),
        ui.div(
            ui.input_text_area(
                "condition_text",
                "Diagnosis description",
                placeholder="Example: Asthma, unspecified, uncomplicated",
                rows=3,
                width="100%",
            ),
            ui.input_action_button(
                "classify",
                "Classify description",
                class_="classify-button",
            ),
            class_="query-area",
        ),
        ui.div(ui.output_ui("prediction_result"), class_="result-wrap"),
        class_="app-shell",
    )
)

model_details_page = ui.div(
    ui.div(
        ui.div(
            ui.span("Technical overview", class_="brand"),
            ui.span("Model documentation", class_="demo-tag"),
            class_="topline",
        ),
        ui.div(
            ui.h1("How the classifier works"),
            ui.p(
                "This TensorFlow text-classification project maps standardized "
                "ICD-10-CM diagnosis descriptions to one of eight chapter-based categories."
            ),
            class_="intro",
        ),
        ui.tags.section(
            ui.h2("Data and labels"),
            ui.p(
                "Diagnosis codes and long descriptions come from the free National "
                "Library of Medicine Clinical Tables ICD-10-CM API. The ICD-10-CM "
                "chapter letter supplies the project label; the model receives only "
                "the description text."
            ),
            ui.div(
                ui.div("Chapter", class_="mapping-heading"),
                ui.div("Organ-system label", class_="mapping-heading"),
                ui.div("D", class_="mapping-code"), ui.div("Blood, blood-forming, and immune disorders", class_="mapping-label"),
                ui.div("E", class_="mapping-code"), ui.div("Endocrine, nutritional, and metabolic diseases", class_="mapping-label"),
                ui.div("G", class_="mapping-code"), ui.div("Nervous system", class_="mapping-label"),
                ui.div("I", class_="mapping-code"), ui.div("Cardiovascular system", class_="mapping-label"),
                ui.div("J", class_="mapping-code"), ui.div("Respiratory system", class_="mapping-label"),
                ui.div("K", class_="mapping-code"), ui.div("Digestive system", class_="mapping-label"),
                ui.div("M", class_="mapping-code"), ui.div("Musculoskeletal system", class_="mapping-label"),
                ui.div("N", class_="mapping-code"), ui.div("Genitourinary system", class_="mapping-label"),
                class_="mapping-grid",
            ),
            ui.p(
                "Training uses all retrieved descriptions from the eight selected chapters, "
                "plus curated common names and abbreviations added only to the training "
                "split. The official descriptions use a stratified 80/20 train/test split, "
                "and inverse-frequency class weights reduce the effect of uneven chapter sizes.",
                class_="detail-note",
            ),
            class_="detail-section",
        ),
        ui.tags.section(
            ui.h2("TensorFlow model"),
            ui.p("The model processes raw description text through these Keras layers:"),
            ui.tags.ol(
                ui.tags.li("TextVectorization creates integer token sequences (up to 8,000 tokens and 40 positions)."),
                ui.tags.li("Embedding learns 64-dimensional representations for tokens."),
                ui.tags.li("GlobalAveragePooling1D combines token vectors into a description representation."),
                ui.tags.li("Dense (64 units, ReLU), Dropout (0.25), and Dense (8 units, softmax) produce class probabilities."),
                class_="architecture-list",
            ),
            class_="detail-section",
        ),
        ui.tags.section(
            ui.h2("Held-out evaluation"),
            ui.p(
                "Precision, recall, and F1 are macro averages, so each organ-system "
                "category contributes equally. These are model-level test-set metrics, "
                "not confidence scores for an individual prediction."
            ),
            ui.output_ui("evaluation_summary"),
            class_="detail-section",
        ),
        ui.tags.section(
            ui.h2("Scope and limitations"),
            ui.p(
                "The chapter-based labels are a simplified educational proxy, not "
                "official MedDRA System Organ Classes. A condition may affect multiple "
                "systems, while this exercise assigns one label. This classifier is "
                "not intended for diagnosis, treatment, or real-world medical coding."
            ),
            ui.a(
                "NLM Clinical Tables ICD-10-CM API documentation",
                href="https://clinicaltables.nlm.nih.gov/apidoc/icd10cm/v3/doc.html",
                target="_blank",
                rel="noopener noreferrer",
            ),
            class_="detail-section",
        ),
        class_="app-shell details-shell",
    )
)

app_ui = ui.page_navbar(
    head_content(
        ui.tags.link(
            rel="stylesheet",
            href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.7.2/css/all.min.css",
            crossorigin="anonymous",
        ),
        ui.include_css(Path(__file__).parent / "styles.css"),
    ),
    ui.nav_panel(
        ui.span(
            ui.tags.i(class_="fa-solid fa-house nav-icon", aria_hidden="true"),
            ui.span("Classifier"),
            class_="nav-item-label",
        ),
        classifier_page,
        value="classifier",
    ),
    ui.nav_panel(
        ui.span(
            ui.tags.i(class_="fa-solid fa-brain nav-icon", aria_hidden="true"),
            ui.span("Model details"),
            class_="nav-item-label",
        ),
        model_details_page,
        value="model-details",
    ),
    ui.nav_control(
        ui.a(
            ui.span(
                ui.tags.i(class_="fa-brands fa-github nav-icon", aria_hidden="true"),
                ui.span("Source Code"),
                class_="nav-item-label",
            ),
            href="https://github.com/shopkins-seagen/ai_fall_homework04",
            target="_blank",
            rel="noopener noreferrer",
            class_="nav-link external-nav-link",
            aria_label="Open the project repository on GitHub",
        )
    ),
    title=ui.span(
        ui.span("BIA 562 Homework 4: TensorFlow Text Classifier", class_="navbar-title-text"),
        ui.span("Shawn Hopkins", class_="navbar-title-author"),
        class_="navbar-title-row",
    ),
    id="main_nav",
    selected="classifier",
    window_title="Organ-System Classifier",
)


def server(input, output, session):
    result = reactive.value(None)
    model_metrics = reactive.value(read_saved_metrics())

    @reactive.effect
    @reactive.event(input.classify)
    def _classify():
        if input.classify() == 0:
            return

        description = input.condition_text().strip()
        if not description:
            result.set({"error": "Enter a diagnosis description to classify."})
            return

        try:
            prediction = classify_condition(description)
            model_metrics.set(prediction["metrics"])
            result.set(prediction)
        except Exception:
            result.set({"error": "The description could not be classified. Try another description."})

    @render.ui
    def evaluation_summary():
        metrics = model_metrics.get()
        if metrics is None:
            return ui.div(
                "Evaluation scores will appear here after the classifier has been trained.",
                class_="detail-note",
            )
        return metric_grid(metrics)

    @render.ui
    def prediction_result():
        current_result = result.get()
        if current_result is None:
            return ui.div(
                "Your classification will appear here.",
                class_="empty-state",
            )

        if "error" in current_result:
            return ui.div(current_result["error"], class_="error-state")

        percentage = max(0.0, min(100.0, current_result["confidence"] * 100))
        metrics = current_result["metrics"]
        return ui.div(
            ui.div("Predicted organ system", class_="result-kicker"),
            ui.h2(current_result["predicted_system"], class_="result-system"),
            ui.div(
                ui.span("Model confidence"),
                ui.span(f"{percentage:.1f}%", class_="confidence-value"),
                class_="confidence-row",
            ),
            ui.div(
                ui.div(
                    class_="confidence-fill",
                    style=f"width: {percentage:.1f}%",
                ),
                class_="confidence-track",
                role="img",
                aria_label=f"Confidence {percentage:.1f}%",
            ),
            ui.div(
                ui.div("Model performance · held-out test set", class_="metrics-heading"),
                metric_grid(metrics),
                class_="model-metrics",
            ),
            class_="result-card",
        )


app = App(app_ui, server)