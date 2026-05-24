"""
IS 108 – Business Intelligence Predictive Modeling Application
================================================================
Loan approval classification (primary use case) with full BI workflow:
  1. Problem identification
  2. Data collection
  3. Preprocessing (impute, encode, scale, train/test split)
  4. Feature selection (optional SelectKBest)
  5. Model training — KNN, SVM, ANN (scikit-learn)
  6. Evaluation — accuracy, precision, recall, F1, confusion matrix
  7. Prediction on new loan applications

Run: python -m streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

# --- scikit-learn: preprocessing & feature selection ---
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import clone

# --- scikit-learn: the three required classifiers ---
from sklearn.neighbors import KNeighborsClassifier  # KNN
from sklearn.svm import SVC                          # SVM
from sklearn.neural_network import MLPClassifier     # ANN (multi-layer perceptron)

from sklearn.datasets import load_breast_cancer, load_iris


# Streamlit page layout (wide mode for tables and metric comparison)
st.set_page_config(
    page_title="BI Predictive Modeling",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Project folder paths (loan_approval_dataset.csv lives next to this file)
APP_DIR = Path(__file__).resolve().parent
LOCAL_LOAN_CSV = APP_DIR / "loan_approval_dataset.csv"


# =============================================================================
# DATA LOADING — Step 2 of the rubric (data collection)
# =============================================================================


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Strip stray spaces in column names and text fields (common in CSV exports)."""
    out = df.copy()
    out.columns = out.columns.str.strip()
    for col in out.select_dtypes(include=["object", "string"]).columns:
        out[col] = out[col].astype(str).str.strip()
    return out


def load_uploaded_file(uploaded: Any) -> pd.DataFrame:
    """Read user-uploaded CSV or Excel into a cleaned DataFrame."""
    name = uploaded.name.lower()
    if name.endswith(".csv"):
        return normalize_dataframe(pd.read_csv(uploaded))
    if name.endswith((".xlsx", ".xls")):
        return normalize_dataframe(pd.read_excel(uploaded))
    raise ValueError("Please upload a CSV or Excel file.")


def load_local_loan_csv() -> pd.DataFrame:
    """Load loan_approval_dataset.csv from the same folder as app.py."""
    if not LOCAL_LOAN_CSV.is_file():
        raise FileNotFoundError(
            f"Expected `{LOCAL_LOAN_CSV.name}` in the project folder: {APP_DIR}"
        )
    return normalize_dataframe(pd.read_csv(LOCAL_LOAN_CSV))


# Default business-problem text shown on Tab 1 when a loan dataset is loaded
LOAN_PROBLEM_TEXT = (
    "A lender must decide whether to approve loan applications. Using applicant "
    "profile, income, loan amount, credit history, and property area, we predict "
    "Loan_Status (Approved = Y / Rejected = N) to reduce default risk and speed "
    "up underwriting."
)


def load_loan_approval_dataset(random_state: int = 42) -> pd.DataFrame:
    """Synthetic loan-approval data (classic banking classification schema)."""
    rng = np.random.default_rng(random_state)
    n = 610
    gender = rng.choice(["Male", "Female"], n)
    married = rng.choice(["Yes", "No"], n, p=[0.65, 0.35])
    dependents = rng.choice(["0", "1", "2", "3+"], n, p=[0.48, 0.22, 0.18, 0.12])
    education = rng.choice(["Graduate", "Not Graduate"], n, p=[0.78, 0.22])
    self_employed = rng.choice(["Yes", "No"], n, p=[0.14, 0.86])
    property_area = rng.choice(
        ["Urban", "Semiurban", "Rural"], n, p=[0.35, 0.35, 0.30]
    )
    applicant_income = rng.lognormal(mean=8.45, sigma=0.55, size=n)
    applicant_income = np.clip(applicant_income, 1500, 85000).astype(int)
    coapplicant_income = np.zeros(n, dtype=int)
    has_co = rng.random(n) < 0.58
    coapplicant_income[has_co] = np.clip(
        rng.lognormal(mean=7.8, sigma=0.75, size=has_co.sum()), 0, 50000
    ).astype(int)
    loan_amount = (applicant_income + coapplicant_income) * rng.uniform(0.04, 0.28, n) / 1000
    loan_amount = np.clip(loan_amount, 9, 700)
    loan_amount[rng.random(n) < 0.04] = np.nan
    loan_term = rng.choice([12, 24, 36, 60, 120, 180, 240, 300, 360], n)
    credit_history = rng.choice([1.0, 0.0], n, p=[0.82, 0.18])

    # Logistic-style rule so approval correlates with income, credit, etc.
    grad = (education == "Graduate").astype(float)
    urban = (property_area == "Urban").astype(float)
    logit = (
        -1.2
        + 2.8 * credit_history
        + 0.35 * grad
        + 0.15 * urban
        + 3.5e-5 * applicant_income
        + 2.0e-5 * coapplicant_income
        - 0.012 * loan_amount
        + 0.25 * (married == "Yes")
        - 0.35 * (self_employed == "Yes")
    )
    prob = 1.0 / (1.0 + np.exp(-logit))
    loan_status = np.where(rng.random(n) < prob, "Y", "N")

    return pd.DataFrame(
        {
            "Loan_ID": [f"LP{i:06d}" for i in range(1, n + 1)],
            "Gender": gender,
            "Married": married,
            "Dependents": dependents,
            "Education": education,
            "Self_Employed": self_employed,
            "ApplicantIncome": applicant_income,
            "CoapplicantIncome": coapplicant_income,
            "LoanAmount": np.round(loan_amount, 1),
            "Loan_Amount_Term": loan_term,
            "Credit_History": credit_history,
            "Property_Area": property_area,
            "Loan_Status": loan_status,
        }
    )


def load_sample_dataset(choice: str) -> pd.DataFrame:
    """Built-in demos: synthetic loan, Iris (multiclass), breast cancer (binary)."""
    if choice.startswith("Iris"):
        raw = load_iris(as_frame=True)
        return raw.frame.copy()
    if choice.startswith("Loan"):
        return load_loan_approval_dataset()
    raw = load_breast_cancer(as_frame=True)
    return raw.frame.copy()


def is_loan_dataset(df: pd.DataFrame) -> bool:
    """True when target is loan_status (Approved / Rejected)."""
    return any(c.lower() == "loan_status" for c in df.columns)


def default_target_column(cols: list[str]) -> int:
    """Pick loan_status as target when present; else 'target' or last column."""
    for i, c in enumerate(cols):
        if c.strip().lower() == "loan_status":
            return i
    if "target" in cols:
        return cols.index("target")
    return len(cols) - 1


def is_id_column(col: str) -> bool:
    """IDs are not predictive — exclude loan_id from model features."""
    name = col.strip().lower()
    return name in ("loan_id", "id") or name.endswith("_id")


def uses_classic_loan_form(numeric_cols: list[str], categorical_cols: list[str]) -> bool:
    """Classic Kaggle-style loan form vs. project CSV (income_annum, cibil_score, …)."""
    return "ApplicantIncome" in numeric_cols and "Gender" in categorical_cols


# =============================================================================
# PREPROCESSING — Step 3 (imputation, encoding, scaling)
# =============================================================================


def detect_feature_columns(df: pd.DataFrame, target: str) -> tuple[list[str], list[str]]:
    """Split features into numeric vs. categorical for separate sklearn pipelines."""
    feature_cols = [c for c in df.columns if c != target]
    numeric_cols: list[str] = []
    categorical_cols: list[str] = []
    for c in feature_cols:
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric_cols.append(c)
        else:
            categorical_cols.append(c)
    return numeric_cols, categorical_cols


def build_preprocessor(
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> ColumnTransformer:
    """
    Build preprocessing for both column types:
      - Numeric: fill missing with median, then StandardScaler
      - Categorical: fill missing with mode, then one-hot encoding
    """
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric_cols:
        transformers.append(("num", numeric_pipe, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", categorical_pipe, categorical_cols))
    return ColumnTransformer(transformers=transformers, remainder="drop")


# =============================================================================
# MODEL PIPELINES — Steps 4–5 (feature selection + KNN / SVM / ANN)
# =============================================================================


def make_classifier_pipeline(
    preprocessor: ColumnTransformer,
    k_features: int | None,
    classifier: Any,
) -> Pipeline:
    """
    Chain: preprocessor → optional SelectKBest (ANOVA F-score) → classifier.
    The classifier argument is KNeighborsClassifier, SVC, or MLPClassifier.
    """
    steps: list[tuple[str, Any]] = [("prep", preprocessor)]
    if k_features and k_features > 0:
        steps.append(("select", SelectKBest(score_func=f_classif, k=int(k_features))))
    steps.append(("clf", classifier))
    return Pipeline(steps=steps)


# =============================================================================
# EVALUATION — Step 6 (metrics required by rubric)
# =============================================================================


def train_eval_model(name: str, clf: Any, X_train, X_test, y_train, y_test, labels) -> dict[str, Any]:
    """
    Fit one model pipeline, predict on the test set, and compute:
    accuracy, precision, recall, F1, confusion matrix, classification report.
    """
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    report = classification_report(y_test, y_pred, zero_division=0)
    return {
        "name": name,
        "model": clf,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm,
        "report": report,
        "y_test": y_test,
        "y_pred": y_pred,
    }


def plot_confusion_matrix(cm: np.ndarray, labels: list[Any], title: str) -> plt.Figure:
    """Heatmap of actual vs. predicted classes for one model (KNN, SVM, or ANN)."""
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    plt.tight_layout()
    return fig


# =============================================================================
# STREAMLIT UI — maps each tab to one rubric step
# =============================================================================


def main() -> None:
    st.title("Business Intelligence — Predictive Modeling")
    st.caption(
        "Complete workflow: data loading, preprocessing, feature selection, "
        "KNN / SVM / ANN training, evaluation, and comparison."
    )

    with st.sidebar:
        st.header("Workflow (rubric)")
        st.markdown(
            """
1. **Problem identification** — Choose target & business context below  
2. **Data collection** — Upload CSV/Excel or use a sample dataset  
3. **Preprocessing** — Missing values, encoding, scaling, train/test split  
4. **Feature selection** — Optional *k* best features (ANOVA F-score)  
5. **Model training & testing** — KNN, SVM, ANN  
6. **Evaluation** — Accuracy, Precision, Recall, F1, confusion matrix  
7. **Predictions** — Use trained winner on new rows (optional)  
            """
        )

    tab_intro, tab_data, tab_process, tab_models, tab_predict = st.tabs(
        [
            "1. Problem & goal",
            "2. Dataset",
            "3. Preprocessing",
            "4. Models & evaluation",
            "5. Predict",
        ]
    )

    # ----- Tab 1: Problem identification (rubric step 1) -----
    with tab_intro:
        st.subheader("Problem identification")
        intro_default = (
            LOAN_PROBLEM_TEXT
            if st.session_state.get("df") is not None
            and is_loan_dataset(st.session_state.df)
            else "Predict the target class from available features to support operational decisions "
            "(e.g. risk screening, retention, or quality control)."
        )
        business_problem = st.text_area(
            "Describe the business problem (e.g. customer churn, loan approval, fraud detection)",
            value=intro_default,
            height=100,
        )
        st.info(
            "This app supports **classification** problems so we can report "
            "precision, recall, F1, and confusion matrices as required."
        )

    # Persist loaded data and target column across Streamlit reruns
    if "df" not in st.session_state:
        st.session_state.df = None
    if "target_col" not in st.session_state:
        st.session_state.target_col = None

    # ----- Tab 2: Data collection (rubric step 2) -----
    with tab_data:
        st.subheader("Data collection & dataset handling")
        source = st.radio(
            "Data source",
            [
                "Load loan_approval_dataset.csv (this folder)",
                "Upload CSV / Excel",
                "Use built-in sample",
            ],
            horizontal=True,
        )

        if source == "Load loan_approval_dataset.csv (this folder)":
            if LOCAL_LOAN_CSV.is_file():
                if st.button("Load project loan CSV", type="primary"):
                    try:
                        st.session_state.df = load_local_loan_csv()
                        st.session_state.target_col = next(
                            c for c in st.session_state.df.columns if c.lower() == "loan_status"
                        )
                    except Exception as e:
                        st.error(str(e))
                st.caption(f"Found **`{LOCAL_LOAN_CSV.name}`** in the project folder (4,269 applications).")
            else:
                st.warning(
                    f"Place **`loan_approval_dataset.csv`** in `{APP_DIR}` or use Upload instead."
                )
        elif source == "Upload CSV / Excel":
            uploaded = st.file_uploader("Import dataset", type=["csv", "xlsx", "xls"])
            if uploaded is not None:
                try:
                    st.session_state.df = load_uploaded_file(uploaded)
                    st.success(f"Loaded **{uploaded.name}** — shape `{st.session_state.df.shape}`.")
                except Exception as e:
                    st.error(str(e))
        else:
            sample_choice = st.selectbox(
                "Sample dataset",
                [
                    "Loan approval (binary)",
                    "Iris (multiclass)",
                    "Breast cancer (binary)",
                ],
                key="sample_choice",
            )
            if st.button("Load sample into session"):
                st.session_state.df = load_sample_dataset(sample_choice)
                if sample_choice.startswith("Loan"):
                    st.session_state.target_col = "Loan_Status"

        df = st.session_state.df
        if df is not None:
            st.subheader("Tabular preview")
            st.dataframe(df.head(50), use_container_width=True)
            st.subheader("Basic information")
            c1, c2, c3 = st.columns(3)
            c1.metric("Rows", f"{len(df):,}")
            c2.metric("Columns", len(df.columns))
            missing = int(df.isna().sum().sum())
            c3.metric("Missing cells (before cleaning)", f"{missing:,}")
            with st.expander("Column names and dtypes"):
                st.dataframe(
                    pd.DataFrame(
                        {"column": df.columns, "dtype": df.dtypes.astype(str).values}
                    ),
                    use_container_width=True,
                )
            cols = list(df.columns)
            st.session_state.target_col = st.selectbox(
                "Target column (what to predict)",
                options=cols,
                index=default_target_column(cols),
            )

    df = st.session_state.df
    target = st.session_state.target_col

    # ----- Tab 3: Preprocessing & train/test split (rubric step 3) -----
    with tab_process:
        st.subheader("Data preprocessing & split")
        if df is None or target is None:
            st.warning("Load a dataset and pick a target column in the **Dataset** tab first.")
        else:
            test_size = st.slider("Test set fraction", 0.1, 0.5, 0.2, 0.05)
            random_state = st.number_input("Random seed", 0, 9999, 42)
            handle_rare = st.checkbox(
                "Drop target classes with very few rows (< 5) for stable metrics",
                value=True,
            )
            k_best = st.number_input(
                "Feature selection: top K numeric features after encoding (0 = no selection)",
                min_value=0,
                max_value=500,
                value=0,
                help="ANOVA F-score (SelectKBest) on the full numeric matrix after encoding/scaling. "
                "If training errors, set 0. K must not exceed the number of columns after preprocessing.",
            )

            work = df.copy()
            y_raw = work[target]
            # Encode target: Approved/Rejected or Y/N → integers for sklearn
            if not pd.api.types.is_numeric_dtype(y_raw):
                y_enc, uniques = pd.factorize(y_raw)
                class_names = list(uniques.astype(str))
                y = pd.Series(y_enc, index=work.index)
            else:
                y_num = pd.to_numeric(y_raw, errors="coerce")
                if y_num.isna().any():
                    st.error("Target column has non-numeric values. Use a categorical/text target or clean the column.")
                    st.stop()
                y = y_num.astype(int)
                class_names = [str(x) for x in np.sort(np.unique(y))]

            drop_id_cols = [c for c in work.columns if is_id_column(c)]
            X = work.drop(columns=[target] + drop_id_cols)  # feature matrix

            if handle_rare:
                vc = y.value_counts()
                keep = vc[vc >= 5].index
                mask = y.isin(keep)
                X, y = X.loc[mask], y.loc[mask]
                st.caption(f"Rows after filtering rare classes: **{len(X)}**")

            numeric_cols, categorical_cols = detect_feature_columns(
                work.drop(columns=drop_id_cols, errors="ignore"), target
            )

            if len(np.unique(y)) < 2:
                st.error("Need at least two classes in the target after preprocessing.")
                st.stop()

            # Hold out test_size for unbiased evaluation (stratify keeps class balance)
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y,
                    test_size=test_size,
                    random_state=int(random_state),
                    stratify=y,
                )
            except ValueError:
                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y,
                    test_size=test_size,
                    random_state=int(random_state),
                    stratify=None,
                )
                st.caption("Stratified split unavailable (small/rare classes); used random split.")

            cat_options: dict[str, list[Any]] = {}
            for col in categorical_cols:
                vals = X_train[col].dropna().unique().tolist()
                cat_options[col] = sorted(vals, key=lambda v: str(v))
            num_defaults = {
                col: float(X_train[col].median())
                for col in numeric_cols
                if col in X_train.columns
            }

            # Save split and column metadata for training (tab 4) and prediction (tab 5)
            st.session_state["train_bundle"] = {
                "X_train": X_train,
                "X_test": X_test,
                "y_train": y_train,
                "y_test": y_test,
                "numeric_cols": numeric_cols,
                "categorical_cols": categorical_cols,
                "class_names": class_names,
                "k_best": int(k_best) if k_best else None,
                "cat_options": cat_options,
                "num_defaults": num_defaults,
                "is_loan": is_loan_dataset(work),
                "classic_loan_form": uses_classic_loan_form(
                    numeric_cols, categorical_cols
                ),
            }
            st.success("Preprocessing configuration saved. Go to **Models & evaluation** to train.")

            st.markdown("**Planned steps (checklist)**")
            st.markdown(
                f"- Missing values: median (numeric), most frequent (categorical)\n"
                f"- Encoding: one-hot for categorical ({len(categorical_cols)} columns)\n"
                f"- Scaling: StandardScaler on numeric features\n"
                f"- Train/test split: {100 * (1 - test_size):.0f}% / {100 * test_size:.0f}%"
            )

    # ----- Tab 4: Train KNN, SVM, ANN and compare metrics (rubric steps 5–6) -----
    with tab_models:
        st.subheader("Model training, testing, and comparison")
        bundle = st.session_state.get("train_bundle")
        if not bundle:
            st.warning("Complete the **Preprocessing** tab first (load data, set target, confirm split).")
        else:
            X_train = bundle["X_train"]
            X_test = bundle["X_test"]
            y_train = bundle["y_train"]
            y_test = bundle["y_test"]
            numeric_cols = bundle["numeric_cols"]
            categorical_cols = bundle["categorical_cols"]
            class_names = bundle["class_names"]
            k_best = bundle["k_best"]

            # Hyperparameters exposed in the UI (tune before training)
            knn_k = st.slider("KNN — number of neighbors", 1, 25, 5)
            svm_c = st.number_input("SVM — C (regularization)", 0.01, 100.0, 1.0)
            ann_h1 = st.number_input("ANN — hidden layer 1 size", 5, 200, 32)
            ann_h2 = st.number_input("ANN — hidden layer 2 size (0 = none)", 0, 200, 16)

            if st.button("Train KNN, SVM, and ANN", type="primary"):
                # ANN architecture: one or two hidden layers
                hidden = [int(ann_h1)]  
                if int(ann_h2) > 0:
                    hidden.append(int(ann_h2))

                # --- KNN: classifies by majority vote among k nearest training points ---
                knn = make_classifier_pipeline(
                    clone(build_preprocessor(numeric_cols, categorical_cols)),
                    k_best,
                    KNeighborsClassifier(n_neighbors=int(knn_k)),
                )
                # --- SVM: finds a decision boundary with maximum margin (RBF kernel) ---
                svm = make_classifier_pipeline(
                    clone(build_preprocessor(numeric_cols, categorical_cols)),
                    k_best,
                    SVC(kernel="rbf", C=float(svm_c)),
                )
                # --- ANN: feed-forward neural network (MLP) with backpropagation ---
                ann = make_classifier_pipeline(
                    clone(build_preprocessor(numeric_cols, categorical_cols)),
                    k_best,
                    MLPClassifier(
                        hidden_layer_sizes=tuple(hidden),
                        max_iter=500,
                        random_state=42,
                        early_stopping=True,
                        validation_fraction=0.1,
                    ),
                )

                labels_sorted = np.sort(np.unique(np.concatenate([y_train.values, y_test.values])))
                try:
                    # Train and evaluate all three models on the same train/test split/ actual training na gyud
                    results = [
                        train_eval_model("KNN", knn, X_train, X_test, y_train, y_test, labels_sorted),
                        train_eval_model("SVM", svm, X_train, X_test, y_train, y_test, labels_sorted),
                        train_eval_model("ANN", ann, X_train, X_test, y_train, y_test, labels_sorted),
                    ]
                except ValueError as err:
                    st.error(
                        f"Training failed ({err}). Try **K = 0** (no feature selection) "
                        "or reduce **K** if it exceeds the number of columns after encoding."
                    )
                    st.stop()
                st.session_state["model_results"] = results
                st.session_state["labels_sorted"] = labels_sorted
                st.session_state["class_names_for_plot"] = class_names

            results = st.session_state.get("model_results")
            if results:
                # Side-by-side accuracy, precision, recall, F1 for KNN vs SVM vs ANN
                st.subheader("Metric comparison")
                comp = pd.DataFrame(
                    [
                        {
                            "Model": r["name"],
                            "Accuracy": r["accuracy"],
                            "Precision (weighted)": r["precision"],
                            "Recall (weighted)": r["recall"],
                            "F1 (weighted)": r["f1"],
                        }
                        for r in results
                    ]
                )
                st.dataframe(
                    comp.style.format(
                        {
                            "Accuracy": "{:.4f}",
                            "Precision (weighted)": "{:.4f}",
                            "Recall (weighted)": "{:.4f}",
                            "F1 (weighted)": "{:.4f}",
                        }
                    ),
                    use_container_width=True,
                )
                best = max(results, key=lambda r: r["f1"])  # pick winner by F1 score
                st.success(
                    f"Best F1 (weighted): **{best['name']}** — use confusion matrices below for detailed errors."
                )

                # One confusion matrix per algorithm
                st.subheader("Confusion matrices")
                cols = st.columns(3)
                ls = np.asarray(st.session_state.get("labels_sorted", []))
                plot_names = st.session_state.get("class_names_for_plot") or class_names
                tick_labels = [
                    plot_names[int(idx)] if int(idx) < len(plot_names) else str(int(idx))
                    for idx in ls
                ]
                for col, r in zip(cols, results):
                    n_cm = r["confusion_matrix"].shape[0]
                    fig = plot_confusion_matrix(
                        r["confusion_matrix"],
                        tick_labels[:n_cm],
                        r["name"],
                    )
                    col.pyplot(fig)
                    plt.close(fig)

                st.subheader("Classification reports (test set)")
                for r in results:
                    with st.expander(f"{r['name']} — full report"):
                        st.text(r["report"])

    # ----- Tab 5: Predict on new loan rows (rubric step 7) -----
    with tab_predict:
        st.subheader("Prediction output")
        results = st.session_state.get("model_results")
        bundle = st.session_state.get("train_bundle")
        if results and bundle:
            model_names = [r["name"] for r in results]
            best = max(results, key=lambda r: r["f1"])
            model_choice = st.selectbox(
                "Model to use",
                model_names,
                index=model_names.index(best["name"]),
            )
            pipe = next(r["model"] for r in results if r["name"] == model_choice)
            class_names = bundle["class_names"]
            is_loan = bundle.get("is_loan", False)

            # Build one applicant row, then pipe.predict() runs full preprocess + model
            if is_loan and bundle.get("classic_loan_form"):
                st.markdown("#### New loan application")
                st.caption("Enter applicant details; the model predicts **Approved** or **Rejected**.")
                c1, c2 = st.columns(2)
                row: dict[str, Any] = {}
                with c1:
                    row["Gender"] = st.selectbox("Gender", bundle["cat_options"].get("Gender", ["Male", "Female"]))
                    row["Married"] = st.selectbox("Married", bundle["cat_options"].get("Married", ["Yes", "No"]))
                    row["Dependents"] = st.selectbox(
                        "Dependents", bundle["cat_options"].get("Dependents", ["0", "1", "2", "3+"])
                    )
                    row["Education"] = st.selectbox(
                        "Education",
                        bundle["cat_options"].get("Education", ["Graduate", "Not Graduate"]),
                    )
                    row["Self_Employed"] = st.selectbox(
                        "Self employed", bundle["cat_options"].get("Self_Employed", ["Yes", "No"])
                    )
                    row["Property_Area"] = st.selectbox(
                        "Property area",
                        bundle["cat_options"].get("Property_Area", ["Urban", "Semiurban", "Rural"]),
                    )
                with c2:
                    row["ApplicantIncome"] = st.number_input(
                        "Applicant income",
                        min_value=0,
                        value=int(bundle["num_defaults"].get("ApplicantIncome", 5000)),
                        step=500,
                    )
                    row["CoapplicantIncome"] = st.number_input(
                        "Co-applicant income",
                        min_value=0,
                        value=int(bundle["num_defaults"].get("CoapplicantIncome", 0)),
                        step=500,
                    )
                    row["LoanAmount"] = st.number_input(
                        "Loan amount (thousands)",
                        min_value=0.0,
                        value=float(bundle["num_defaults"].get("LoanAmount", 120.0)),
                        step=5.0,
                    )
                    row["Loan_Amount_Term"] = st.number_input(
                        "Loan term (months)",
                        min_value=1,
                        value=int(bundle["num_defaults"].get("Loan_Amount_Term", 360)),
                        step=12,
                    )
                    row["Credit_History"] = st.selectbox(
                        "Credit history (1 = good, 0 = poor)",
                        [1.0, 0.0],
                        format_func=lambda x: "Good (1)" if x == 1.0 else "Poor (0)",
                    )
                applicant = pd.DataFrame([row])
            else:
                if is_loan:
                    st.markdown("#### New loan application")
                    st.caption(
                        "Enter feature values from your dataset; prediction is "
                        "**Approved** vs **Rejected**."
                    )
                else:
                    st.markdown("#### Custom input (one row)")
                applicant = pd.DataFrame()
                for col in bundle["numeric_cols"]:
                    applicant[col] = [
                        st.number_input(
                            col,
                            value=float(bundle["num_defaults"].get(col, 0.0)),
                            key=f"pred_num_{col}",
                        )
                    ]
                for col in bundle["categorical_cols"]:
                    opts = bundle["cat_options"].get(col, [])
                    if not opts:
                        st.warning(f"No categories found for **{col}**.")
                        continue
                    applicant[col] = [
                        st.selectbox(col, opts, key=f"pred_cat_{col}")
                    ]

            if st.button("Predict", type="primary"):
                pred_idx = int(pipe.predict(applicant)[0])  # 0/1 index → Approved/Rejected label
                label = (
                    class_names[pred_idx]
                    if pred_idx < len(class_names)
                    else str(pred_idx)
                )
                if is_loan:
                    approved = str(label).upper() in ("Y", "YES", "APPROVED", "1")
                    if approved:
                        st.success(f"**Loan approved** (predicted: {label}) — model: {model_choice}")
                    else:
                        st.error(f"**Loan rejected** (predicted: {label}) — model: {model_choice}")
                else:
                    st.info(f"Predicted class: **{label}** — model: {model_choice}")

            st.divider()
            st.markdown("#### Batch demo (first 10 test rows)")
            X_test = bundle["X_test"]
            sample = X_test.head(10)
            st.dataframe(sample, use_container_width=True)
            if st.button(f"Predict sample with {model_choice}"):
                preds = pipe.predict(sample)
                out = sample.copy()
                out["predicted_class"] = [
                    class_names[int(p)] if int(p) < len(class_names) else int(p)
                    for p in preds
                ]
                st.dataframe(out, use_container_width=True)
        else:
            st.info("Load the **Loan approval** sample (or your own CSV), set target, preprocess, then train in tab 4.")

    st.divider()
    st.caption(
        "Course alignment: IS 108 — Intelligence Systems. "
        "Algorithms: KNN (`KNeighborsClassifier`), SVM (`SVC`), ANN (`MLPClassifier`)."
    )


if __name__ == "__main__":
    main()  # entry point when running: python -m streamlit run app.py
