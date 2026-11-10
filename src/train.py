"""Train the final vehicle classifier and save it to models/.

Pipeline: median imputation -> Yeo-Johnson PowerTransformer -> SVM (RBF).
The model is first evaluated on the same stratified 80/20 split as in the
notebook. It is then refit on all data and saved for production use.

Usage (from the project root):
    python src/train.py
"""
from pathlib import Path

import joblib
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer
from sklearn.svm import SVC
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "vehicle.csv"
MODEL = ROOT / "models" / "svm_pipeline.joblib"
RANDOM_STATE = 42


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("power", PowerTransformer(method="yeo-johnson", standardize=True)),
        ("clf", SVC(C=10, gamma="scale", probability=True, random_state=RANDOM_STATE)),
    ])


def main() -> None:
    df = pd.read_csv(DATA)
    X, y = df.drop(columns="class"), df["class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(build_pipeline(), X_train, y_train, cv=cv)
    model = build_pipeline().fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"5-fold CV accuracy (train): {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")
    print(f"Test accuracy:              {accuracy_score(y_test, y_pred):.3f}\n")
    print(classification_report(y_test, y_pred, digits=3))

    # Refit on all labelled data for production
    final = build_pipeline().fit(X, y)
    MODEL.parent.mkdir(exist_ok=True)
    joblib.dump({"model": final, "features": X.columns.tolist()}, MODEL)
    print(f"Saved model trained on {len(X)} samples -> {MODEL.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
