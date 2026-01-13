import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from command_classifier.common import (
    load_df_final, split_60_20_20, weighted_scores,
    get_paths, CUSTOM_TOKEN_PATTERN
)

def main():
    df_final = load_df_final(prefer_processed=True, save_processed=True)
    X_train, X_val, X_test, y_train, y_val, y_test, w_train, w_val, w_test = split_60_20_20(df_final)

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_val_enc = le.transform(y_val)
    y_test_enc = le.transform(y_test)

    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        lowercase=True,
        max_features=5000,
        token_pattern=CUSTOM_TOKEN_PATTERN,
    )
    Xtr = vec.fit_transform(X_train)
    Xva = vec.transform(X_val)
    Xte = vec.transform(X_test)

    clf = LogisticRegression(
        max_iter=1000,
        solver="saga",
        n_jobs=-1
    )
    clf.fit(Xtr, y_train_enc, sample_weight=w_train)

    pred = clf.predict(Xte)
    acc, f1w = weighted_scores(y_test_enc, pred, w_test)
    print(f"[LR + TF-IDF + custom token pattern] weighted accuracy={acc:.4f}, weighted f1={f1w:.4f}")

    p = get_paths()
    p.models_dir.mkdir(parents=True, exist_ok=True)
    out = p.models_dir / "lr_custom.joblib"
    joblib.dump({"vectorizer": vec, "model": clf, "label_encoder": le}, out)
    print("Saved:", out)

if __name__ == "__main__":
    main()
