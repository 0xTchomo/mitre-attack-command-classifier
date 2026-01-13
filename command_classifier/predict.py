import joblib
from command_classifier.common import get_paths

def main():
    p = get_paths()
    model_path = p.models_dir / "lr_custom.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Train first: python -m command_classifier.train_lr")

    bundle = joblib.load(model_path)
    vec = bundle["vectorizer"]
    clf = bundle["model"]
    le = bundle["label_encoder"]

    while True:
        cmd = input("Enter a command (or 'quit'): ").strip()
        if cmd.lower() in {"quit", "exit"}:
            break

        X = vec.transform([cmd])
        pred = clf.predict(X)[0]
        label = le.inverse_transform([pred])[0]
        print("Predicted technique_grouped:", label)

if __name__ == "__main__":
    main()
