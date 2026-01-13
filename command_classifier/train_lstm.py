import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from command_classifier.common import load_df_final, split_60_20_20, get_paths

def main():
    df_final = load_df_final(prefer_processed=True, save_processed=True)
    X_train, X_val, X_test, y_train, y_val, y_test, w_train, w_val, w_test = split_60_20_20(df_final)

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_val_enc = le.transform(y_val)
    y_test_enc = le.transform(y_test)

    num_classes = len(le.classes_)
    y_train_oh = to_categorical(y_train_enc, num_classes=num_classes)
    y_val_oh = to_categorical(y_val_enc, num_classes=num_classes)

    tokenizer = Tokenizer(num_words=5000, oov_token="<unk>", lower=True)
    tokenizer.fit_on_texts(X_train)

    max_len = 20
    X_train_pad = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=max_len, padding="post", truncating="post")
    X_val_pad   = pad_sequences(tokenizer.texts_to_sequences(X_val),   maxlen=max_len, padding="post", truncating="post")
    X_test_pad  = pad_sequences(tokenizer.texts_to_sequences(X_test),  maxlen=max_len, padding="post", truncating="post")

    vocab_size = len(tokenizer.word_index) + 1
    model = Sequential([
        Embedding(vocab_size, 100, input_length=max_len),
        LSTM(64),
        Dropout(0.5),
        Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    model.fit(
        X_train_pad, y_train_oh,
        sample_weight=w_train.to_numpy(),
        epochs=10,
        batch_size=32,
        validation_data=(X_val_pad, y_val_oh, w_val.to_numpy()),
        callbacks=[EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)]
    )

    proba = model.predict(X_test_pad)
    y_pred = proba.argmax(axis=1)

    acc = accuracy_score(y_test_enc, y_pred, sample_weight=w_test)
    f1w = f1_score(y_test_enc, y_pred, average="weighted", sample_weight=w_test)
    print(f"[LSTM] weighted accuracy={acc:.4f}, weighted f1={f1w:.4f}")

    p = get_paths()
    p.models_dir.mkdir(parents=True, exist_ok=True)
    out = p.models_dir / "lstm.keras"
    model.save(out)
    print("Saved:", out)

if __name__ == "__main__":
    main()
