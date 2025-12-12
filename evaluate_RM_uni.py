from tensorflow.keras.models import load_model
import numpy as np
from sklearn.metrics import classification_report
from metrics.err_hri_calc import get_metrics


def prepare_seqs(loaded):
    x_sequences = loaded[loaded.files[0]]
    y_sequences = loaded[loaded.files[1]]
    sessions = loaded[loaded.files[2]]
    return x_sequences, y_sequences, sessions


def eval(model_path, data_path):
    loaded = np.load(data_path, allow_pickle=True)
    val_x_sequences, val_y_sequences, val_sessions = prepare_seqs(loaded)
    print(val_x_sequences.shape)

    X_val = val_x_sequences
    y_val = val_y_sequences

    model = load_model(model_path, compile=False)

    # Print the model summary
    model.summary()
    preds = model.predict(X_val)
    preds = (preds.squeeze() > 0.5).astype(int)
    print(preds.shape, y_val.shape)
    print(get_metrics(preds, y_val))
    print(classification_report(y_val, preds))
    return preds


if __name__ == '__main__':
    model_path = 'models/unimodal_LSTM_768_RM.keras'
    data_path = 'data/val_org_split_RobotMistake.npz'
    preds = eval(model_path, data_path)
