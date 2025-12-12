import pandas as pd
from tensorflow.keras.models import load_model
import numpy as np
from sklearn.metrics import classification_report
from metrics.err_hri_calc import get_metrics


def prepare_seqs(loaded):
    x_sequences = loaded[loaded.files[0]]
    y_sequences = loaded[loaded.files[1]]
    sessions = loaded[loaded.files[2]]
    return x_sequences, y_sequences, sessions


def eval(model_path, data_path, task_name):
    loaded = np.load(data_path, allow_pickle=True)
    print(loaded.files)
    val_x_sequences, _, val_sessions = prepare_seqs(loaded)
    print(val_x_sequences.shape, val_sessions.shape)

    X_val = val_x_sequences

    model = load_model(model_path, compile=False)

    # Print the model summary
    model.summary()
    modality_1_indices = list(range(35))  # 'AU01_r' to 'AU45_c'
    modality_2_indices = list(range(35, 65))  # 'dist_4_7' to 'vel_dist_7_16'
    modality_3_indices = list(range(65, 90))  # 'Loudness_sma3' to 'F3amplitudeLogRelF0_sma3nz'

    preds = model.predict(
        [X_val[:, :, modality_1_indices], X_val[:, :, modality_2_indices], X_val[:, :, modality_3_indices]])
    preds = (preds.squeeze() > 0.5).astype(int)
    print(preds.shape)
    #np.savez(task_name + '_onehot', preds)
    preds_binary = np.argmax(preds, axis=1)
    print(preds_binary.shape)
    #np.savez(task_name + '_binary', preds_binary)
    df = pd.DataFrame({"sessions": val_sessions, task_name: preds_binary})
    df.to_csv(task_name + "_preds.csv", index=False)
    return val_sessions, preds


if __name__ == '__main__':
    model_path = 'weights/multimodal_LSTM_768_RM.keras'
    data_path = 'data/test.npz'
    sessions, preds = eval(model_path, data_path, task_name='RM')
