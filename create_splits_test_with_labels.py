import numpy as np
import pandas
import pandas as pd
import random
import warnings

warnings.filterwarnings('ignore')


def create_sequences(data, target, sessions, sequence_length):
    sequences = []
    targets = []

    # Split data by session and then create sequences
    unique_sessions = np.unique(sessions)
    for session in unique_sessions:
        session_indices = np.where(sessions == session)[0]
        session_data = data[session_indices]
        session_target = target[session_indices]

        if len(session_data) >= sequence_length:
            for i in range(len(session_data) - sequence_length + 1):
                sequences.append(session_data[i: i + sequence_length])
                targets.append(session_target[i + sequence_length - 1])

    # hot one encode the target
    targets = pd.get_dummies(targets).values

    return np.array(sequences), np.array(targets)


def createDataSplits_test(df, label_column='UserAkwardness', results_directory='../logs/', seed_value=42,
                          sequence_length=1):
    """
    Split the data into train, validation, and test sets for each fold.
    df - dataframe containing the data
    fold_no - number of fold to get the data for
    with_val - whether to include validation set or not
    label_column - column name for the target class
    fold_column - column name for the fold number
    results_directory - directory to store the results
    seed_value - seed value for reproducibility
    sequence_length - length of the sequence for RNNs or transformers

    """

    try:

        # # Set seed
        random.seed(seed_value)
        np.random.seed(seed_value)

        # get features and target class
        features = df.iloc[:, 6:]
        target_class = df[label_column].values
        target_class = target_class.astype('int')
        sessions = df['session'].values

        X_test = features
        y_test = target_class

        print(X_test.shape, y_test.shape)

        X_test = X_test.reset_index(drop=True)

        # Create sequences for LSTM

        X_test_sequences, y_test_sequences = create_sequences(X_test.values, y_test, sessions, sequence_length)

        return sessions, X_test, y_test, X_test_sequences, y_test_sequences

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == '__main__':
    df = pandas.read_csv('data/test.csv')
    label_column = 'UserAwkwardness'
    sessions, X_test, y_test, X_test_sequences, y_test_sequences = createDataSplits_test(df,
                                                                                         sequence_length=5,
                                                                                         label_column=label_column)

    np.savez_compressed('data/test_with_labels_' + label_column, X_test_sequences=X_test_sequences,
                        y_test_sequences=y_test_sequences, test_sessions=sessions,
                        feature_cols=[' AU01_r', ' AU01_c', ' AU02_r', ' AU04_r', ' AU05_r', ' AU06_r',
                                      ' AU07_r', ' AU09_r', ' AU10_r', ' AU12_r', ' AU14_r', ' AU15_r',
                                      ' AU17_r', ' AU20_r', ' AU23_r', ' AU25_r', ' AU26_r', ' AU45_r',
                                      ' AU02_c', ' AU04_c', ' AU05_c', ' AU06_c', ' AU07_c', ' AU09_c',
                                      ' AU10_c', ' AU12_c', ' AU14_c', ' AU15_c', ' AU17_c', ' AU20_c',
                                      ' AU23_c', ' AU25_c', ' AU26_c', ' AU28_c', ' AU45_c', 'dist_4_7',
                                      'vel_dist_4_7', 'dist_4_2', 'vel_dist_4_2', 'dist_4_5', 'vel_dist_4_5',
                                      'dist_4_1', 'vel_dist_4_1', 'dist_4_17', 'vel_dist_4_17', 'dist_4_15',
                                      'vel_dist_4_15', 'dist_4_18', 'vel_dist_4_18', 'dist_4_16',
                                      'vel_dist_4_16', 'dist_7_2', 'vel_dist_7_2', 'dist_7_5', 'vel_dist_7_5',
                                      'dist_7_1', 'vel_dist_7_1', 'dist_7_17', 'vel_dist_7_17', 'dist_7_15',
                                      'vel_dist_7_15', 'dist_7_18', 'vel_dist_7_18', 'dist_7_16',
                                      'vel_dist_7_16', 'Loudness_sma3', 'alphaRatio_sma3',
                                      'hammarbergIndex_sma3', 'slope0-500_sma3', 'slope500-1500_sma3',
                                      'spectralFlux_sma3', 'mfcc1_sma3', 'mfcc2_sma3', 'mfcc3_sma3',
                                      'mfcc4_sma3', 'F0semitoneFrom27.5Hz_sma3nz', 'jitterLocal_sma3nz',
                                      'shimmerLocaldB_sma3nz', 'HNRdBACF_sma3nz', 'logRelF0-H1-H2_sma3nz',
                                      'logRelF0-H1-A3_sma3nz', 'F1frequency_sma3nz', 'F1bandwidth_sma3nz',
                                      'F1amplitudeLogRelF0_sma3nz', 'F2frequency_sma3nz',
                                      'F2bandwidth_sma3nz', 'F2amplitudeLogRelF0_sma3nz',
                                      'F3frequency_sma3nz', 'F3bandwidth_sma3nz',
                                      'F3amplitudeLogRelF0_sma3nz'])
