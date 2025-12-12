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


def create_sequences_no_labels(data, sessions, sequence_length):
    sequences = []
    session_ids = []

    # Split data by session and then create sequences
    unique_sessions = np.unique(sessions)
    for session in unique_sessions:
        session_indices = np.where(sessions == session)[0]
        session_data = data[session_indices]

        if len(session_data) >= sequence_length:
            for i in range(len(session_data) - sequence_length + 1):
                sequences.append(session_data[i: i + sequence_length])
                session_ids.append(session)

    return np.array(sequences), np.array(session_ids)


def createDataSplits(df, fold_no, with_val=0, label_column=None, fold_column='fold_id',
                     results_directory='../logs/', seed_value=42, sequence_length=1):
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
    if label_column is None:
        raise ValueError
    print(df.columns)
    print(len(df.columns))
    try:

        # # Set seed
        random.seed(seed_value)
        np.random.seed(seed_value)

        # get features and target class
        features = df.copy().iloc[:, 3:93]
        print(features.columns)
        target_class = df[label_column].values
        target_class = target_class.astype('int')
        sessions = df['session'].values

        # get number of classes
        num_classes = len(np.unique(target_class))
        # if fold number is none
        if fold_no is None:
            train_fold = df[fold_column].unique()
            print('folds:', train_fold)
            train_indices = df[df[fold_column].isin(train_fold)].index
            X_train = features.loc[train_indices]
            y_train = target_class[train_indices]
            session_train = sessions[train_indices]
            X_train = X_train.reset_index(drop=True)
            X_train_sequences, y_train_sequences = create_sequences(X_train.values, y_train, session_train,
                                                                    sequence_length)
            X_val = None
            y_val = None
            X_val_sequences = None
            y_val_sequences = None
            X_test = None
            y_test = None
            X_test_sequences = None
            y_test_sequences = None


        else:
            num_folds = df[fold_column].unique()
            fold_sessions = df[df[fold_column] == fold_no]['session'].unique()

            if with_val == 1:
                val_fold = fold_no
                # test fold should be the next fold, but if fold is max(num_folds), then test fold is 1
                if fold_no == np.max(num_folds):
                    test_fold = 1
                else:
                    test_fold = fold_no + 1
                train_fold = [f for f in num_folds if f not in [val_fold, test_fold]]


            else:
                test_fold = fold_no
                train_fold = [f for f in num_folds if f != test_fold]
                val_fold = None
            print('folds:', train_fold, val_fold, test_fold)

            # Split the data into train, validation, and test sets
            train_indices = df[df[fold_column].isin(train_fold)].index
            if with_val == 1:
                val_indices = df[df[fold_column] == val_fold].index
            test_indices = df[df[fold_column] == test_fold].index

            X_train = features.loc[train_indices]
            y_train = target_class[train_indices]
            session_train = sessions[train_indices]
            if with_val == 1:
                X_val = features.loc[val_indices]
                y_val = target_class[val_indices]
                session_val = sessions[val_indices]

            X_test = features.loc[test_indices]
            y_test = target_class[test_indices]
            session_test = sessions[test_indices]

            # print size of all sets
            print(X_train.shape, y_train.shape)
            if with_val == 1:
                print(X_val.shape, y_val.shape)
            print(X_test.shape, y_test.shape)

            # reset indexes
            X_train = X_train.reset_index(drop=True)
            if with_val == 1:
                X_val = X_val.reset_index(drop=True)
            X_test = X_test.reset_index(drop=True)

            # Create sequences for LSTM
            X_train_sequences, y_train_sequences = create_sequences(X_train.values, y_train, session_train,
                                                                    sequence_length)
            if with_val == 1:
                X_val_sequences, y_val_sequences = create_sequences(X_val.values, y_val, session_val, sequence_length)
            else:
                X_val_sequences = None
                y_val_sequences = None
                X_val = None
                y_val = None

            X_test_sequences, y_test_sequences = create_sequences(X_test.values, y_test, session_test, sequence_length)

        print('here', X_train_sequences.shape, session_train.shape)
        return num_classes, X_train, X_val, X_test, y_train, y_val, y_test, X_train_sequences, y_train_sequences, X_val_sequences, y_val_sequences, X_test_sequences, y_test_sequences

    except Exception as e:
        print(f"An error occurred: {e}")


def createDataSplits_test_no_labels(df, seed_value=42,
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
        print(df.columns)
        # get features only
        features = df.iloc[:, 3:]

        sessions = df['session'].values

        X_test = features

        X_test = X_test.reset_index(drop=True)

        # Create sequences for LSTM
        print(X_test.shape, sessions.shape)
        X_test_sequences, test_sessions = create_sequences_no_labels(X_test.values, sessions, sequence_length)
        print(X_test_sequences.shape, test_sessions.shape)

        return test_sessions, X_test, X_test_sequences

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == '__main__':
    df = pandas.read_csv('data/train_val.csv')
    print(df.columns)

    label_column = 'UserAwkwardness'
    (num_classes, X_train, X_val, X_test, y_train, y_val, y_test, X_train_sequences,
     y_train_sequences, X_val_sequences, y_val_sequences,
     X_test_sequences, y_test_sequences) = createDataSplits(df, fold_no=1, with_val=0, sequence_length=5,
                                                            label_column=label_column)
    print(num_classes)
    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)
    print(X_train_sequences.shape, y_test_sequences.shape)
    print()
    exit()
    np.savez_compressed('data/train_org_split_' + label_column, X_train_sequences=X_train_sequences,
                        y_train_sequences=y_train_sequences, train_sessions=None,
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
    np.savez_compressed('data/val_org_split_' + label_column, X_val_sequences=X_test_sequences,
                        y_val_sequences=y_test_sequences, train_sessions=None,
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
