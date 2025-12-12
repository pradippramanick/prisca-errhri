import os

import numpy as np
import pandas as pd
import warnings

from create_splits import createDataSplits_test_no_labels

#ignore warnings
warnings.filterwarnings('ignore')


def preprocess_test(data_path):
    # open each feature folder, get the csvs into a single dataframe, with the session number as a column
    feature_folders = ['openface', 'openpose', 'opensmile']

    for folder in feature_folders:
        print('Processing', folder)
        # list files in path
        files = os.listdir(data_path + folder)
        # remove hidden files
        files = [file for file in files if not file.startswith('.')]
        columns = []
        for file in files:
            # open the first file to get the column names
            df = pd.read_csv(data_path + folder + '/' + file)
            # add column with session number
            session = file.split('.')[0]
            if len(columns) == 0:
                columns = df.columns
                data = df
            else:
                cols = df.columns
                if not all(elem in columns for elem in cols):
                    print('Columns do not match')
                    print('Columns in', folder, 'not in data:', [elem for elem in cols if elem not in columns])
                    print(session)
            # now, check for completely empty columns
            empty_cols = df.columns[df.isnull().all()]
            if len(empty_cols) > 0:
                print('Empty columns in', folder, ':', empty_cols)
                print(session)

    # open each feature folder, get the csvs into a single dataframe, with the session number as a column
    feature_folders = ['openface', 'openpose', 'opensmile']

    # initialize the dataframe
    # check if the file exists
    if os.path.exists('data/test.csv'):
        train_val = pd.read_csv('data/test.csv')
        print('train_val:', train_val.shape)
        sessions_already = train_val['session'].unique()
    else:
        train_val = pd.DataFrame()
        sessions_already = []

    # save session names when there is a difference for time and frames
    diff_session = dict()

    # get session names
    files_folders = os.listdir(data_path + feature_folders[0])
    sessions = [file.split('.')[0] for file in files_folders if not file.startswith('.')]
    for session in sessions:
        print('session:', session)
        if session in sessions_already:
            print('session already in train_val')
            continue
        for folder in feature_folders:

            # if folder openface
            if folder == 'openface':
                # get the csv
                session_csv = pd.read_csv(data_path + folder + '/' + session + '.csv')
                # if empty, skip this session
                if session_csv.empty:
                    print('empty openface')
                    continue

                # remove rows with nan values
                print('openface prenan:', session_csv.shape)
                session_csv.dropna(inplace=True)
                print('openface postnan:', session_csv.shape)
                # add session column as the first column

                # decrease the frame by one
                session_csv['frame'] = session_csv['frame'] - 1
                session_csv['session'] = session
                # change session from last to first column
                cols = session_csv.columns.tolist()
                cols = cols[-1:] + cols[:-1]
                session_csv = session_csv[cols]
                # change [' timestamp'] to timestamp
                session_csv.rename(columns={' timestamp': 'timestamp'}, inplace=True)
                print('openface:', session_csv.shape)
                # print(session_csv.columns)
                # print(session_csv.head())

            # if folder openpose
            if folder == 'openpose':
                # get the csv
                open_csv = pd.read_csv(data_path + folder + '/' + session + '.csv')
                # if empty, skip this session
                if open_csv.empty:
                    print('empty openpose')
                    continue
                # reduce one in frame_id
                open_csv['frame_id'] = open_csv['frame_id']
                # remove columns ['person_id', 'week_id', 'robot_group'] if existing
                if 'person_id' in open_csv.columns:
                    open_csv.drop(columns=['person_id', 'week_id', 'robot_group'], inplace=True)

                # remove columns ['vel_1_x', 'vel_1_y', 'vel_8_x', 'vel_8_y', 'dist_1_8', 'vel_dist_1_8', 'dist_7_0', 'dist_4_0', 'vel_7_x', 'vel_7_y', 'vel_4_x', 'vel_4_y','vel_dist_7_0', 'vel_dist_4_0']
                if 'vel_1_x' in open_csv.columns:
                    open_csv.drop(columns=['vel_1_x', 'vel_1_y', 'vel_8_x', 'vel_8_y', 'dist_1_8', 'vel_dist_1_8',
                                           'dist_7_0', 'dist_4_0', 'vel_7_x', 'vel_7_y', 'vel_4_x', 'vel_4_y',
                                           'vel_dist_7_0', 'vel_dist_4_0'], inplace=True)

                # remove rows with nan values
                print('openpose prenan:', open_csv.shape)
                open_csv.dropna(inplace=True)
                print('openpose postnan:', open_csv.shape)

                # merge horizontally with the session_csv, through column "frame_id" and "frame" in open_csv and session_csv, respectively
                session_csv = pd.merge(session_csv, open_csv, how='inner', left_on='frame', right_on='frame_id')
                # drop the frame_id column
                session_csv.drop(columns='frame_id', inplace=True)

                print('openpose prenan:', session_csv.shape)
                session_csv.dropna(inplace=True)
                print('openpose postnan:', session_csv.shape)

                print('openpose:', session_csv.shape)
                # print(session_csv.columns)
                # print(session_csv.head())

            # if folder opensmile
            if folder == 'opensmile':
                # get the csv
                smile_csv = pd.read_csv(data_path + folder + '/' + session + '.csv')
                if smile_csv.empty:
                    print('empty opensmile')
                    continue

                # now, open the corresponding speaker_diarization file
                sd_path = data_path + 'speaker_diarization/'
                sd_csv = pd.read_csv(sd_path + session + '.csv')
                # if empty, skip this session
                if sd_csv.empty:
                    print('empty speaker diarization')
                    print('***************************************************************************')
                    continue

                print('opensmile prenan:', smile_csv.shape)
                smile_csv.dropna(inplace=True)
                print('opensmile postnan:', smile_csv.shape)

                # drop column "file"
                if 'Unnamed: 0' in smile_csv.columns:
                    smile_csv.drop(columns='Unnamed: 0', inplace=True)

                # time is as "0 days 00:00:02.510000"
                # turn this into only seconds - 2.51
                # first, turn into time instead of string
                smile_csv['start'] = pd.to_timedelta(smile_csv['start'])
                # print(smile_csv[['start']].head())
                smile_csv['time'] = smile_csv['start'].apply(lambda x: x.total_seconds())

                # print(smile_csv['time'])
                # print(session_csv.columns)
                subset_smile = pd.DataFrame()
                # go row by row in session_csv, and look at timestamp. use the timestamp column as a reference to get the opensmile features, and get the average of the features in opensmile within the interval
                prev_time = 0
                for ind, row in session_csv.iterrows():
                    # get the timestamp
                    timestamp = row['frame'] / 30  # in seconds, for 30 fps
                    # if timestamp is 0, then avg_features is the first row of smile_csv
                    if timestamp == 0:
                        avg_features = smile_csv.iloc[0]
                        avg_features['time'] = timestamp
                        # drop start and end columns
                        avg_features.drop(['start', 'end'], inplace=True)
                        subset_smile = pd.concat([subset_smile, avg_features], axis=1)
                        prev_time = 0
                        continue

                    # get the opensmile features that are in the interval
                    interval = smile_csv[(smile_csv['time'] > prev_time) & (smile_csv['time'] <= timestamp)]

                    # now, check who was speaking. Column "speaker" in sd_csv is robot, person or pause. time is in seconds, and there are two columns, start_turn and end_turn
                    # if the timestamp is within the interval of a speaker, then keep the interval, otherwise, zero out the features
                    speaker = sd_csv[(sd_csv['start_turn'] <= timestamp) & (sd_csv['end_turn'] > timestamp)]['speaker']
                    if speaker.empty:
                        speaker = pd.DataFrame(['pause'])
                    #    print(timestamp, 'empty speaker')

                    # if empty, print warning
                    if interval.empty:
                        if timestamp > smile_csv['time'].max():
                            print('timestamp is bigger than max time')
                            print('timestamp max:', session_csv['frame'].max() / 30, 'max time opensmile:',
                                  smile_csv['time'].max())
                            diff_session[session] = (session_csv['frame'].max() / 30, smile_csv['time'].max())
                        else:
                            print('empty interval')
                        # skip rest of the loop
                        break

                    # print(interval)
                    interval['time'] = timestamp
                    interval['frame'] = row['frame']
                    # remove start and end columns
                    interval.drop(columns=['start', 'end'], inplace=True)
                    # get the average of the features
                    if speaker.values[0] == 'participant':
                        avg_features = interval.mean()
                    else:
                        avg_features = interval.mean()
                        avg_features[:] = 0

                    avg_features['time'] = timestamp
                    avg_features['frame'] = row['frame']
                    # avg_features['speaker'] = speaker.values[0]
                    # print(speaker.values[0])
                    # print(avg_features)
                    # append the features to the subset_smile
                    subset_smile = pd.concat([subset_smile, avg_features], axis=1)
                    # print(subset_smile.shape)
                    # update the prev_time
                    prev_time = timestamp

                print('done')
                # transpose the subset_smile
                subset_smile = subset_smile.T
                # reindex
                subset_smile.reset_index(drop=True, inplace=True)
                print(subset_smile.shape)

                # print(subset_smile.columns)
                # print(subset_smile)
                # merge horizontally with the session_csv
                session_csv = pd.merge(session_csv, subset_smile, how='inner', left_on='frame', right_on='frame')

                print('opensmile:', session_csv.shape)
                # print(session_csv.head())

        # append the session_csv to the train_val
        train_val = pd.concat([train_val, session_csv], axis=0)
        print('train_val:', train_val.shape)
        print('train_val columns:', train_val.columns)
        train_val.reset_index(drop=True, inplace=True)
        # save the train_val
        train_val.to_csv('data/test.csv', index=False)

        print('DIFF SESSION:', diff_session)

    train_val.reset_index(drop=True, inplace=True)
    print(train_val.shape)
    train_val.drop(columns='timestamp', inplace=True)
    # make fold_id the first column
    cols = train_val.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    train_val = train_val[cols]
    print(train_val.columns)
    # save
    train_val.to_csv('data/test.csv', index=False)
    return train_val


def serialize(test_df):
    # Serialize
    sessions, X_test, X_test_sequences = (
        createDataSplits_test_no_labels(test_df, sequence_length=5))
    print(X_test)
    print(X_test_sequences.shape)  # Should be (241257, 5, 90)
    print(sessions.shape)
    return sessions, X_test_sequences


if __name__ == '__main__':
    data_path = '/home/pradip/Desktop/ERR@HRI/TransformerBaseline/org_baseline/data/ERR@HRI dataset - test/'
    #preprocess_test(data_path)
    test_df = pd.read_csv('data/test.csv')
    feature_cols = [' AU01_r', ' AU01_c', ' AU02_r', ' AU04_r', ' AU05_r', ' AU06_r',
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
                    'F3amplitudeLogRelF0_sma3nz']
    assert feature_cols == test_df.columns.tolist()[3:]
    test_sessions, X_test_sequences = serialize(test_df)
    np.savez_compressed('data/test', X_test_sequences=X_test_sequences,
                        y_train_sequences=None, test_sessions=test_sessions,
                        )
