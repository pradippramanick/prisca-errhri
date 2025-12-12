import os
import pandas as pd

def preprocess(fold_set, label_path, data_path,fold_name='train_val'):
    print(fold_set.head())
    # list files in path
    files = os.listdir(label_path)
    # remove hidden files
    files = [file for file in files if not file.startswith('.')]
    print('Number of sessions:', len(files))
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
    print('Number of sessions:', len(files))
    if os.path.exists('data/' + fold_name + '.csv'):
        train_val = pd.read_csv('data/' + fold_name + '.csv')
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
                # see difference in index numbers
                # index_session = session_csv['frame'].values
                # index_open = open_csv['frame_id'].values
                # print('openpose:', index_session, index_open)
                # see if they are the same, if not print
                # if not np.array_equal(index_session, index_open):
                #    print('Different indexes')
                #    #which are different
                #    diff = np.where(index_session != index_open)
                #    print(diff)

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

        # add a column with fold_id to the session_csv
        fold_id = fold_set[fold_set['id'] == session]['fold-subject-independent'].values[0]
        session_csv['fold_id'] = fold_id
        print('fold_id:', fold_id)

        # append the session_csv to the train_val
        train_val = pd.concat([train_val, session_csv], axis=0)
        print('train_val:', train_val.shape)
        print('train_val columns:', train_val.columns)
        train_val.reset_index(drop=True, inplace=True)
        # save the train_val
        train_val.to_csv('data/' + fold_name + '.csv', index=False)

        print('DIFF SESSION:', diff_session)

    train_val.reset_index(drop=True, inplace=True)
    print(train_val.shape)
    # save the train_val
    train_val.to_csv('data/' + fold_name + '.csv', index=False)
    print(train_val.shape)
    train_val = pd.read_csv('data/' + fold_name + '.csv')
    # remove time column
    train_val.drop(columns='timestamp', inplace=True)
    # make fold_id the first column
    cols = train_val.columns.tolist()
    cols = cols[-2:] + cols[:-2]
    train_val = train_val[cols]
    print(train_val.columns)
    # make the 3 label columns zeros
    train_val['UserAwkwardness'] = 0
    train_val['RobotMistake'] = 0
    train_val['InteractionRupture'] = 0

    # list files in path
    files = os.listdir(label_path)
    # remove hidden files
    files = [file for file in files if not file.startswith('.')]
    print('Number of sessions:', len(files))
    label_column_0 = []
    label_column_1 = []
    label_column_2 = []
    for file in files:
        # open the csv
        label_df = pd.read_csv(label_path + file)
        # get the session number
        session = file.split('.')[0]
        if session in train_val['session'].values:
            # get the label
            train_val_session = train_val[train_val['session'] == session]
        else:
            print('Session NOT in train_val:', session)
            continue

        # for each row in train_val_session, get the time, and match it to the time in the label dataset
        # if the time is within the interval, get the label
        for ind, row in label_df.iterrows():
            time_min = row['Begin Time - ss.msec']
            time_max = row['End Time - ss.msec']
            # get labels for each
            lab_uawk = row['UserAwkwardness']
            lab_rmist = row['RobotMistake']
            lab_irupt = row['InteractionRupture']
            # get the rows in train_val_session that are within the interval
            train_val_interval = train_val_session[
                (train_val_session['time'] >= time_min) & (train_val_session['time'] <= time_max)]
            # get the indexes
            index_interval = train_val_interval.index
            print('lenght of train_val_interval:', len(train_val_interval))
            print('index_interval:', index_interval)
            if len(train_val_interval) == 0:
                print('empty interval')
                print('time_min:', time_min)
                print('time_max:', time_max)
                print('session:', session)
            else:
                if lab_uawk == 1:
                    train_val.loc[index_interval, 'UserAwkwardness'] = lab_uawk
                if lab_rmist == 1:
                    train_val.loc[index_interval, 'RobotMistake'] = lab_rmist
                if lab_irupt == 1:
                    train_val.loc[index_interval, 'InteractionRupture'] = lab_irupt
    train_val.to_csv('data/' + fold_name + '.csv', index=False)
    return train_val


if __name__ == '__main__':
    data_path = '/home/pradip/Desktop/ERR@HRI/TransformerBaseline/org_baseline/data/ERR@HRI dataset - test/'
    fold_set = pd.read_csv(data_path + 'fold_split.csv')
    label_path = data_path + 'labels/'
    train_val = preprocess(fold_set, label_path, data_path)
