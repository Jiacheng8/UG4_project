import pandas as pd
import numpy as np


def x_y_data_reader(x_data_file, y_data_file, time_lag_dict, label_dict):
    file_key = y_data_file.split("\\")[1].split("_")
    time_lag_value = time_lag_dict.get(file_key[0] + "_" + file_key[1] + "_" + file_key[2])
    time_lag_value *= 1000

    # x_data提取
    ##########################

    frame_data = []

    with open(x_data_file, 'r') as file:
        while 'END_OF_HEADER' not in file.readline():
            pass
        for line in file:
            line_data = [float(item) for item in line.strip().split()[:-1]]
            frame_data.append(line_data)

    x_data = np.array(frame_data)

    frames = x_data.shape[0]

    # y_data提取
    ##########################

    df = pd.read_csv(y_data_file)

    df['start_time'] = df['start_time'] - time_lag_value
    df['end_time'] = df['end_time'] - time_lag_value

    df_clean = df.dropna(subset=['type', 'start_time', 'end_time'])
    y_data_raw = df_clean[['type', 'start_time', 'end_time']].to_dict('records')

    y_data = np.zeros(frames, dtype=int)

    frames_per_second = 100

    for event in y_data_raw:
        event_type = event['type'].lower()

        start_frame = int(event['start_time'] * frames_per_second / 1000)
        end_frame = int(event['end_time'] * frames_per_second / 1000)

        if start_frame >= len(y_data):
            pass
        else:
            end_frame = min(end_frame, len(y_data) - 1)

            y_data[start_frame:end_frame + 1] = label_dict[event_type]

    return x_data, y_data
