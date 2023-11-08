import os
import re
import numpy as np
import x_y_data

def get_all_file_paths(directory):
    file_paths = []
    for root, directories, files in os.walk(directory):
        for filename in files:
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)
    return file_paths


def get_data():
    time_lag_dict = {}
    time_lag_file = 'time_lag_calltone_mocap.txt'
    with open(time_lag_file, 'r') as file:
        for line in file:
            parts = re.split(r'\s+', line.strip(), maxsplit=1)
            if len(parts) == 2:
                key, value = parts
                time_lag_dict[key.lower()] = float(value)
            else:
                pass

    label_dict = {
        'fd': 1,
        'ti': 2,
        't': 3,
        'fu': 4,
        'sh': 5,
        'start': 6,
        'nd': 7,
        'mnd': 8
    }

    x_train_list = get_all_file_paths("x_train")
    y_train_list = get_all_file_paths("y_train")

    sequences_list_x = []
    sequences_list_y = []

    largest_seq_len = 0

    for index, x_data_file in enumerate(x_train_list):
        x_data, y_data = x_y_data.x_y_data_reader(x_data_file, y_train_list[index], time_lag_dict, label_dict)
        sequences_list_x.append(x_data)
        sequences_list_y.append(y_data)
        frames = x_data.shape[0]
        largest_seq_len = max(largest_seq_len, frames)

    padded_sequences_x = [np.pad(x, ((0, largest_seq_len - x.shape[0]), (0, 0)), mode='constant') for x in
                          sequences_list_x]
    x_data_batch = np.stack(padded_sequences_x)

    padded_sequences_y = [np.pad(y, (0, largest_seq_len - y.shape[0]), mode='constant') for y in sequences_list_y]
    y_data_batch = np.stack(padded_sequences_y)
    return x_data_batch, y_data_batch

if __name__ == '__main__':
    x,y = get_data()
    print(x.shape, y.shape)