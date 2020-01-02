import os

DATA_PATH = os.path.join('../data', "notMNIST_large")

if not os.path.isdir(DATA_PATH):
    raise Exception(f'No data found in path: {DATA_PATH}')