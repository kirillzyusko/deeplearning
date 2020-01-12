import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from goto import with_goto
import imageio
import numpy as np

letters = ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J')
TRAIN_DATA_PATH = os.path.join('..', 'data', 'notMNIST_large')
TEST_DATA_PATH = os.path.join('..', 'data', 'notMNIST_small')

if not os.path.isdir(TRAIN_DATA_PATH) or not os.path.isdir(TEST_DATA_PATH):
    raise Exception(f'No data found in path: {TRAIN_DATA_PATH}')

def to_tuple(x):
    return tuple(tuple(tuple(l2) for l2 in l1) for l1 in x)

def get_overlaps(X1, X2, y1, y2):
    x_1 = to_tuple(X1)
    x_2 = to_tuple(X2)
    dict1 = {}
    dict2 = {}
    set1 = set()
    set2 = set()
    for i in range(len(x_1)):
        dict1[x_1[i]] = y1[i]
    for i in range(len(x_2)):
        dict2[x_2[i]] = y2[i]
    del x_1
    del x_2
    set1.update(dict1)
    set2.update(dict2)
    diff = set.difference(set1, set2)
    del set1
    del set2
    del dict2
    x = []
    y = []
    for i in diff:
        x.append(i)
        y.append(dict1[i])
    return x, y


def get_dataset(dataset_path):
    dataset = {}
    for i in letters:
        dataset[i] = []
        path = os.path.join(dataset_path, i)
        for file in os.listdir(path):
            dataset[i].append(file)

    return dataset


def cut_dataset(dataset, from_range, to_range):
    dataset_copy = {}
    for letter in letters:
        dataset_copy[letter] = dataset[letter][from_range:to_range]

    return dataset_copy


def transform_to_array(dataset):
    X, y = [], []
    for letter in letters:
        for file in dataset[letter]:
            X.append(file)
            y.append(letter)

    return X, y


def load_images(X, y, source):
    x_res, y_res = [], []
    for i in range(len(X)):
        path = os.path.join(source, y[i], X[i])
        try:
            image_data = imageio.imread(path)
            x_res.append(image_data)
            y_res.append(y[i])
        except:
            print(f'Bad file: {path}')

    return x_res, y_res


@with_goto
def main():
    goto .task
    # 1
    fig=plt.figure(figsize=(8, 8))
    columns = 5
    rows = 2
    for i in range(1, columns*rows + 1):
        path = os.path.join(TRAIN_DATA_PATH, letters[i - 1], 'aG9tZXdvcmsgbm9ybWFsLnR0Zg==.png')
        img = mpimg.imread(path)
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)
    plt.show()

    # 2
    fig, ax = plt.subplots()
    density = []
    for i in letters:
        path = os.path.join(TRAIN_DATA_PATH, i)
        density.append(len(os.listdir(path)))

    print(density)

    ax.bar(letters, density, width=0.5)
    ax.set_ylabel('Amount of files')
    ax.set_title('Files in each group')
    plt.show()

    # 3
    label .task
    X, y = transform_to_array(cut_dataset(get_dataset(TRAIN_DATA_PATH), 0, 21000))
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.047619, random_state=1)
    X_test, y_test = transform_to_array(get_dataset(TEST_DATA_PATH))

    # 4
    print('Files info fetched...')
    X_train, y_train = load_images(X_train, y_train, TRAIN_DATA_PATH)
    X_val, y_val = load_images(X_val, y_val, TRAIN_DATA_PATH)
    X_test, y_test = load_images(X_test, y_test, TEST_DATA_PATH)
    print('Images loaded into memory...')
    print(len(X_train)) # 200.000
    X_without_val, y_without_val = get_overlaps(X_train, X_val, y_train, y_val)
    print(len(X_without_val)) # 174.808
    X_without_val_test, y_without_val_test = get_overlaps(X_without_val, X_test, y_without_val, y_test)
    print(len(X_without_val_test)) # 171.395
    np.save('X_train', np.array(X_without_val_test))
    np.save('y_train', np.array(y_without_val_test))
    np.save('X_val', np.array(X_val))
    np.save('y_val', np.array(y_val))
    np.save('X_test', np.array(X_test))
    np.save('y_test', np.array(y_test))

    # 5



if __name__ == '__main__':
    main()
