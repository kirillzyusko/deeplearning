import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from goto import with_goto

letters = ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J')
TRAIN_DATA_PATH = os.path.join('..', 'data', 'notMNIST_large')
TEST_DATA_PATH = os.path.join('..', 'data', 'notMNIST_small')

if not os.path.isdir(TRAIN_DATA_PATH) or not os.path.isdir(TEST_DATA_PATH):
    raise Exception(f'No data found in path: {TRAIN_DATA_PATH}')


def get_overlaps(images1, images2):
    images1.flags.writeable=False
    images2.flags.writeable=False
    hash1 = set([hash(image1.data) for image1 in images1])
    hash2 = set([hash(image2.data) for image2 in images2])
    all_overlaps = set.intersection(hash1, hash2)
    return all_overlaps


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

    # 5


if __name__ == '__main__':
    main()
