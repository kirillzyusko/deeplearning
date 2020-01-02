import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from goto import with_goto

letters = ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J')
DATA_PATH = os.path.join('..' , 'data', 'notMNIST_large')

if not os.path.isdir(DATA_PATH):
    raise Exception(f'No data found in path: {DATA_PATH}')


@with_goto
def main():
    goto .task
    # 1
    fig=plt.figure(figsize=(8, 8))
    columns = 5
    rows = 2
    for i in range(1, columns*rows + 1):
        path = os.path.join(DATA_PATH, letters[i-1], 'aG9tZXdvcmsgbm9ybWFsLnR0Zg==.png')
        img = mpimg.imread(path)
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)
    plt.show()

    # 2
    fig, ax = plt.subplots()
    density = []
    for i in letters:
        path = os.path.join(DATA_PATH, i)
        density.append(len(os.listdir(path)))

    print(density)

    ax.bar(letters, density, width=0.5)
    ax.set_ylabel('Amount of files')
    ax.set_title('Files in each group')
    plt.show()

    # 3
    label .task
    print('TODO: implement')

    # 4

    # 5


if __name__ == '__main__':
    main()
