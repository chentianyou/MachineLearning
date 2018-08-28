# For plotting the images
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import preprocessing

from som import SOM


def main():
    # Training inputs for RGBcolors
    credits_data = pd.read_csv("testdata/credit_train.csv", header=None)
    columns = credits_data.columns
    train_data = credits_data[columns[:-1]].copy().values
    label_data = credits_data[columns[-1]].copy().values
    train_data_scale = preprocessing.scale(train_data)
    # colors = np.array(
    #     [[0., 0., 0.],
    #      [0., 0., 1.],
    #      [0., 0., 0.5],
    #      [0.125, 0.529, 1.0],
    #      [0.33, 0.4, 0.67],
    #      [0.6, 0.5, 1.0],
    #      [0., 1., 0.],
    #      [1., 0., 0.],
    #      [0., 1., 1.],
    #      [1., 0., 1.],
    #      [1., 1., 0.],
    #      [1., 1., 1.],
    #      [.33, .33, .33],
    #      [.5, .5, .5],
    #      [.66, .66, .66]])
    # color_names = \
    #     ['black', 'blue', 'darkblue', 'skyblue',
    #      'greyblue', 'lilac', 'green', 'red',
    #      'cyan', 'violet', 'yellow', 'white',
    #      'darkgrey', 'mediumgrey', 'lightgrey']

    # Train a 20x30 SOM with 400 iterations
    som = SOM(1, 2, len(columns) - 1, 400)
    som.train(train_data_scale)

    # Get output grid
    image_grid = som.get_centroids()

    # Map colours to their closest neurons
    mapped = som.map_vects(train_data_scale)

    # Plot
    plt.imshow(image_grid)
    plt.title('Color SOM')
    for i, m in enumerate(mapped):
        plt.text(m[1], m[0], label_data[i], ha='center', va='center',
                 bbox=dict(facecolor='white', alpha=0.5, lw=0))
    plt.show()


if __name__ == '__main__':
    main()
