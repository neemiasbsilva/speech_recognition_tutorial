import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import matplotlib.pyplot as plt


DATASET_PATH = "./data.json"


def get_dataset(dataset_path):
    with open(dataset_path, 'r') as fp:
        dataset = json.load(fp)

    inputs = np.asarray(dataset["mfcc"])
    targets = np.array(dataset["labels"])

    return inputs, targets


def plot_history(history):

    fig, axs = plt.subplots(2)

    # create accuracy subplot
    axs[0].plot(history.history["acc"], label="train accuracy")
    axs[0].plot(history.history["val_acc"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")

    # create error subplot
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epochs")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")

    plt.show()

def prepare_datasets(test_size, val_size):

    # load data
    X, y = get_dataset(DATASET_PATH)


    # create train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=test_size,
                                                        stratify=y,
                                                        random_state=777,
                                                        shuffle=True)
    # create train/val split
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                        test_size=val_size,
                                                        stratify=y,
                                                        random_state=777,
                                                        shuffle=True)
    # 3d array -> (130, 13, 1)
    X_train = X_train[..., np.newaxis] # 4d array -> (nb_samples, 130, 13, 1)
    X_val = X_val[..., np.newaxis]  # 4d array -> (nb_samples, 130, 13, 1)
    X_test = X_test[..., np.newaxis]  # 4d array -> (nb_samples, 130, 13, 1)


    return X_train, X_val, X_test, y_train, y_val, y_test

def build_model(input_shape):

    # create model
    model = keras.Sequentia()

    # 1st conv layer
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPool2D( (3,3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # 2nd conv layer
    model.add(keras.layers.Conv2D(
        32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # 3rd conv layer
    model.add(keras.layers.Conv2D(
        32, (2, 2), activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPool2D((2, 2), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # flatten the output and feed it int dense layer
    model.add(keras.layers.Flatten())

    model.add(keras.layers.Dense(64, activation='relu',
                                 kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(keras.layers.Dropout(0.3))

    # output layer
    model.add(keras.layers.Dense(10, activation='softmax'))

    return model


if __name__ == "__main__":

    # create train, validation and test sets
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_datasets(0.25, 0.2)

    # build the CNN network
    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
    model = build_model(input_shape)
    
    # plot accuracy and error over the epochs
    plot_history(history)
