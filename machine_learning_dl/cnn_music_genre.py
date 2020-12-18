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


if __name__ == "__main__":

    # create train, validation and test sets
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_datasets(0.25, 0.2)

    
    # plot accuracy and error over the epochs
    plot_history(history)
