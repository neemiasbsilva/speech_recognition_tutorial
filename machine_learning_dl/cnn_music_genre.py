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


if __name__ == "__main__":

    X, y = get_dataset(DATASET_PATH)

    # split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        stratify=y,
                                                        random_state=777,
                                                        shuffle=True)

    
    # plot accuracy and error over the epochs
    plot_history(history)
