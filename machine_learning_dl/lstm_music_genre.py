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
                                                      random_state=777)
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def build_model(input_shape):

    # create model
    model = keras.Sequential()

    # 2 LSTM layers
    model.add(keras.layers.LSTM(64, input_shape=input_shape, return_sequences=True))
    model.add(keras.layers.LSTM(64))

    # dense layer
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.3))
    
    # output layer
    model.add(keras.layers.Dense(10, activation='softmax'))

    return model


def predict(model, X, y):
    if len(X.shape) < 4:
        X = X[np.newaxis, ...]

    # predictions = [ [0.1, ...] ]
    predictions = model.predict(X)  # X -> (nb_samples, 130, 13, 1)

    # extract index with max value
    predicted_index = np.argmax(predictions, axis=1)

    print("Expected index: {}, Predicted index: {}".format(y, predicted_index))


if __name__ == "__main__":

    # create train, validation and test sets
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_datasets(
        0.25, 0.2)

    # build the CNN network
    input_shape = (X_train.shape[1], X_train.shape[2]) # 130, 13
    model = build_model(input_shape)

    # compile the network
    opt = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=opt, metrics=['accuracy'])

    # train the CNN network
    history = model.fit(X_train, y_train, validation_data=(
        X_val, y_val), batch_size=32, epochs=100, verbose=1)

    # plot accuracy and error over the epochs
    plot_history(history)

    # evaluate the CNN on the test set
    test_error, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
    print("Accuracy on test set is: {}".format(test_accuracy))
    print("Error on test set is: {}".format(test_error))

    # make prediction on a sample
    X = X_test[30]
    y = y_test[30]
    predict(model, X, y)
