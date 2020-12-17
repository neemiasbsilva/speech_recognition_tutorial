import json
import numpy as np
from sklearn.model_selection import train_test_split

DATASET_PATH = "./data.json"

def get_dataset(dataset_path):
    with open(dataset_path, 'r') as fp:
        dataset = json.load(fp)

    inputs = np.asarray(dataset["mfcc"])
    targets = np.array(dataset["labels"])

    return inputs, targets


if __name__ == "__main__":

    X, y = get_dataset(DATASET_PATH)
    
    # split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                            test_size=0.2,
                                            stratify=y,
                                            random_state=777,
                                            shuffle=True)
    
    

