import os
import librosa
import math
import json


DATASET_PATH = "./Data/genres_original/"
JSON_PATH = "data.json"
SAMPLE_RATE = 22050
DURATION = 30 # measured in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

def save_mfcc(dataset_path, json_path, n_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):

    # build dictionary to store data
    data = {
        "mapping": [],
        "mfcc": [],
        "labels": []
    }

    num_samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    expected_n_mfcc_vectors_per_segment = math.ceil(num_samples_per_segment / hop_length)
    # loop through all the genres
    for count, (dir_path, dir_names, file_names) in enumerate(os.walk(dataset_path)):
        
        # ensure that we're not at the root level
        if dir_path is not dataset_path:
            # save the semanticc label
            dir_path_components = dir_path.split('/')
            semantic_label = dir_path_components[-1]
            data["mapping"].append(semantic_label)
            print("\nProcessing {}".format(semantic_label))
            # process files for a specific genre
            for f in file_names:
                # load audio file
                file_path = os.path.join(dir_path, f)
                if int(f[-6:-4]) < 20:
                    signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)

                    # process segments extracting mfcc and storing data
                    for s in range(num_segments):
                        start_sample = num_samples_per_segment * s
                        finish_sample = start_sample + num_samples_per_segment

                        mfcc = librosa.feature.mfcc(signal[start_sample:finish_sample],
                                                    sr=sr, n_fft=n_fft, n_mfcc=n_mfcc, hop_length=hop_length)
                        mfcc = mfcc.T

                        # store mfcc for segment if it has the expected length
                        if len(mfcc) == expected_n_mfcc_vectors_per_segment:
                            data["mfcc"].append(mfcc.tolist())
                            data["labels"].append(count-1)
                            print("all_filepath:{}, segment: {}".format(file_path, s))  
    
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)


if __name__ == "__main__":
    save_mfcc(DATASET_PATH, JSON_PATH, num_segments=5)
