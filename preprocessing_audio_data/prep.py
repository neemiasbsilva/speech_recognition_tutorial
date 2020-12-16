import librosa, librosa.display
import matplotlib.pyplot as plt

# waveform

def load_waveform(file_path):
    signal, sr = librosa.load(file_path, sr=22050) # sr * T -> 22050 * 30

    librosa.display.waveplot(signal, sr=sr)
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.show()
    return signal, sr

if __name__ == "__main__":

    file_path = "./blues.00000.wav"

    signal, sr = load_waveform(file_path)




