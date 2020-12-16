import librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np

# waveform
def load_waveform(file_path):
    signal, sr = librosa.load(file_path, sr=22050) # sr * T -> 22050 * 30

    librosa.display.waveplot(signal, sr=sr)
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.show()
    return signal, sr

# fft
def apply_fft(signal):
    fft = np.fft.fft(signal)

    magnitude = np.abs(fft)
    frequency = np.linspace(0,  sr, len(magnitude))

    left_frequency = frequency[:int(len(frequency)/2)]
    left_magnitude = magnitude[:int(len(magnitude)/2)]

    plt.plot(left_frequency, left_magnitude)
    plt.xlabel("Frequency")
    plt.ylabel("Magnitude")
    plt.show()

    return left_magnitude, left_frequency

if __name__ == "__main__":

    file_path = "./blues.00000.wav"

    signal, sr = load_waveform(file_path)
    magnitude, frequency = apply_fft(signal)





