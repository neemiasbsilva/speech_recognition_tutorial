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

# fft -> spectrum
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

# stft -> spectrogram
def apply_stft(signal, sr, n_fft=None, hop_length=None):
    if n_fft is None:
        n_fft = 2048 # number of samples
    if hop_length is None:
        hop_length = 512 # slide to shift to right


    stft = librosa.core.stft(signal, hop_length=hop_length, n_fft=n_fft)
    
    spectrogram = np.abs(stft)
    
    log_spectrogram = librosa.amplitude_to_db(spectrogram)

    librosa.display.specshow(log_spectrogram, sr=sr, hop_length=hop_length)
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.colorbar()
    plt.show()
    return spectrogram


# MFCCs
def apply_mfcc(signal, n_fft=None, hop_length=None):
    
    if n_fft is None:
        n_fft = 2048 # number of samples
    if hop_length is None:
        hop_length = 512 # slide to shift to right

    MFCCs = librosa.feature.mfcc(signal, n_ftt=n_fft, hop_lenght=hop_length, n_mfcc=13)

    librosa.display.specshow(MFCCs, sr=sr, hop_length=hop_length)
    plt.xlabel("Time")
    plt.ylabel("MFCC")
    plt.colorbar()
    plt.show()

    return MFCCs

if __name__ == "__main__":

    file_path = "./blues.00000.wav"

    signal, sr = load_waveform(file_path)
    magnitude, frequency = apply_fft(signal)
    spectrogram = apply_stft(signal, sr)
    MFCCs = apply_mfcc(signal)



