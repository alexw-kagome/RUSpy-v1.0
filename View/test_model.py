import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from scipy.interpolate import interp1d

# Function to generate Lorentzian data
def generate_lorentzian(w, w0, gamma, A, phi):
    return A * np.exp(phi * 1j) / (w - w0 + gamma / 2 * 1j)

# Function to prepare a single sample for the model
def prepare_single_sample(real_part, imag_part):
    num_features = real_part.shape[0]
    # No error checking here; just concatenate and add batch dimension
    X = np.concatenate([real_part, imag_part])
    X = np.expand_dims(X, axis=0)  # Add batch dimension
    return X

# Function to resample or interpolate data
def resample_data(original_data, new_length):
    x_original = np.linspace(0, 1, len(original_data))
    x_new = np.linspace(0, 1, new_length)
    interpolation_function = interp1d(x_original, original_data, kind='linear', fill_value='extrapolate')
    return interpolation_function(x_new)

def openbinfile(filename):
    dat = open(filename, 'r')
    data = np.fromfile(dat, dtype=np.dtype('>f8'))  # the raw data are binary files, so this imports them
    split_index = int((len(data) - 1) / 3)
    frequency = data[1: split_index + 1]
    x = data[split_index + 1: 2 * split_index + 1]
    y = data[2 * split_index + 1: 3 * split_index + 1]
    return x[20:], y[20:], np.sqrt(x ** 2 + y**2)[20:]

# Load the trained model
model = load_model('lorentzian_model.h5')

# Load data from files
path = r'D:\Alexander\Bi2Se3-2407A\corner\cooled\77.59664K_1400_2500_2.00000V_0.00000T_Jul18-2024__20-47-54_001_071824.bin'
real_part, imag_part, magnitude = openbinfile(path)

# Check dimensions of loaded data
print(f"Real part shape: {real_part.shape}")
print(f"Imaginary part shape: {imag_part.shape}")
print(f"Magnitude shape: {magnitude.shape}")

# Number of data points (adjust if necessary)
expected_points = 100000  # The number of points the model was trained with

# Resample data to match the model input size
real_part = resample_data(real_part, expected_points)
imag_part = resample_data(imag_part, expected_points)
magnitude = resample_data(magnitude, expected_points)

# Prepare input data for prediction
X_test = prepare_single_sample(real_part, imag_part)

# Make predictions
y_pred = model.predict(X_test)

# Number of peaks (adjust based on your setup)
num_peaks = 20  # This should be adjusted based on your model and data

# Reshape predictions
y_pred = y_pred.reshape(-1, num_peaks * 4)  # Each sample has 4 parameters per peak

# Generate frequency array
w = np.linspace(1e6, 1e7, expected_points)  # Adjust according to your data

# Plot results
def plot_with_predictions(w, magnitude, predicted_params):
    plt.figure(figsize=(14, 7))
    
    # Plot magnitude
    plt.plot(w, magnitude, label='Magnitude')

    # Plot predicted Lorentzian peaks
    for i in range(num_peaks):
        w0, gamma, A, phi = predicted_params[i*4:i*4+4]
        lorentzian = generate_lorentzian(w, w0, gamma, A, phi)
        plt.plot(w, np.abs(lorentzian), label=f'Peak {i+1} Prediction', linestyle='--')
        plt.axvline(x=w0, color='red', linestyle='--', label=f'w0 {i+1} = {w0:.2e}')

    plt.xlabel('Frequency')
    plt.ylabel('Magnitude')
    plt.title('Magnitude with Predicted Lorentzian Peaks')
    plt.legend()
    plt.show()

# Example plotting
plot_with_predictions(w, magnitude, y_pred[0])

# Print predicted parameters
print("Predicted Parameters:")
for i in range(num_peaks):
    params = y_pred[0][i*4:i*4+4]
    w0, gamma, A, phi = params
    print(f"\nPeak {i+1}:")
    print(f"  - w0: {w0:.2e}")
    print(f"  - gamma: {gamma:.2e}")
    print(f"  - A: {A:.2e}")
    print(f"  - phi: {phi:.2f}")
