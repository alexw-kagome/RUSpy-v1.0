import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Function to generate Lorentzian data
def generate_lorentzian(w, w0, gamma, A, phi):
    return A * np.exp(phi * 1j) / (w - w0 + gamma / 2 * 1j)

def generate_data(num_peaks, num_points, noise_level=1e-7, gamma_range=(1e2, 1e3), amplitude_range=(1e-3, 1e-1)):
    # Randomize frequency range within the global limits (1 MHz to 9 MHz)
    start_freq = np.random.uniform(1e6, 8e6)  # Ensure that the range can fit within 1 MHz
    freq_range = (start_freq, start_freq + 1e6)  # 1 MHz wide frequency range
    
    w = np.linspace(freq_range[0], freq_range[1], num_points)
    real_part = np.zeros(num_points, dtype=np.float32)
    imag_part = np.zeros(num_points, dtype=np.float32)
    
    resonance_params = []

    # num_peaks = int(num_peaks * np.log((start_freq / 1e6 + 3)))
    # print(start_freq)

    for _ in range(num_peaks):
        w0 = np.random.uniform(freq_range[0], freq_range[1])
        gamma = np.random.uniform(gamma_range[0], gamma_range[1])
        A = np.random.uniform(amplitude_range[0], amplitude_range[1])
        phi = np.random.uniform(0, 2 * np.pi)
        
        resonance_params.append((w0, gamma, A, phi))
        
        lorentzian = generate_lorentzian(w, w0, gamma, A, phi)
        real_part += np.real(lorentzian)
        imag_part += np.imag(lorentzian)

    scale = 5e-9
    # scale = 0
    # Adding a linear background
    background_slope = np.random.uniform(-scale, scale)
    background_intercept = np.random.uniform(-scale, scale)
    linear_background = background_slope * w + background_intercept
    
    real_part += linear_background
    imag_part += linear_background

    # Adding noise
    real_part += np.random.normal(0, noise_level, num_points)
    imag_part += np.random.normal(0, noise_level, num_points)

    return w, real_part, imag_part, resonance_params

# Prepare training data
def prepare_data(num_samples, num_peaks, num_points):
    X = []
    y = []
    i = 0
    for _ in range(num_samples):
        
        peaks = np.random.randint(num_peaks * 0.5, num_peaks + 1.2)
        w, real_part, imag_part, resonance_params = generate_data(peaks, num_points)
        
        # Flatten data and append to X
        X.append(np.concatenate([real_part, imag_part]))
        
        # Flatten parameters and append to y
        y_params = []
        for params in resonance_params:
            y_params.extend(params)
        y.extend(y_params + [0] * (num_peaks * 4 - len(y_params)))  # Use 0 instead of NaN

        print(i)
        i = i + 1
    
    X = np.array(X)
    y = np.array(y).reshape(-1, num_peaks * 4)  # Each sample has 4 parameters per peak
    
    return X, y

# Check for NaN or Inf in data
def check_data(X, y):
    if np.isnan(X).any() or np.isnan(y).any():
        raise ValueError("Data contains NaN values")
    if np.isinf(X).any() or np.isinf(y).any():
        raise ValueError("Data contains Inf values")
    
# Plot one of the generated data samples
def plot_sample(w, real_part, imag_part, resonance_params):
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(w, real_part, label='Real Part')
    plt.title('Generated Data Sample')
    plt.xlabel('Frequency')
    plt.ylabel('Real Part')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(w, imag_part, label='Imaginary Part', color='orange')
    plt.xlabel('Frequency')
    plt.ylabel('Imaginary Part')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# Define parameters
num_points = 10000
num_samples = 10000
num_peaks = 20

# Generate and plot a single sample
w, real_part, imag_part, resonance_params = generate_data(num_peaks, num_points=num_points)
print(real_part[5])
plot_sample(w, real_part, imag_part, resonance_params)


# Generate data
X, y = prepare_data(num_samples, num_peaks, num_points)

check_data(X, y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale data
scaler_X = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)

# Define the model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],), kernel_initializer='he_normal'),
    Dense(64, activation='relu', kernel_initializer='he_normal'),
    Dense(y_train.shape[1], activation='linear')
])

# Adjust learning rate
optimizer = Adam(learning_rate=1e-4)
model.compile(optimizer=optimizer, loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

# Evaluate the model
loss = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')

# Save the trained model
model.save('lorentzian_model.h5')
print("Model saved as 'lorentzian_model.h5'")

# Plot training history
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training History')
plt.legend()
plt.show()

# Loading the model (for demonstration)
model_loaded = load_model('lorentzian_model.h5')
print("Model loaded from 'lorentzian_model.h5'")
