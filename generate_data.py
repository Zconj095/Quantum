import numpy as np


def generate_data(timesteps, data_dim, num_samples=1000):
    """
    Generates synthetic time series data simulating encoded glyphs and cryostasis responses.
    Each sample will have a cyclical pattern with added noise to simulate variations in glyph encoding.
   
    :param timesteps: Number of time steps per sample.
    :param data_dim: Number of features (simulating different types of glyphs).
    :param num_samples: Total number of samples to generate.
    :return: Tuple of numpy arrays (X, y) representing time series data and corresponding targets.
    """
    np.random.seed(42)  # Ensure reproducibility
   
    # Generating cyclical patterns
    x_values = np.linspace(0, 2 * np.pi, timesteps)
    cyclical_data = np.sin(x_values)  # Sinusoidal pattern to simulate cyclical glyph effects
   
    # Generating data samples
    X = np.zeros((num_samples, timesteps, data_dim))
    y = np.zeros((num_samples, data_dim))
   
    for i in range(num_samples):
        for d in range(data_dim):
            noise = np.random.normal(0, 0.1, timesteps)  # Adding noise to simulate variations
            X[i, :, d] = cyclical_data + noise
            y[i, d] = cyclical_data[-1] + np.random.normal(0, 0.1)  # Target is the final step of the cycle with noise
           
    return X, y
