import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

# Ideally, move these to a setup script or ensure it's downloaded once
# nltk.download('punkt')
# nltk.download('stopwords')

# Load stopwords once
english_stopwords = set(stopwords.words('english'))

# Sample user input
user_input = "Show system of user input of user outputs."

# Preprocessing
def preprocess_text(text):
    text = text.lower()  # Lowercase text
    text = re.sub(r'\W', ' ', text)  # Remove all non-word characters
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    words = nltk.word_tokenize(text)  # Tokenize into words
    words = [word for word in words if word not in english_stopwords]  # Remove stopwords
    ps = PorterStemmer()
    words = [ps.stem(word) for word in words]  # Stemming
    return ' '.join(words)

preprocessed_input = preprocess_text(user_input)

# Feature Extraction
vectorizer = TfidfVectorizer()
vectorized_input = vectorizer.fit_transform([preprocessed_input]).toarray()

# Continue with logic based on vectorized input

print(preprocessed_input)
print(vectorized_input)

import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression

# Example time series data (could represent anything, e.g., daily temperatures)
time_series_data = np.array([20, 21, 19, 22, 24, 23, 25, 26, 27, 28, 29, 30, 31, 29, 28, 27, 26, 25, 24, 23])

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
time_series_data_scaled = scaler.fit_transform(time_series_data.reshape(-1, 1))

# Function to split time series data into samples
def split_sequence(sequence, n_steps):
    X, y = [], []
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence)-1:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

# Prepare input data
n_input = 3  # Number of time steps
n_features = 1  # Assuming a univariate time series
X, y = split_sequence(time_series_data_scaled, n_input)

# Reshape from [samples, timesteps] to [samples, timesteps, features]
X = X.reshape((X.shape[0], X.shape[1], n_features))

# Define LSTM model
model = Sequential([
    LSTM(50, activation='relu', input_shape=(n_input, n_features)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X, y, epochs=200, verbose=0)

# Function to make a prediction and generate an explanation
def predict_and_explain(model, last_observations, scaler, threshold=0.5):
    # Reshape and normalize input
    last_observations_scaled = scaler.transform(np.array(last_observations).reshape(-1, 1))
    last_observations_scaled = last_observations_scaled.reshape((1, n_input, n_features))
    
    # Make prediction
    prediction_scaled = model.predict(last_observations_scaled)
    prediction = scaler.inverse_transform(prediction_scaled)[0][0]
    
    # Generate explanation
    explanation = "The forecast indicates a rise in the next time step." if prediction > threshold else "The forecast indicates a drop in the next time step."
    
    return prediction, explanation

# Example usage
last_observations = time_series_data[-n_input:]
prediction, explanation = predict_and_explain(model, last_observations, scaler)
print(f"Predicted value for the next time step: {prediction}")
print(explanation)

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Continue with model training, prediction, and evaluation...

# Assuming you've reserved some data for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model on the training set
model.fit(X_train, y_train, epochs=200, verbose=0)

# Make predictions on the test set
y_pred_scaled = model.predict(X_test)
y_pred = scaler.inverse_transform(y_pred_scaled)

# Calculate the Mean Squared Error on the test set
mse = mean_squared_error(scaler.inverse_transform(y_test), y_pred)
print(f"Test MSE: {mse}")

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np

# Example time series data (could represent anything, e.g., daily temperatures)
time_series_data = np.array([20, 21, 19, 22, 24, 23, 25, 26, 27, 28, 29, 30, 31, 29, 28, 27, 26, 25, 24, 23])

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
time_series_data_scaled = scaler.fit_transform(time_series_data.reshape(-1, 1))

# Now, you would follow the steps to create sequences from this scaled data
# and proceed with training or making predictions with your model as previously described

# Now you can use `cluster_labels` to analyze the clustering of your sentences


# Example definitions
n_input = 3  # Number of time steps (e.g., using the last 3 observations to predict the next one)
n_features = 1  # Assuming a univariate time series (e.g., only one observation per time step)

# Assuming time_series_data is a list or numpy array of your time series values
# Select the last n_input values from your time series data
last_observations = time_series_data[-n_input:]

# Reshape this data into the shape expected by the model: [1, n_input, n_features]
next_time_step_features = np.array(last_observations).reshape((1, n_input, n_features))


# Assuming you have a trained model named `model`
next_step_prediction = model.predict(next_time_step_features)

print(f"Predicted value for the next time step: {next_step_prediction}")


# Example time series forecasting model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(n_input, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Train the model
# Here, `X_train` and `y_train` represent your time series data prepared for the LSTM
# model.fit(X_train, y_train, epochs=200, verbose=0)

from keras.models import Sequential
from keras.layers import LSTM, Dense

# Define model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(n_input, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

from deap import base, creator, tools
import random
import numpy as np

# Create Fitness and Individual Types
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# Initialize Toolbox
toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, -10, 10)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=2)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evaluate(individual):
    x, y = individual
    # Example evaluation function: Sphere function negated for maximization
    return -1.0 * (x**2 + y**2),

toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)

def main():
    random.seed(64)  # For reproducibility
    POP_SIZE = 100
    NGEN = 50
    CXPB = 0.7
    MUTPB = 0.2

    # Create an initial population
    pop = toolbox.population(n=POP_SIZE)

    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    for g in range(NGEN):
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Replace the old population by the offspring
        pop[:] = offspring

        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]

        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5

        print(f"Gen {g}: Min {min(fits)}, Max {max(fits)}, Avg {mean}, Std {std}")

main()

import numpy as np
from sklearn.linear_model import LinearRegression

def simulate_streaming_data():
    # Placeholder for simulating streaming data
    return np.random.rand()

def forecast_future_data_point(history):
    model = LinearRegression()
    # Create a feature array with indices of the last 5 data points
    past_indices = np.array(range(len(history) - 5, len(history))).reshape(-1, 1)
    # Use the last 5 data points as targets
    past_data_points = np.array(history[-5:]).reshape(-1, 1)
    # Train the model
    model.fit(past_indices, past_data_points)
    # Predict the next data point
    next_point_index = np.array([[len(history)]]).reshape(-1, 1)
    next_point = model.predict(next_point_index)
    return next_point.flatten()[0]

# Generate historical data
history = [simulate_streaming_data() for _ in range(50)]
predicted_next_point = forecast_future_data_point(history)
print(f"Predicted next data point: {predicted_next_point}")

# Define a threshold for the narrative summary
threshold = 0.5  # This is arbitrary and should be adjusted based on your data and needs

def generate_narrative_summary(predicted_point):
    if predicted_point > threshold:
        return "The system predicts an upward trend in the next period."
    else:
        return "The system forecasts a stable or downward trend in the upcoming period."

narrative_summary = generate_narrative_summary(predicted_next_point)
print(narrative_summary)
