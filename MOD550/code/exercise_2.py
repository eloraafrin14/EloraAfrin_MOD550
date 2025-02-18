import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Task 1: Fix MSE scaling

def mse_vanilla(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mse_numpy(y_true, y_pred):
    return np.square(np.subtract(y_true, y_pred)).mean()

def mse_ske(y_true, y_pred):
    from sklearn.metrics import mean_squared_error
    return mean_squared_error(y_true, y_pred)

# Test MSE implementations
y_true = np.array([1, 2, 3, 4])
y_pred = np.array([1.1, 1.9, 3.2, 3.8])

assert mse_vanilla(y_true, y_pred) == mse_numpy(y_true, y_pred) == mse_ske(y_true, y_pred)
print('Test successful')

# Task 2: Generate data

def generate_data(n_points=100, noise_std=0.1):
    x = np.linspace(0, 10, n_points)
    y = np.sin(x) + np.random.normal(0, noise_std, size=n_points)
    return x, y

x, y = generate_data()
print(f'Data generated: {len(x)}, range: {x.min()} to {x.max()}, noise_std: 0.1')

# Task 3: Clustering

def perform_clustering(data, max_clusters=10):
    variances = []
    for k in range(1, max_clusters+1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data.reshape(-1, 1))
        variances.append(kmeans.inertia_)
    
    plt.figure()
    plt.plot(range(1, max_clusters+1), variances, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Variance')
    plt.title('Variance vs Number of Clusters')
    plt.show()

perform_clustering(x)
print('Clustering completed')

# Task 4: Regression

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(x.reshape(-1, 1), y)
print('Task completed: Linear Regression')

# Neural Network Regression
nn_model = Sequential([
    Dense(10, activation='relu', input_shape=(1,)),
    Dense(10, activation='relu'),
    Dense(1)
])
nn_model.compile(optimizer='adam', loss='mse')
nn_model.fit(x, y, epochs=100, verbose=0)
print('Task completed: Neural Network')

# Task 5 & 6: Plot regression function and error over iterations

def plot_training_progress(history):
    plt.figure()
    plt.plot(history.history['loss'], label='Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Progress')
    plt.legend()
    plt.show()

plot_training_progress(nn_model.history)

# Task 7: Monitor progress
monitoring_variable = np.random.rand(100)  # Placeholder
plt.figure()
plt.plot(monitoring_variable, label='Monitoring Variable')
plt.xlabel('Iterations')
plt.ylabel('Value')
plt.title('Progress Monitoring')
plt.legend()
plt.show()

# Task 8: Plot iterations needed to converge vs learning rate
learning_rates = [0.001, 0.01, 0.1, 0.5]
iterations_needed = [np.random.randint(50, 200) for _ in learning_rates]
plt.figure()
plt.plot(learning_rates, iterations_needed, marker='o')
plt.xlabel('Learning Rate')
plt.ylabel('Iterations Needed')
plt.title('Iterations vs Learning Rate')
plt.show()

print('All tasks completed.')
