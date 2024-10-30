import numpy as np
import matplotlib.pyplot as plt

# Parameters
sigma_x, sigma_y = 0.25, 0.25
sigma_noise = 0.4
grid_range = np.linspace(-2, 2, 100)

# Generate the true position randomly within unit circle
np.random.seed(0)
r_true = np.random.rand()
theta_true = 2 * np.pi * np.random.rand()
x_true, y_true = r_true * np.cos(theta_true), r_true * np.sin(theta_true)

def generate_landmarks(K):
    angles = np.linspace(0, 2 * np.pi, K, endpoint=False)
    return np.array([(np.cos(theta), np.sin(theta)) for theta in angles])

def generate_measurements(landmarks):
    measurements = []
    for lx, ly in landmarks:
        distance_true = np.sqrt((x_true - lx)**2 + (y_true - ly)**2)
        noisy_distance = distance_true + np.random.normal(0, sigma_noise)
        measurements.append(max(noisy_distance, 0))
    return np.array(measurements)

def map_objective(x, y, landmarks, measurements):
    likelihood = sum((measurements[i] - np.sqrt((x - landmarks[i, 0])**2 + (y - landmarks[i, 1])**2))**2 / (2 * sigma_noise**2)
                     for i in range(len(landmarks)))
    prior = x**2 / (2 * sigma_x**2) + y**2 / (2 * sigma_y**2)
    return likelihood + prior

def plot_map_contours(K):
    landmarks = generate_landmarks(K)
    measurements = generate_measurements(landmarks)
    
    # Evaluate objective over grid
    X, Y = np.meshgrid(grid_range, grid_range)
    Z = np.array([[map_objective(x, y, landmarks, measurements) for x in grid_range] for y in grid_range])

    # Plotting
    plt.figure(figsize=(8, 6))
    cp = plt.contour(X, Y, Z, levels=20, cmap='viridis')
    plt.colorbar(cp, label='MAP Objective')
    plt.plot(x_true, y_true, 'r+', markersize=12, label='True Position')
    plt.plot(landmarks[:, 0], landmarks[:, 1], 'bo', markersize=8, label='Landmarks')
    plt.title(f'MAP Objective Contours for K={K}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.axis('equal')
    plt.show()

# Plot for each K
for K in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
    plot_map_contours(K)