import numpy as np

def main():
    X, y = generate_data(1000)

    data = np.hstack((X, np.transpose([y])))
    np.savetxt('data.csv', data, delimiter=',')

def generate_data(num_samples, r_neg=2, r_pos=4, sigma=1, error_rate=0.05):

    # Randomly assign class labels
    y = np.random.choice([-1, 1], size=num_samples)

    theta = np.random.uniform(-np.pi, np.pi, num_samples)
    radii = np.where(y == -1, r_neg, r_pos)

    # Compute clean signal
    x_clean = np.vstack((radii * np.cos(theta), radii * np.sin(theta))).T

    # Add Gaussian noise
    noise = np.random.normal(0, sigma, x_clean.shape)
    X = x_clean + noise


    return X, y

if __name__ == '__main__':
    main()