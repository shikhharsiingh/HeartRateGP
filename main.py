import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern, RationalQuadratic

# Simulated time and wearable data (e.g., heart rate)
time = np.linspace(0, 10, 100)  # time in arbitrary units (e.g., minutes)
ground_truth = 60 + 10 * np.sin(time)  # ground truth heart rate data
observed_data = ground_truth + np.random.normal(0, 2, time.shape)  # add noise

# Reshape time for GP model
time = time.reshape(-1, 1)

# Define different Gaussian Processes
kernels = [
    RBF(length_scale=1.0) + WhiteKernel(noise_level=1),
    Matern(length_scale=1.0, nu=1.5) + WhiteKernel(noise_level=1),
    RationalQuadratic(length_scale=1.0, alpha=0.1) + WhiteKernel(noise_level=1),
]

gp_models = [
    GaussianProcessRegressor(kernel=k, n_restarts_optimizer=10) for k in kernels
]
gp_names = ["RBF", "Matern", "Rational Quadratic"]


# Recursive function to add missing data and evaluate GPs
def evaluate_gps_with_missing_data(gap_size=10):
    global observed_data
    gap_indices = np.arange(40, 40 + gap_size)  # Induce missing data in a range
    masked_data = observed_data.copy()
    masked_data[gap_indices] = np.nan  # Set gap to NaN

    print(f"\nEvaluating with gap size {gap_size}...")

    plt.figure(figsize=(15, 10))

    for i, gp in enumerate(gp_models):
        # Mask time and data for training (exclude NaNs)
        train_time = time[~np.isnan(masked_data)]
        train_data = masked_data[~np.isnan(masked_data)]

        # Train the GP
        gp.fit(train_time, train_data)

        # Predict on the full time range
        mean_pred, std_pred = gp.predict(time, return_std=True)

        # Calculate mean error on missing values
        missing_values = ground_truth[gap_indices]
        predicted_values = mean_pred[gap_indices]
        error = np.mean(np.abs(predicted_values - missing_values))

        # Plot the results
        plt.subplot(2, 2, i + 1)
        plt.plot(time, observed_data, "r.", label="Observed Data")
        plt.plot(time, ground_truth, "g-", label="Ground Truth", alpha=0.5)
        plt.plot(time, mean_pred, "b-", label=f"{gp_names[i]} Prediction")
        plt.fill_between(
            time.ravel(),
            mean_pred - 1.96 * std_pred,
            mean_pred + 1.96 * std_pred,
            alpha=0.2,
            color="blue",
            label="95% Confidence Interval",
        )
        plt.scatter(
            time[gap_indices],
            missing_values,
            color="purple",
            marker="x",
            s=50,
            label="Missing Data",
        )
        plt.xlabel("Time")
        plt.ylabel("Wearable Data")
        plt.title(f"{gp_names[i]} Model (Mean Error on Missing Data: {error:.2f} bpm)")
        plt.legend()

    plt.tight_layout()
    plt.show()

    # If all models produce <2 bpm error, increase gap size and repeat
    if all(
        np.mean(np.abs(mean_pred[gap_indices] - ground_truth[gap_indices])) < 2
        for _ in gp_models
    ):
        evaluate_gps_with_missing_data(gap_size=gap_size + 5)


# Run the recursive evaluation
evaluate_gps_with_missing_data(gap_size=10)
