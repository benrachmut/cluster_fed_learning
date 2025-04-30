import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Example data: rows = iterations, columns = trials
data = np.random.randn(1000, 30)  # 1000 iterations, 30 runs

# Compute mean and confidence interval for each iteration
means = np.mean(data, axis=1)
confidence = 0.95
n = data.shape[1]
stderr = stats.sem(data, axis=1)
h = stderr * stats.t.ppf((1 + confidence) / 2, n - 1)

# Plot
iterations = np.arange(len(means))
plt.plot(iterations, means, label='Mean')
plt.fill_between(iterations, means - h, means + h, color='blue', alpha=0.2, label='95% CI')
plt.xlabel("Iteration")
plt.ylabel("Value")
plt.legend()
plt.title("Average with 95% Confidence Interval")
plt.show()
