import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

def plot_gaussian(mean,variance, color, label):
    sigma = np.sqrt(variance)
    x = np.linspace(mean - 3*sigma, mean + 3*sigma, 1000)
    y = (1 / (sigma * np.sqrt(2 *  np.pi))) * np.exp(-0.5 * ((x - mean) / sigma) ** 2)

    # Plot shape
    plt.plot(x, y, color=color, label = f'Mean = {mean}, Variance = {variance}')
    plt.fill_between(x, y, color=color, alpha=0.1)

    # Mark the mean and one standard deviation
    plt.axvline(x = mean, color = color, linestyle = '--', linewidth = 0.5)
    plt.axvline(x = mean + sigma, color = color, linestyle = ':', linewidth = 0.5)
    plt.axvline(x = mean - sigma, color = color, linestyle = ':', linewidth = 0.5)

    # plt.show()

def shannon_entropy_gaussian(variance):
    # Compute the entropy of a Gaussian distribution with given variance
    return 0.5*(1 + np.log(2 * np.pi)) + np.log(np.sqrt(variance))
    # return 0.5 * np.log2(2 * np.pi * np.e * variance)

def generate_gaussian_samples(mean, variance, num_samples = 1000):
    return np.random.normal(loc=mean, scale=np.sqrt(variance), size=num_samples)

def discretize_gaussian(mu, sigma, bins, epsilon=1e-10):
    """Discretize a Gaussian distribution into specified bins."""
    print('Hello')
    edges = np.linspace(mu - 10*sigma, mu+10*sigma, bins+1) # Cover most of the distribution
    cdf_values = norm.cdf(edges, mu, sigma)
    probabilities = np.diff(cdf_values)

    # Avoid zero probabilities
    probabilities = np.clip(probabilities, epsilon, 1)
    return probabilities

def theoretical_renyi_entropy(variance, alpha):
    return np.log(np.sqrt(variance))+0.5*np.log(2*np.pi)+np.log(alpha)/(2*(alpha-1))
    # return 0.5 * np.log2(2 * np.pi * np.e * variance * alpha**(1/(alpha-1)))

def discretize_shannon_entropy_gaussian(probabilities):
    """Calculate the Shaanon entropy from a list of probabilities."""
    probabilities = probabilities[probabilities > 0]
    return -np.sum(probabilities * np.log2(probabilities))
# def plot_entropy_gaussian(variances):
#     entropies = entropy_gaussian(variances)
    