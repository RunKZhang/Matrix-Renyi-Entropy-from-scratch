from validations import plot_gaussian, entropy_gaussian, generate_gaussian_samples
from renyi import Gram, RBF, Renyi_Calculator
import matplotlib.pyplot as plt
import numpy as np

# means = [0,0,2]
# variances = [1,2,1]
# colors = ['blue', 'green', 'red']

# plt.figure(figsize=(10,5))
# for mean, variance, color in zip(means, variances, colors):
#     plot_gaussian(mean,variance,color,'Gaussian Distribution')
variances = np.linspace(0.1, 5, 20)
alpha = 10
num_samples = 1000

# Shannon Entropy
entropies = entropy_gaussian(variances)
# Renyi Entropy
renyi_entropies = []

for variance in variances:
    samples = generate_gaussian_samples(mean=0, variance=variance, num_samples=num_samples)
    A = Gram(RBF(np.expand_dims(samples, axis=1), 1))
    renyi_calc = Renyi_Calculator('entropy', alpha, A)
    entropy_val = renyi_calc.fit()
    print(entropy_val)
    renyi_entropies.append(entropy_val)
    
    # print(A.shape)
renyi_entropies = np.array(renyi_entropies)
plt.plot(variances, entropies, label = 'Renyi Entropy of Gaussian Distribution')
plt.plot(variances, entropies, label = 'Entropy of Gaussian Distribution')
# plt.title('Gaussian Distributions Under Different Conditions')
# plt.xlabel('x')
# plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)
plt.show()