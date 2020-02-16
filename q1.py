import matplotlib.pyplot as plt
mu = 1.0
sigma = 9.0
n = 10.0
lam = [float(i) for i in range(100)]
print(lam)
bias = []
variance = []
for i in range(100):
    bias.append(((lam[i]*mu)/(lam[i]+1))**2)
    variance.append(sigma/(n*(lam[i]+1)**2))
print(bias)
plt.plot(lam, bias)
plt.show()
plt.plot(lam, variance)
plt.show()