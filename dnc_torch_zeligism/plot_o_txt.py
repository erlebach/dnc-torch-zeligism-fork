import matplotlib.pyplot as plt
import numpy as np

data = np.loadtxt("o.txt")
print(data)
plt.plot(data)
plt.title("dnc-torch-zeligism")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.grid(True)
plt.show()
