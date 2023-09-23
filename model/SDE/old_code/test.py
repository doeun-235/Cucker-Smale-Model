import numpy as np
import matplotlib as mlp
from matplotlib import pyplot as plt
import os

x = np.linspace(0,np.pi,10)
y = np.sin(x)

plt.plot(x,y)
plt.savefig('./d.png')
plt.show()

path = os.getcwd()
print(path)