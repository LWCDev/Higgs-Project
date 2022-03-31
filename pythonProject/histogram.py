
from matplotlib import pyplot as plt

list = open('predictions.txt').readlines()
for i in range(len(list)):
    list[i] = float(list[i])

print(list)
plt.hist(list, 100)
plt.xlim(-5,5)
plt.ylim(0,50)
plt.show()