# coding: UTF-8


import numpy as np
import matplotlib.pyplot as plt


H = 3
W = 3 
bCH = 3
aCH = 1


# Phi = np.random.choice((0, 1), H * W * bCH, p=(.5, .5))
Phi = np.ones(H * W * bCH)
split_phi = np.zeros((H * W * aCH, H * W * bCH))


for i in range(H * W * aCH):
    split_phi[i, i::H * W] = 1


Phi = Phi * split_phi
print(Phi[:5, :2 * H * W])
print(Phi.shape)
plt.imshow(Phi)
plt.show()
