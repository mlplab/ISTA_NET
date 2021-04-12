# coding: utf-8


import scipy.io
import numpy as np
import matplotlib.pyplot as plt
# from colour_func import RGB_10_degree_cmfs_to_XYZ_10_degree_cmfs
from colour.colorimetry import transformations
from utils import normalize


func_name = transformations.RGB_10_degree_cmfs_to_XYZ_10_degree_cmfs
start_wave = 420
last_wave = 720
# data_name = 'dataset/icvl/mat/icvl.mat'
# data_name = 'dataset/test/stuffed_toys_ms.mat'
data_name = 'img2.mat'
x = np.arange(start_wave, last_wave + 1, 10)


f = scipy.io.loadmat(data_name)
data = normalize(f['ref'])
# rgb = normalize(f['rgb'])
# plt.imsave('output_img/label.png', rgb)
print(data.shape)
trans_filter = func_name(x)
h, w, ch = data.shape
data = normalize(data.reshape(h * w, ch))
trans_img = data.dot(trans_filter)
trans_img = trans_img.reshape(h, w, 3)
# X, Y, Z = trans_img[:, :, 0], trans_img[:, :, 1], trans_img[:, :, 2]
# R =  3.2404542*X - 1.5371385*Y - 0.4985314*Z
# G = -0.9692660*X + 1.8760108*Y + 0.0415560*Z
# B =  0.0556434*X - 0.2040259*Y + 1.0572252*Z
# show_data = np.array([R, G, B], dtype=np.float32).transpose(1, 2, 0)
show_data = trans_img
show_data = np.where(show_data > 1., 1., show_data)
show_data = np.where(show_data < .0, 0., show_data)
print(show_data.max())
print(show_data.min())
plt.imshow(show_data)
plt.show()
# for i, ch in enumerate(range(start_wave, last_wave + 1, 10)):
#      print(i, ch)
#      trans_data = normalize(np.expand_dims(data[:, :, i], axis=-1).dot(np.expand_dims(trans_filter[i], axis=0)))
#      show_data = trans_data
#      plt.imsave(f'output_img/{i}_{ch}.png', show_data)
#      plt.close()
# g_ch =11
# b_ch =4
# for i in range(19, 31):
#      show_data = normalize(data[:, :, (i, g_ch, b_ch)])
#      # show_data = trans_data
#      # plt.imshow(show_data)
#      plt.imsave(f'output_img/RGB_{i}_{g_ch}_{b_ch}.png', show_data)
#      # plt.savefig(f'output_img/RGB_{i}_{g_ch}_{b_ch}.png')
#      # plt.close()
#      # plt.show()
