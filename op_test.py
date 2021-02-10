#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import torch


# In[2]:


from torch.utils.data import DataLoader
from generate_data import generate_op_data
from utils import load_model
from problems import OP


# In[3]:


model, _ = load_model('pretrained/op_const_50/')
torch.manual_seed(1234)
dataset_1 = OP.make_dataset(size=100, num_samples=1)
dataset = OP.make_dataset(filename="asist\\falcon.pkl")


# In[4]:


print(dataset.data)
# print(dataset_1.data)


# In[5]:


# Need a dataloader to batch instances
dataloader = DataLoader(dataset, batch_size=1)

# Make var works for dicts
batch = next(iter(dataloader))

print(type(batch))
print(batch)

# Run the model
model.eval()
model.set_decode_type('greedy')
with torch.no_grad():
    length, log_p, pi = model(batch, return_pi=True)
tour = pi


# In[6]:


print(tour)

#
# # In[25]:
#
#
# get_ipython().run_line_magic('matplotlib', 'inline')
# from matplotlib import pyplot as plt
#
# from matplotlib.collections import PatchCollection
# from matplotlib.patches import Rectangle
# from matplotlib.lines import Line2D
#
# # Code inspired by Google OR Tools plot:
# # https://github.com/google/or-tools/blob/fb12c5ded7423d524fc6c95656a9bdc290a81d4d/examples/python/cvrptw_plot.py
#
# def plot_op(xy, tour, ax1):
#     """
#     Plot the TSP tour on matplotlib axis ax1.
#     """
#
#     ax1.set_xlim(0, 1)
#     ax1.set_ylim(0, 1)
#
#     xs, ys = xy[tour].transpose()
#     xs, ys = xy[tour].transpose()
#     dx = np.roll(xs, -1) - xs
#     dy = np.roll(ys, -1) - ys
#     d = np.sqrt(dx * dx + dy * dy)
#     lengths = d.cumsum()
#
#     # Scatter nodes
#     ax1.scatter(xs, ys, s=40, color='blue')
#     # Starting node
#     ax1.scatter([xs[0]], [ys[0]], s=100, color='red')
#
#     # Arcs
#     qv = ax1.quiver(
#         xs, ys, dx, dy,
#         scale_units='xy',
#         angles='xy',
#         scale=1,
#     )
#
#     ax1.set_title('{} nodes, total length {:.2f}'.format(len(tour), lengths[-1]))
#
# fig, ax = plt.subplots(figsize=(10, 10))
# xy = dataset.data[0]['loc']
# print(xy)
# plot_op(xy, tour, ax)


# In[ ]:




