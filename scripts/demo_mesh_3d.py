'''
Adapted from https://plotly.com/python/3d-mesh/
'''
import numpy as np
import plotly.graph_objects as go
URL = 'https://raw.githubusercontent.com/plotly/datasets/master/mesh_dataset.txt'
pts = np.loadtxt(np.DataSource().open(URL)) #NOTE Downloading data set from plotly repo
x, y, z = pts.T
fig = go.Figure(data=[go.Mesh3d(x=x, y=y, z=z, color='lightpink', opacity=0.50)])
fig.show()