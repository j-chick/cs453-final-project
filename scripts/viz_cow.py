'''
Rendrs the 3D cow model from the original paper.
'''
from trimesh import load as load_trimesh
mesh = load_trimesh('./models/cow.obj')
print('Press \'w\' for wireframe.')
mesh.show(smooth=False)
