import re
import numpy as np
from numpy.core.defchararray import split
from plyfile import PlyData, PlyElement

"""
TODO Citrine
Hex #E4D00A
RGB 228, 208, 10
"""

class PlyFile:
    @staticmethod
    def load(path: str) -> tuple:
        vertices: list = []
        faces: list = []

        with open(path, 'r') as ply:
            for line in ply.readlines()[1:]:
                if re.match(r'^3 [0-9]+ [0-9]+ [0-9]+', line):
                    assert len(line.split(' ')) == 4
                    faces += [tuple(map(int, line.split(' ')[1:]))]
                elif re.match(r'^-?[0-9.]+ -?[0-9.]+ -?[0-9.]+', line):
                    assert len(line.split(' ')) == 3, line
                    vertices += [np.array(list(map(float, line.split(' '))))]

        # FIXME
        # assert len(vertices) == 12
        # assert len(faces) == 20

        return vertices, faces

    @staticmethod
    def save(v: list, f: list, path: str) -> None:
        assert isinstance(v, list)
        assert isinstance(f, list)
        assert isinstance(path, str)
        assert all(v_i.shape == (3,) for v_i in v)
        assert all(isinstance(face, tuple) and len(face) == 3 for face in f)

        with open(path, 'w') as ply:
            ply.writelines([
                'ply\n', 
                'format ascii 1.0\n',
                f'element vertex {len(v)}\n',
                'property float x\n',
                'property float y\n',
                'property float z\n',
                f'element face {len(f)}\n',
                'property list uchar int vertex_indices\n',
                # 'property uchar red\n', REVIEW
                # 'property uchar green\n',
                # 'property uchar blue\n',
                'end_header\n'
            ])
            ply.writelines([
                ' '.join(
                    f'{component:.06f}' if component != 0.0 else '0' 
                        for component in v_i
                ) + '\n' 
                    for v_i in v
            ])
            ply.writelines([
                '3 ' + ' '.join(
                    str(index)
                        for index in face
                ) + '\n'
                    for face in f
            ])
