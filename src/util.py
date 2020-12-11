'''
TODO docstring
'''
import os
import re
import numpy as np

NULL_COORDINATE: float = round(np.sqrt(2), 6)
NULL_VERTEX: np.ndarray = np.array([NULL_COORDINATE] * 3)

# SECTION .ply file
def load_from_plyfile(path: str) -> tuple:
    '''
    TODO docstring
    '''
    vertices: list = []
    faces: list = []

    with open(path, 'r') as ply:
        for line in ply.readlines()[1:]:
            if re.match(r'^3 [0-9]+ [0-9]+ [0-9]+', line):
                faces += [list(map(int, line.split(' ')[1:]))]
            elif re.match(r'^-?[0-9.]+ -?[0-9.]+ -?[0-9.]+', line):
                vertices += [np.array(list(map(float, line.split(' '))))]

    assert_correctness(vertices, faces)

    return vertices, faces

def save_to_plyfile(path: str, v: list, f: list) -> None:
    '''
    TODO docstring
    '''
    assert isinstance(v, list)
    assert isinstance(f, list)
    assert isinstance(path, str)
    assert all(v_i.shape == (3,) for v_i in v)
    assert all(isinstance(face, list) and len(face) == 3 for face in f)

    os.makedirs(os.path.split(path)[0], exist_ok=True)
    with open(path, 'w+') as ply:
        ply.writelines([
            'ply\n',
            'format ascii 1.0\n',
            f'element vertex {len(v)}\n',
            'property float x\n',
            'property float y\n',
            'property float z\n',
            f'element face {len(f)}\n',
            'property list uchar int vertex_indices\n',
            'end_header\n'
        ])
        ply.writelines([
            ' '.join('0' if c == 0.0 else f'{c:.06f}' for c in v_i) + '\n' for v_i in v
        ])
        ply.writelines([
            '3 ' + ' '.join(str(index) for index in face) + '\n' for face in f
        ])
# !SECTION .ply file I/O

# SECTION Validation
def assert_correctness(v: list, faces: list) -> None:
    '''
    TODO docstring
    '''
    assert isinstance(v, list)
    assert len(v) >= 4
    assert all(v_i.shape == (3,) for v_i in v)

    assert isinstance(faces, list)
    assert len(faces) >= 4
    assert all(isinstance(face, list) for face in faces)
    assert all(len(face) == 3 for face in faces)
    assert all(all(v[i] is not NULL_VERTEX for i in face) for face in faces)

    # occurrences: dict = {i: 0 for i, _ in enumerate(v)}
    # for face in faces:
    #     for i in face:
    #         occurrences[i] += 1
    # for i, n in occurrences.items():
    #     assert n == 0 or 3 <= n, f'{i} {n}'

# !SECTION Validation

# SECTION Miscellaneous
def jostle_vertices(v: list) -> list:
    '''
    TODO docstring
    '''
    v_jostled: list = []

    assert all(v_i.shape == (3,) for v_i in v)

    for v_i in v:
        v_jostled += [np.array([
            component + np.random.choice([-0.01, 0.01]) for component in v_i
        ])]

    return v_jostled
# SECTION Miscellaneous
