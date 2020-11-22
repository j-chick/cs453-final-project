'''
TODO
'''
import numpy as np
import trimesh

# TEST_FILE_PATH = '/Users/jessechick/School/f20/cs453/cs453-final-project/models/teapot.ply'
TEST_FILE_PATH = '/Users/jessechick/School/f20/cs453/cs453-final-project/models/panther.stl'

mesh: trimesh.Trimesh = trimesh.load(TEST_FILE_PATH)
quadrics: list = [None for _ in range(len(mesh.vertices))]
valid_pairs: set

def plane(v_1: np.ndarray, v_2: np.ndarray, v_3: np.ndarray) -> np.ndarray:
    for v in {v_1, v_2, v_3}:
        assert isinstance(v, np.ndarray), f'{v}: {type(v)}'
        assert v.shape == (3,), str(v.shape)
    n_arrow = np.cross(v_2 - v_1, v_3 - v_1)
    a, b, c = n_arrow
    d = -np.dot(n_arrow, v_1)
    p = np.array([a, b, c, d])
    assert not all(s == 0.0 for s in p)
    return p

def fundamental_q(p: np.ndarray) -> np.ndarray:
    k_p: np.ndarray
    
    assert isinstance(p, np.ndarray), str(type(p))
    assert p.shape == (4,), str(p.shape)
    p = np.expand_dims(p, -1)
    k_p = np.matmul(p, p.T)
    
    return k_p

def initialize_quadrics() -> None: # REVIEW Refactor; this is messy...
    v_1: np.ndarray
    v_2: np.ndarray
    v_3: np.ndarray
    p: np.ndarray
    q: np.ndarray

    for vertex_index, face_indices in enumerate(mesh.vertex_faces):
        q = np.zeros((4, 4))

        for face_index in filter(lambda i: i != -1, face_indices):
            face = mesh.faces[face_index]
            v_1 = mesh.vertices[face[0]]
            v_2 = mesh.vertices[face[1]]
            v_3 = mesh.vertices[face[2]]

            p = plane(v_1, v_2, v_3)
            k_p = fundamental_q(p)
            q += k_p
        
        quadrics[vertex_index] = q

def select_valid_pairs() -> None:
    global valid_pairs

    # NOTE This is the vanilla way to choose valid vertices, according to Garland and Heckbert.
    # TODO Try it out with all vertex combinations beneath the distance threshold `t`.
    valid_pairs = {tuple(edge) for edge in mesh.edges}

def garland_heckbert() -> None:
    initialize_quadrics()
    select_valid_pairs()

garland_heckbert()

if __name__ == '__main__':
    
    import pprint
    pprint.pprint(quadrics[-1])
    pprint.pprint(list(valid_pairs)[-1])

    # NOTE This rentering method may not showcase the triangles in the mesh very well.
    mesh.show()
