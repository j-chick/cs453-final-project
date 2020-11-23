'''
TODO
'''
import heapq
import numpy as np
import trimesh

TEST_FILE_PATH = '/Users/jessechick/School/f20/cs453/cs453-final-project/models/cow.obj'
# TEST_FILE_PATH = '/Users/jessechick/School/f20/cs453/cs453-final-project/models/panther.stl'

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
        assert q.shape == (4, 4)
        quadrics[vertex_index] = q

def select_valid_pairs() -> None:
    global valid_pairs

    # NOTE This is the vanilla way to choose valid vertices, according to Garland and Heckbert.
    # TODO Try it out with all vertex combinations beneath the distance threshold `t`.
    valid_pairs = {tuple(edge) for edge in mesh.edges}

def delta(v: np.ndarray, q: np.ndarray) -> float:
    assert isinstance(v, np.ndarray), str(type(v))
    assert v.shape == (4, 1)
    assert isinstance(q, np.ndarray), str(type(v))
    assert q.shape == (4, 4), str(q.shape)
    '''
    TODO
    '''
    error: float

    [[error]] = np.matmul(np.matmul(v.T, q), v)

    assert isinstance(error, float), str(error)
    return error

def compute_contraction_target(edge: tuple) -> tuple: #TODO q_bar type and asserts
    assert isinstance(edge, tuple), str(type(edge))
    assert len(edge) == 2, str(len(edge))
    '''
    See Section 4 of the paper.
    '''
    cost: float
    q_1: np.ndarray = quadrics[edge[0]]
    q_2: np.ndarray = quadrics[edge[1]]
    q_bar: np.ndarray
    v_1: np.ndarray = mesh.vertices[edge[0]]
    v_2: np.ndarray = mesh.vertices[edge[1]]
    v_bar: np.ndarray

    q_bar = q_1 + q_2
    try:
        v_bar = np.matmul(
            np.linalg.inv([
                q_bar[0][:],
                q_bar[1][:],
                q_bar[2][:],
                [0, 0, 0, 1]
            ]), [
                [0],
                [0],
                [0],
                [1]
            ]
        )
    except np.linalg.LinAlgError:
        v_bar = (v_1 + v_2) / 2
        v_bar = np.expand_dims([*v_bar, 1], 0).T
    cost = delta(v_bar, q_bar)
    v_bar = np.squeeze(v_bar[:3])

    return (cost, v_bar)

def garland_heckbert() -> None:
    initialize_quadrics()
    select_valid_pairs()

if __name__ == '__main__':

    garland_heckbert()
    
    import pprint
    pprint.pprint(quadrics[-1])
    pprint.pprint(list(valid_pairs)[-1])

    # NOTE This rentering method may not showcase the triangles in the mesh very well.
    mesh.show()

    target_costs = []
    for i, j in mesh.edges:
        cost, v_bar = compute_contraction_target((i, j))
        target_costs.append(((i, j), v_bar, cost))
    
    # NOTE Triaging edges the easy way, i.e. without a priotiy queue.
    target_costs = sorted(target_costs, key=lambda t: t[2])
    target_costs = target_costs[::2] # NOTE eliminates duplicate edges
    
    for n, ((i, j), target, cost) in enumerate(target_costs, start=1):
        print(
            f'vbar == {target[0]:+.3f}, {target[1]:+.3f}, {target[2]:+.3f}>',
            f'cost == {cost}',
            f'@ {i},{j}',
            sep='\t'
        )
        if n == 32:
            break
