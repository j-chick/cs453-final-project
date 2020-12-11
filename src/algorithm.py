'''
TODO
'''
# TODO import heapq
import numpy as np
from util import assert_correctness, NULL_VERTEX

quadric_matrices: dict = {}

def select_valid_contraction_pairs(faces: list, t: float = None) -> list:
    edges: set = set()
    non_adjacent: set

    for face in faces:
        [i, j, k] = sorted(face)
        edges.update({(i, j), (j, k), (i, k)})
    if t is None:
        non_adjacent = []
    else:
        raise NotImplementedError

    assert not any((j, i) in edges for (i, j) in edges)
    assert not any((j, i) in non_adjacent for (i, j) in non_adjacent)
    
    return list(edges)

def planes(vertices: list, faces: list, v_index: int) -> set:
    a: float
    b: float
    c: float
    d: float
    n: np.ndarray
    p: tuple
    p_set: set = set()
    v_1: np.ndarray
    v_2: np.ndarray
    v_3: np.ndarray

    assert 0 <= v_index < len(vertices)

    for i, j, k in {tuple(f) for f in faces if v_index in f}:
        v_1 = vertices[i]
        v_2 = vertices[j]
        v_3 = vertices[k]
        n = np.cross(v_2 - v_1, v_3 - v_1)
        a, b, c = n
        d = -np.dot(n, v_1)
        p = (a, b, c, d)
        p_set.add(p)
    
    return p_set

def compute_quadric_matrix(vertices: list, faces: list, v_index: int) -> np.ndarray:
    '''
    TODO
    '''
    q: np.ndarray = np.zeros((4, 4))

    for p in planes(vertices, faces, v_index):
        p = np.expand_dims(p, 1)
        k_p = np.matmul(p, p.T)
        q += k_p
    
    return q

def compute_replacement_vertex(v_1: np.ndarray, v_2: np.ndarray, q_1: np.ndarray, q_2: np.ndarray) -> np.ndarray:
    qbar: np.ndarray

    # TODO assert

    qbar = q_1 + q_2
    try:
        vbar = np.matmul(
            np.linalg.inv([
                qbar[0][:],
                qbar[1][:],
                qbar[2][:],
                [0, 0, 0, 1]
            ]), [
                [0],
                [0],
                [0],
                [1]
            ]
        )
        vbar = np.squeeze(vbar[:3])
    except np.linalg.LinAlgError:
        # print('Using suboptimal vbar.')
        # NOTE advised in the paper
        vbar = (v_1 + v_2) / 2
    
    return vbar

def compute_contraction_cost(vbar: np.ndarray, q_1: np.ndarray, q_2: np.ndarray) -> float:
    delta: callable

    delta = lambda v: float(np.matmul(np.matmul(v.T, q_1 + q_2), v))

    return delta(np.expand_dims([*vbar, 1], 1))

def perform_contraction(v: list, faces: list, pair: tuple, v_replacement: np.ndarray) -> tuple:
    i: int = pair[0]
    j: int = pair[1]

    assert isinstance(v, list)
    assert isinstance(faces, list) and all(len(face) == 3 for face in faces)
    assert isinstance(i, int) and 0 <= i < len(v), str(i)
    assert isinstance(j, int) and 0 <= j < len(v), str(j)
    assert isinstance(v_replacement, np.ndarray) and v_replacement.shape == (3,), str(v_replacement)

    assert v[i] is not NULL_VERTEX, str(pair)
    assert v[j] is not NULL_VERTEX, str(pair)
    v[i] = NULL_VERTEX
    v[j] = NULL_VERTEX
    v.append(v_replacement)
    # n_faces = len(faces)
    for i_faces in range(len(faces)):
        for i_face in range(3):
            if faces[i_faces][i_face] in {i, j}:
                assert faces[i_faces][i_face] != len(v) - 1
                faces[i_faces][i_face] = len(v) - 1
    faces = [face for face in faces if len(set(face)) == 3]

    # assert len(faces) - n_faces == -2
    assert all(len(face) == 3 for face in faces)
    assert not any(i in face or j in face for face in faces)
    # assert len(v) == len(np.unique(faces))
    
    return v, faces

def update_contraction_queue(heap: list, v: list, faces: list) -> list:
    idx: int = 0

    assert not any(len(v) - 1 in pair for _, pair, _ in heap)

    while idx < len(heap):
        _, (i, j), _ = heap[idx]
        assert i != j
        if v[i] is NULL_VERTEX or v[j] is NULL_VERTEX:
            if v[i] is NULL_VERTEX:
                i = len(v) - 1
            if v[j] is NULL_VERTEX:
                j = len(v) - 1
            if i == j:
                _ = heap.pop(idx)
                continue
            q_i = compute_quadric_matrix(v, faces, i)
            q_j = compute_quadric_matrix(v, faces, j)
            vbar = compute_replacement_vertex(v[i], v[j], q_i, q_j)
            cost = compute_contraction_cost(vbar, q_i, q_j)
            heap[idx] = (cost, (i, j), vbar)
        idx += 1
    heap = sorted(heap, key=lambda v: v[0], reverse=True)
    # assert isinstance(heap, list) and all(isinstance(cost, float) and isinstance(pair, tuple) and isinstance(vbar, np.ndarray) for cost, pair, vbar in heap)
    # heapq.heapify(heap)
    
    assert all(all(v[index] is not NULL_VERTEX for index in pair) for _, pair, _ in heap)

    return heap

def plot_costs(v: list, faces: list, costs: list) -> None:
    raise NotImplementedError

def garland_heckbert(v: list, faces: list, total_contractions: int = 1, use_midpoint: bool = False) -> tuple:
    queue: list = []
    valid_contraction_pairs: list
    # contraction_costs: list = []

    assert all(v_i.shape == (3,) for v_i in v)
    assert all(all(isinstance(c, float) for c in v_i) for v_i in v)
    assert all(isinstance(face, list) for face in faces)
    assert all(all(isinstance(idx, int) for idx in face) for face in faces), str(type(faces[0][0]))
    assert all(all(0 <= idx < len(v) for idx in face) for face in faces)

    valid_contraction_pairs = select_valid_contraction_pairs(faces)
    print(f'Selected {len(valid_contraction_pairs)} candidate contraction pairs.')

    for i, j in valid_contraction_pairs:
        q_i = compute_quadric_matrix(v, faces, i)
        q_j = compute_quadric_matrix(v, faces, j)
        if use_midpoint:
            vbar = (v[i] + v[j]) / 2
        else:
            vbar = compute_replacement_vertex(v[i], v[j], q_i, q_j)
        cost = compute_contraction_cost(vbar, q_i, q_j)
        assert isinstance(cost, float)
        queue.append((cost, (i, j), vbar))

    queue = sorted(queue, key=lambda v: v[0], reverse=True)
    # heapq.heapify(queue)

    for _ in range(total_contractions):
        cost, pair, vbar = queue.pop()
        # _, pair, vbar = heapq.heappop(queue)
        v, faces = perform_contraction(v, faces, pair, vbar)
        queue = update_contraction_queue(queue, v, faces)
        assert_correctness(v, faces)
        # contraction_costs.append(cost)

    assert all(v_i.shape == (3,) for v_i in v)
    assert all(all(isinstance(c, float) for c in v_i) for v_i in v)
    assert all(isinstance(face, list) for face in faces)
    assert all(all(isinstance(idx, int) for idx in face) for face in faces), str(type(faces[0][0]))
    assert all(all(0 <= idx < len(v) for idx in face) for face in faces)

    return v, faces # , contraction_costs

# def approximate_error(m_n: dict, m_i: dict)
