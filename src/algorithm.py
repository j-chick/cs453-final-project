import heapq
import numpy as np

dead_vertices: set = set()

def jostle_vertices(v: list) -> list:
    v_jostled: list = []

    assert all(v_i.shape == (3,) for v_i in v)

    for v_i in v:
        v_jostled += [np.array(
            [component + np.random.choice([-0.01, 0.01]) for component in v_i]
        )]

    return v_jostled

def select_valid_contraction_pairs2(faces: list) -> list:
    pairs: list = []
    idx: int = 0

    for i, j, k in faces:
        pairs += [(i, j), (i, k), (j, k)]
    while idx < len(pairs):
        i, j = pairs[idx]

        idx += 1

def select_valid_contraction_pairs(faces: list) -> list:
    pairs: set

    pairs = set()
    for face in faces:
        [i, j, k] = sorted(face)
        pairs.update({(i, j), (i, k), (j, k)})

    assert not any((j, i) in pairs for (i, j) in pairs)
    
    return list(pairs)

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

    # assert TODO

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

    n_vertices = len(v)
    # _ = int(v[i][0]), int(v[j][0]) # FIXME
    # v[i] = np.empty((3,)) NOTE
    # v[j] = np.empty((3,)) NOTE
    v.append(v_replacement)
    n_faces = len(faces)
    for i_faces in range(n_faces):
        idx_1, idx_2, idx_3 = faces[i_faces]
        faces[i_faces] = (
            len(v) - 1 if idx_1 in {i, j} else idx_1,
            len(v) - 1 if idx_2 in {i, j} else idx_2,
            len(v) - 1 if idx_3 in {i, j} else idx_3
        )
    faces = [(a, b, c) for a, b, c in faces if a != b and b != c and c != a]

    assert len(v) - n_vertices == 1
    # assert len(faces) - n_faces == -2
    assert all(len(face) == 3 for face in faces)
    assert not any(i in face or j in face for face in faces)
    
    return v, faces



def update_contraction_costs(heap: list, v: list, faces: list) -> list:
    contracted_pair = heap.pop(0)
    i: int = contracted_pair[0]
    j: int = contracted_pair[1]

    assert not any(i in face or j in face for face in faces)
    assert not any(len(v) - 1 in pair for _, pair, _ in heap)

    for idx in range(len(heap)):
        _, pair, _ = heap[idx]
        if i in pair or j in pair:
            (idx_1, idx_2) = pair
            if idx_1 == i or idx_1 == j:
                idx_1 = len(v) - 1
            elif idx_2 == i or idx_2 == j:
                idx_2 = len(v) - 1
            else:
                raise ValueError
            assert idx_1 != idx_2
            q_i = compute_quadric_matrix(v, faces, idx_1)
            q_j = compute_quadric_matrix(v, faces, idx_2)
            vbar = compute_replacement_vertex(v[idx_1], v[idx_2], q_i, q_j)
            cost = compute_contraction_cost(vbar, q_i, q_j)
            heap[idx] = (cost, (idx_1, idx_2), vbar)

    assert not any(i in pair or j in pair for _, pair, _ in heap)

    return heap

def garland_heckbert(v: list, faces: list, total_contractions: int = 1, use_midpoint: bool = False) -> tuple:
    queue: list = []
    valid_contraction_pairs: list

    assert all(v_i.shape == (3,) for v_i in v)
    assert all(all(isinstance(component, float) for component in v_i) for v_i in v)
    assert all(isinstance(face, tuple) for face in faces)
    assert all(all(isinstance(idx, int) for idx in face) for face in faces), str(type(faces[0][0]))
    assert all(all(0 <= idx < len(v) for idx in face) for face in faces)

    valid_contraction_pairs = select_valid_contraction_pairs(faces)
    print(f'Selected {len(valid_contraction_pairs)} candidate contraction pairs.')

    for i, j in valid_contraction_pairs:
        q_i = compute_quadric_matrix(v, faces, i)
        q_j = compute_quadric_matrix(v, faces, j)
        vbar = compute_replacement_vertex(v[i], v[j], q_i, q_j)
        cost = compute_contraction_cost(vbar, q_i, q_j)
        queue.append((cost, (i, j), vbar))

    # heapq.heapify(queue) NOTE
    queue = sorted(queue, key=lambda v: v[0])

    # while len(faces) > 100:
    for n in range(total_contractions):
        # _, pair, vbar = heapq.heappop(queue) NOTE
        _, pair, vbar = queue[0]
        # if pair[0] in dead_vertices or pair[1] in dead_vertices:
        #     continue
        print(f'Contracting {pair}')
        v, faces = perform_contraction(v, faces, pair, vbar)
        # dead_vertices.update(set(pair)) # FIXME
        queue = update_contraction_costs(queue, v, faces)
        
        # heapq.heapify(queue) NOTE
        queue = sorted(queue, key=lambda v: v[0])

        # print(f' {n}', end='\r')

    assert all(v_i.shape == (3,) for v_i in v)
    assert all(all(isinstance(component, float) for component in v_i) for v_i in v)
    assert all(isinstance(face, tuple) for face in faces)
    assert all(all(isinstance(idx, int) for idx in face) for face in faces), str(type(faces[0][0]))
    assert all(all(0 <= idx < len(v) for idx in face) for face in faces)

    return v, faces
