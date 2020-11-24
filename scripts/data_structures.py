import numpy as np
import trimesh
mesh = trimesh.load('./models/cow.obj')

vertices = [np.array(v) for v in mesh.vertices]
quadrics = [None for _ in range(len(vertices))]

faces = [tuple(v_indices) for v_indices in mesh.faces]
planes = [None for _ in range(len(faces))]

edges = {(i, j) for i, j in mesh.edges if i < j}

def contract_edge(edge: tuple, v_new: np.ndarray) -> None:
    assert isinstance(edge, tuple)
    assert len(edge) == 2
    assert isinstance(v_new, np.ndarray)
    assert v_new.shape == (3,)
    global edges
    global faces
    global vertices
    idx_new: int
    edges_to_add: set
    edges_to_remove: set

    edges.remove(edge)

    vertices[edge[0]] = None
    vertices[edge[1]] = None
    vertices.append(v_new)
    idx_new = len(vertices)

    edges_to_remove = {e for e in edges if e[0] in edge or e[1] in edge}
    edges_to_add = {(
            idx_new if i in edge else i,
            idx_new if j in edge else j
        ) for (i, j) in edges_to_remove
    }
    for edge in edges_to_remove:
        edges.remove(edge)
    for edge in edges_to_add:
        edges.add(edge)


n_edges = len(edges)
# n_faces = len(faces)
n_vertices = len(vertices)
print(f'Initial # of edges: {n_edges}')
print(f'Initial # of vertices: {n_vertices}')

i, j = 1964, 2225
v = (vertices[i] + vertices[j]) / 2
print(f'Contracting edge ({i}, {j})')
contract_edge((i, j), v)

print(f'Final # of edges: {len(edges)}')
print(f'Final # of vertices: {len(vertices)}')

assert n_vertices - len(vertices) == -1, str(n_vertices - len(vertices))
assert n_edges - len(edges) == 3, str(n_edges - len(edges))
