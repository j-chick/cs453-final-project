'''
TODO
'''
import numpy as np
import trimesh
from itertools import combinations
#from plyfile import PlyData, PlyElement

#TEST_FILE_PATH: str = '/Users/jessechick/School/f20/cs453/cs453-final-project/models/cow.obj'
# TEST_FILE_PATH: str = '/Users/jessechick/School/f20/cs453/cs453-final-project/models/panther.stl'
TEST_FILE_PATH: str = '/Users/odysseywilson/cs453-final-project/models/panther.stl'

mesh: trimesh.Trimesh = trimesh.load(TEST_FILE_PATH)
valid_pairs: set

vertices = [np.array(v) for v in mesh.vertices]
quadrics = [None for _ in range(len(vertices))]

faces = [tuple(v_indices) for v_indices in mesh.faces]
planes = [None for _ in range(len(faces))]

edges = {(i, j) for i, j in mesh.edges if i < j}

'''
def write_to_plyfile(v: np.ndarray, f: np.ndarray, save_path: str, v_contracted: tuple = None, vbar: np.ndarray = None) -> None:
    # REVIEW https://github.com/dranjan/python-plyfile#creating-a-ply-file
    
    red: list
    green: list

    if v_contracted is None:
        red = []
    else:
        assert all(v_.shape == (3,) for v_ in v_contracted)
        red = [(*v_, (255, 0, 0)) for v_ in v_contracted]
    
    if vbar is None:
        green = []
    else:
        assert vbar.shape == (3,)
        v_1, v_2 = v_contracted
        green = [(*vbar, (0, 255, 0)), (*((v_1 + v_2) / 2), (127, 255, 0))]

    vertex = np.array(
        [(*v_i, (255, 255, 255)) for v_i in map(list, v)] + red + green,
        dtype=[('x', 'f8'), ('y', 'f8'), ('z', 'f8'), ('rgb', 'u1', (3,))]
    )
    face = np.array(
        [(list(f_[:3]),) for f_ in f],
        dtype=[('vertex_index', 'i4', (3,))]
    )
    vertex_element = PlyElement.describe(vertex, 'vertex', val_types={'x': 'float64', 'y': 'float64', 'z': 'float64', 'r': 'uchar', 'g': 'uchar', 'b': 'uchar'})
    face_element = PlyElement.describe(face, 'face', val_types={'vertex_index': 'int32'}, len_types={'vertex_index': 'u4'})
    PlyData([vertex_element, face_element], text=True).write(save_path)
'''
#Given three vertices in a face, determine its plane [a, b, c, d] where ax + by + cz + d = 0 and a^2 + b^2 + c^2 = 1

def plane(v_1: np.ndarray, v_2: np.ndarray, v_3: np.ndarray) -> np.ndarray:
    line1 = np.ndarray((3,))  # Line from v_1 to v_2
    line2 = np.ndarray((3,))  # Line from v_1 to v_3

    for i in range(3):
        line1[i] = v_2[i] - v_1[i]
        line2[i] = v_3[i] - v_1[i]

    normal_vector = np.ndarray((3,))  # Vector normal to the plane
    normal_vector[0] = (line1[1] * line2[2]) - (line1[2] * line2[1])
    normal_vector[1] = ((line1[0] * line2[2]) - (line1[2] * line2[0])) * -1
    normal_vector[2] = (line1[0] * line2[1]) - (line1[1] * line2[0])

    p = np.ndarray((4,))  # plane [a, b, c, d] where ax + by + cz + d = 0
    for i in range(3):
        p[i] = normal_vector[i]

    p[3] = (normal_vector[0] * v_1[0] * -1) + (normal_vector[1]
            * v_1[1] * -1) + (normal_vector[2] * v_1[2] * -1)

    return p

#Given a plane, determine k_p
def fundamental_q(p: np.ndarray) -> np.ndarray:
    k_p = np.ndarray((4, 4))

    for i in range(4):
        k_p[0, i] = p[0] * p[i]
        k_p[1, i] = p[1] * p[i]
        k_p[2, i] = p[2] * p[i]
        k_p[3, i] = p[3] * p[i]

    return k_p

#Compute the initial Q matrix for each initial vertex
def initialize_quadrics() -> None:
    q = np.ndarray((4, 4))

    for i in range(len(vertices)):  # for each vertex
        quadrics[i] = np.ndarray((4, 4))
        quadrics[i].fill(0)
        v_faces = []
        for j in range(len(faces)):  # for each face
            for k in range(3):  # for each vertex in the face
                # if the index of the vertex matches the index of the vertex in the face, store the face's index
                if i == faces[j][k]:
                    v_faces.append(j)
        for l in range(len(v_faces)):
            face_index = v_faces[l]
            p = plane(vertices[faces[face_index][0]], vertices[faces[face_index][1]], vertices[faces[face_index][2]])
            q = fundamental_q(p)
            quadrics[i] += q

#Compute the error delta v given the vertex and the quadric
def delta(v: np.ndarray, q: np.ndarray) -> float:
    error: float

    #v = np.append(v, 1) I don't think this is needed since compute_contraction_target already adds the 1
    v_row = v.reshape(1, -1)
    v_col = v_row.T
    
    error = np.matmul(v, np.matmul(q, v_col))
    return error

def select_valid_pairs() -> None:
    '''
    TODO
    '''
    global valid_pairs

    # NOTE This is the vanilla way to choose valid vertices, according to Garland and Heckbert.
    # TODO Try it out with all vertex combinations beneath the distance threshold `t`.
    valid_pairs = {tuple(edge) for edge in mesh.edges}


# Returns updated quadric for when a pair is contracted given the indices of the two vertices being contracted
def update_quadric(i1: int, i2: int) -> np.ndarray:
    q_bar = np.ndarray((4,4))
    q1 = quadrics[i1]
    q2 = quadrics[i2]

    q_bar = np.add(q1, q2)
    return q_bar

# Given two indices of vertices, returns cost and v_bar
def compute_contraction_target(edge: tuple) -> tuple:
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

    assert isinstance(edge, tuple), str(type(edge))
    assert len(edge) == 2, str(len(edge))

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
        v_bar = v_bar.T
        v_bar = v_bar[0] #Remove extra dimension we had to add to transpose
    except np.linalg.LinAlgError:
        v_bar = (v_1 + v_2) / 2
        #v_bar = np.expand_dims([*v_bar, 1], 0).T
    cost = delta(v_bar, q_bar)
    v_bar = np.squeeze(v_bar[:3])

    return (cost, v_bar)

def garland_heckbert(mode: int, threshold: int) -> None:
    select_valid_pairs()
    initialize_quadrics()

    target_costs = []
    for i, j in valid_pairs:
        cost, v_bar = compute_contraction_target((i, j))
        target_costs.append(((i, j), v_bar, cost))

    target_costs = sorted(target_costs, key=lambda t: t[2]) #Sorted list of (i,j), v_bar, cost by cost
    target_costs = target_costs[::2] #Remove duplicate edges

    if mode == 2:
        for x in range(5): #EDIT: change to condition for num faces
            # Pop off and extract from target_costs
            top_of_queue = target_costs.pop(0)
            i_index = top_of_queue[0][0] 
            j_index = top_of_queue[0][1]
            i = vertices[i_index]
            j = vertices[j_index]
            v_bar = top_of_queue[1]
            cost = top_of_queue[2][0]

            # Remove i, j from vertices
            vertices[i_index] = None
            vertices[j_index] = None

            # Add v_bar to vertices and calculate error/quadrics for it, remove old error quadrics
            vertices.append(v_bar)
            quadrics.append(quadrics[i_index] + quadrics[j_index])
            quadrics[i_index] = None
            quadrics[j_index] = None

            # Any faces that use i or j now use v_bar
            for y in range(len(faces)):
                for z in range(3):
                    print(type(faces[y][z]))
                    print(type(i_index))
                    if faces[y][z] == i_index or faces[y][z] == j_index:
                        # faces[y][z] = vertices[len(vertices) - 1]
                        if z == 0:
                            faces[y] = (vertices[len(vertices) - 1], faces[y][1], faces[y][2])
                        if z == 1:
                            faces[y] = (faces[y][0], vertices[len(vertices) - 1], faces[y][2])
                        if z == 2:
                            faces[y] = (faces[y][0], faces[y][1], vertices[len(vertices) - 1])

            # For each entry in target_costs, if the contraction involves i or j, update to v_bar, recalculate error, and resort list
            to_add_to_target_costs = []
            to_remove_from_target_costs = []
            for y,t in enumerate(target_costs):
                if t[0][0] == i_index or t[0][0] == j_index:
                    to_remove_from_target_costs.append(x)
                    new_cost, new_v_bar = compute_contraction_target((len(vertices) - 1, t[0][1]))
                    to_add_to_target_costs.append(((len(vertices) - 1, t[0][1]), new_v_bar, new_cost))


                if t[0][1] == i_index or t[0][1] == j_index:
                    to_remove_from_target_costs.append(x)
                    new_cost, new_v_bar = compute_contraction_target((t[0][0], len(vertices) - 2))
                    to_add_to_target_costs.append(((t[0][0], len(vertices) - 1), new_v_bar, new_cost))

            for y in to_remove_from_target_costs:
                target_costs.pop(y)
            for y in to_add_to_target_costs:
                target_costs.append(y)

            target_costs = sorted(target_costs, key=lambda t: t[2]) 
        print(target_costs)
        '''
            pop off target_costs
            contract i, j from target costs
                replace i, j with None in vertices
                put v_bar at end of vertices
                calculate quadrics for v_bar at end of vertices, store in end of quadrics
                replace i, j in edges with v_bar
                    for these edges, update cost of contracting that edge
        '''
''' Print target_costs info
    for n, ((i, j), target, cost) in enumerate(target_costs, start=1):
        # cost_normed = (cost - target_costs[0][2]) / (target_costs[-1][2] - target_costs[0][2])
        print(
            f'vbar == <{target[0]:+.3f}, {target[1]:+.3f}, {target[2]:+.3f}>',
            f'normed cost == {cost}',
            f'@ {i},{j}',
            sep='\t'
        )
'''

if __name__ == '__main__':
    #Note: Use either threshold parameter t or desired number of faces
    t = .5 #Threshold parameter for removing edges
    n = 5 #Number of faces desired in approximation

    #Mode 1 uses threshold t, mode 2 uses num faces n
    garland_heckbert(2, n)




    #mesh.show(smooth=False)
