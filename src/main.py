import argparse
import random
import sys

from algorithm import jostle_vertices, garland_heckbert
from util.ply import PlyFile

def main(args: argparse.Namespace) -> None:
    vertices, faces = PlyFile.load(args.obj_path)
    # vertices = jostle_vertices(vertices) # NOTE

    n_vertices: int = len(vertices)
    n_faces: int = len(faces)
    print(n_vertices, n_faces, 0)

    vertices, faces = garland_heckbert(vertices, faces, total_contractions=args.n_contractions)

    print(len(vertices), len(faces), args.n_contractions)
    assert n_vertices + args.n_contractions == len(vertices) # REVIEW
    #assert n_faces - 2 * args.n_contractions == len(faces) # REVIEW
    
    PlyFile.save(vertices, faces, 'temp.ply')

if __name__ == '__main__':

    parser: argparse.ArgumentParser
    clargs: argparse.Namespace

    parser = argparse.ArgumentParser()
    parser.add_argument('obj_path', default='models/cow.obj')
    parser.add_argument('--simple-pair-selection', action='store_true')
    parser.add_argument('--use-midpoint', action='store_true')
    parser.add_argument('--n-contractions', type=int, default=1)

    clargs = parser.parse_args(sys.argv[1:])
    main(clargs)
