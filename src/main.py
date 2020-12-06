import argparse
import sys

from algorithm import garland_heckbert
from util import load_from_plyfile, jostle_vertices, save_to_plyfile

def main(args: argparse.Namespace) -> None:
    n_vertices: int
    n_faces: int

    vertices, faces = load_from_plyfile(args.input_path)
    if args.jostle_vertices:
        vertices = jostle_vertices(vertices) # NOTE

    n_vertices = len(vertices)
    n_faces = len(faces)
    # print(n_vertices, n_faces, 0)

    vertices, faces = garland_heckbert(vertices, faces, total_contractions=args.n_contractions)

    # print(len(vertices), len(faces), args.n_contractions)
    assert n_vertices + args.n_contractions == len(vertices)
    assert n_faces - 2 * args.n_contractions == len(faces)
    
    save_to_plyfile(args.output_path, vertices, faces)

if __name__ == '__main__':

    parser: argparse.ArgumentParser
    clargs: argparse.Namespace

    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', default='models/icosahedron.ply')
    parser.add_argument('--output-path', default='temp/output.ply')
    parser.add_argument('--n-contractions', type=int, default=1)
    parser.add_argument('--jostle-vertices', action='store_true')
    parser.add_argument('--simple-pair-selection', default=False, action='store_true')
    parser.add_argument('--use-midpoint', default=False, action='store_true')

    clargs = parser.parse_args(sys.argv[1:])
    main(clargs)
