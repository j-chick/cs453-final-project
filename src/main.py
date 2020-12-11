import argparse
import sys

import matplotlib.pyplot as plt

from algorithm import garland_heckbert
from util import load_from_plyfile, jostle_vertices, save_to_plyfile

def main(args: argparse.Namespace) -> None:

    vertices, faces = load_from_plyfile(args.input_path)
    print(
        f'From model {args.input_path}',
        f'\t|V| = {len(vertices)}',
        f'\t|F| = {len(faces)}',
        sep='\n'
    )

    if args.jostle_vertices:
        vertices = jostle_vertices(vertices)
        print('Vertices have been jostled.')

    # n_faces = len(faces)
    vertices, faces = garland_heckbert(vertices, faces, total_contractions=args.n_contractions, use_midpoint=args.use_midpoint)

    print(
        f'To model {args.output_path}',
        f'\t|V| = {len(vertices)}',
        f'\t|F| = {len(faces)}',
        sep='\n'
    )
    save_to_plyfile(args.output_path, vertices, faces)

    # fig, ax1 = plt.subplots()
    # ax1.set_xlabel('n. contractions')

    # contractions = list(range(1, 1 + args.n_contractions))
    # n_faces = list(range(n_faces, n_faces - 2 * args.n_contractions, -2))

    # ax1.set_ylabel('n. faces', color='tab:blue')
    # ax1.plot(contractions, n_faces, color='tab:blue')
    # ax1.tick_params(axis='y', labelcolor='tab:blue')

    # ax2 = ax1.twinx()

    # ax2.set_ylabel('cost', color='tab:red')
    # ax2.plot(contractions, costs, color='tab:red')
    # ax2.tick_params(axis='y', labelcolor='tab:red')

    # fig.tight_layout()
    # plt.show()

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
