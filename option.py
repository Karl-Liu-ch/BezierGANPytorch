import argparse
parser = argparse.ArgumentParser(description='Optimize')
parser.add_argument('--latent', type=int, default=3, help='latent dimension')
parser.add_argument('--noise', type=int, default=10, help='noise dimension')
parser.add_argument('--n_runs', type=int, default=10, help='number of runs')
parser.add_argument('--n_eval', type=int, default=1000, help='number of total evaluations per run')
parser.add_argument('--method', type=str, default='gan')
args = parser.parse_args()