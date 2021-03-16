import numpy as np
import argparse
import h5py
from pathlib import Path
from matplotlib import pyplot as plt


def compute_appearances_sample(sample, N, max_len):
    histogram = np.zeros((max_len+2, N+1))
    parenthesis = [('(', ')'), ('[', ']'), ('{', '}'), ('|', '!'),
                   ('A', 'a'), ('B', 'b'), ('C', 'c'), ('D', 'd')]
    opens = set([p[0] for p in parenthesis])
    closings = {p[1]: i+1 for i, p in enumerate(parenthesis)}
    l = [0]
    i = 0
    while i < len(sample):
        if sample[i] in opens:
            l.append(1)
        elif sample[i] == '.':
            l[-1] += 1
        else:
            l[-1] += 1
            histogram[l[-1], 0] += 1
            histogram[l[-1], closings[sample[i]]] += 1
            last = l.pop()
            l[-1] += last
        i += 1
    return histogram


def main(file, output_image, max_len):
    p = Path(file)
    if not p.exists():
        raise ValueError(f'Invalid file {file}')

    lens = []
    with h5py.File(file, 'r') as f:
        t = f['type'][()]
        p = f['p'][()]
        q = f['q'][()]
        if 'odd' in f.keys():
            odd = f['odd'][()]
        else:
            odd = False
        data = f['data']
        labels = f['labels']
        ts = f["timescales"]
        print('Parameters:', t, p, q)
        histogram = np.zeros((max_len + 2, t + 1))
        print('Found', len(data), 'sequences')
        for i in data:
            if i in {'type', 'q', 'p'}:
                continue
            histogram += compute_appearances_sample(data[str(i)][()], t, max_len)
            lens.append(len(data[(i)][()]))
        total_seqs = len(data)
    print(np.sum(histogram, 0))

    max_len = max(lens)
    print('Max. length', max_len)
    print('Odd', odd)
    print('Total seqs.', total_seqs)
    h = histogram[:, 0]
    max_cut = 10000
    if max_len > max_cut:
        max_len = max_cut
    if odd:
        h = h[2:max_len + 1]
        h = h / np.sum(h)
        ts = np.arange(2, max_len + 1)
    else:
        h = h[2:max_len + 1:2]
        h = h / np.sum(h)  # Normalize
        ts = np.arange(2, max_len + 1, 2)

    plt.figure()
    plt.loglog(ts, h, '-k', label='Dyck2', linewidth=4)
    plt.xlabel('Timescale (T)', fontsize=14)
    plt.ylabel('P(T)', fontsize=14)
    plt.grid()
    plt.legend(fontsize=12)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig(output_image + '_timescales.eps')
    plt.savefig(output_image + '_timescales.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', default='./data/dyckn/train.h5', type=str, help='Input file')
    parser.add_argument('-o', default='./dyck', type=str, help='Output image (prefix)')
    parser.add_argument('-l', default=200, type=int, help='Maximum timescale')
    args = parser.parse_args()
    print(args)
    main(args.i, args.o, args.l)
