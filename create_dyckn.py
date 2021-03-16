import numpy as np
import argparse
from joblib import Parallel, delayed
import h5py
from pathlib import Path


def dyck1(p, q, max_r=500, max_length=200, min_length=2, odd=False):
    r = np.random.rand(max_r)
    c = 0
    cp = 0
    p_other = p + q + (1.0 - p - q)/2
    seq = 'S'
    idx = seq.find('S')
    while idx >= 0:
        if c >= max_r:
            c = 0
            r = np.random.rand(max_r)
        if r[c] < p:  # S -> (S)
            seq = seq[:idx] + '(S)' + seq[idx + 1:]
            cp += 2
        elif r[c] < p+q:  # S -> SS
            seq = seq[:idx] + 'SS' + seq[idx + 1:]
        else:  # S -> epsilon
            terminal = ''
            if odd and r[c] < p_other:
                terminal = '.'
                cp += 1
            seq = seq[:idx] + terminal + seq[idx + 1:]
        idx = seq.find('S')
        c += 1
        if cp > max_length:
            return ''
    if cp < min_length:
        return ''
    return seq


def dyckn(p, q, max_r=500, max_length=200, min_length=2, odd=False):
    r = np.random.rand(max_r)
    N = len(p)
    parenthesis = [('(', ')'), ('[', ']'), ('{', '}'), ('|', '!'),
                   ('A', 'a'), ('B', 'b'), ('C', 'c'), ('D','d')]
    total_p = np.sum(p)
    cum_p = np.cumsum(p)
    p_other = total_p + q + (1.0 - total_p - q)/2
    c = 0
    cp = 0
    seq = 'S'
    idx = seq.find('S')
    while idx >= 0:
        if c >= max_r:
            c = 0
            r = np.random.rand(max_r)
        if r[c] < total_p:  # S -> (S) add type of parenthesis
            cp += 2
            j = 0
            while j < N and r[c] > cum_p[j]:
                j += 1
            seq = seq[:idx] + parenthesis[j][0] + 'S' + parenthesis[j][1] + seq[idx + 1:]
        elif r[c] < total_p + q:  # S -> SS
            seq = seq[:idx] + 'SS' + seq[idx + 1:]
        else:  # S -> epsilon
            terminal = ''
            if odd and r[c] < p_other:
                terminal = '.'
                cp += 1
            seq = seq[:idx] + terminal + seq[idx + 1:]
        idx = seq.find('S')
        c += 1
        if cp > max_length:
            return ''
    if cp < min_length:
        return ''
    return seq


def compute_dyckn_output(sample):
    output = np.zeros((len(sample),), dtype=np.int)
    parenthesis = [('(', ')'), ('[', ']'), ('{', '}'), ('|', '!'),
                   ('A', 'a'), ('B', 'b'), ('C', 'c'), ('D', 'd')]
    opens = {p[0]: i for i, p in enumerate(parenthesis)}
    l = [0]
    i = 0
    max_depth = 0
    while i < len(sample):
        if sample[i] in opens:
            output[i] = 2**opens[sample[i]]
            l.append(2**opens[sample[i]])
            if max_depth < len(l) - 1:
                max_depth = len(l) - 1
        elif sample[i] == '.':
            output[i] = output[i-1]
        else:
            l.pop()
            output[i] = l[-1]
        i += 1
    return output, max_depth


def compute_timescales(sample):
    output = np.zeros((len(sample),), dtype=np.int)
    parenthesis = [('(', ')'), ('[', ']'), ('{', '}'), ('|', '!'),
                   ('A', 'a'), ('B', 'b'), ('C', 'c'), ('D', 'd')]
    opens = set([p[0] for p in parenthesis])
    # closings = {p[1]: i + 1 for i, p in enumerate(parenthesis)}
    l = [0]
    i = 0
    while i < len(sample):
        if sample[i] in opens:
            l.append(1)
            output[i] = 0
        elif sample[i] == '.':
            output[i] = 0
            l[-1] += 1
        else:
            l[-1] += 1
            output[i] = l[-1]
            last = l.pop()
            l[-1] += last
        i += 1
    return output


def sample_from_pcfg(fn, p, q, samples, jobs, odd, other_set, max_length, min_length):
    results = Parallel(n_jobs=jobs)(delayed(fn)(p, q, max_length=max_length, min_length=min_length, odd=odd) for i in range(min(512, samples)))
    results = set(results) - other_set
    # remove duplicates
    diff = samples - len(results)
    while diff > 0:
        print('Missing: ', diff)
        results_new = Parallel(n_jobs=jobs)(delayed(fn)(p, q, max_length=max_length, min_length=min_length, odd=odd) for i in range(min(512, diff)))
        results_new = set(results_new)
        # remove duplicates
        results = results | (results_new - other_set)
        diff = samples - len(results)
    return results


def save_to_h5(samples, file, type, p, q, odd):
    with h5py.File(file, 'w') as out:
        out['type'] = type
        out['p'] = p
        out['q'] = q
        out['odd'] = odd
        data = out.create_group("data")
        labels = out.create_group("labels")
        depths = out.create_group("depths")
        timescales = out.create_group("timescales")
        max_depth = 0
        for idx, sample in enumerate(samples):
            data[str(idx)] = sample
            target, depth = compute_dyckn_output(sample)
            ts = compute_timescales(sample)
            labels[str(idx)] = target
            depths[str(idx)] = depth
            timescales[str(idx)] = ts
            if depth > max_depth:
                max_depth = depth
        out['max_depth'] = max_depth
    print('Maximum depth:', max_depth)


def divide_samples(samples, n_train, n_validation, n_test):
    total_samples = len(samples)
    if total_samples < n_train + n_test + n_validation:
        raise ValueError('Not enough samples')
    perm = np.random.permutation(total_samples)
    samples_train = [samples[i] for i in perm[:n_train]]
    samples_validation = [samples[i] for i in perm[n_train:n_train+n_validation]]
    samples_test = [samples[i] for i in perm[-n_test:]]
    return samples_train, samples_validation, samples_test


def main(type, p, q, samples_training, samples_validation, samples_test, directory, jobs, odd,
         max_length, min_length):
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    if not d.is_dir():
        raise ValueError(f'Wrong directory: {directory}')

    if type == 1:
        fn = dyck1
    else:
        fn = dyckn
        if len(p) == 1:
            p = [p[0]/type] * type
        if len(p) < type:
            raise ValueError(f'Wrong number of probabilities {len(p)} vs {type}')
    print(p)

    if np.sum(np.array(p)) + q >= 1.0:
        raise ValueError(f"Wrong probabilities {p} and {q}")

    print('Producing samples')
    results = sample_from_pcfg(fn, p, q, samples_training+samples_validation+samples_test,
                               jobs, odd, {''},
                               max_length, min_length)
    results_training, results_validation, results_test = divide_samples(list(results),
                                                                        samples_training,
                                                                        samples_validation,
                                                                        samples_test)

    # save sequences to files
    save_to_h5(list(results_training), directory + '/train.h5', type, p, q, odd)
    save_to_h5(list(results_validation), directory + '/validation.h5', type, p, q, odd)
    save_to_h5(list(results_test), directory + '/test.h5', type, p, q, odd)

    print('Train \n', len(results_training))
    print('Valid \n', len(results_validation))
    print('Test \n', len(results_test))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('type', type=int, default=2)
    parser.add_argument('-p', default=0.5, nargs='+', type=float)
    parser.add_argument('-q', default=0.4, type=float)
    parser.add_argument('--train', default=10000, type=int)
    parser.add_argument('--validation', default=5000, type=int)
    parser.add_argument('--test', default=5000, type=int)
    parser.add_argument('-o', default='./data/dyckn/', type=str, help='Output directory')
    parser.add_argument('--jobs', default=2, type=int)
    parser.add_argument('--odd', action="store_true", help='Add terminal element of length 1')
    parser.add_argument('--max_length', default=100, type=int)
    parser.add_argument('--min_length', default=2, type=int)

    args = parser.parse_args()
    print(args)

    if len(args.p) > 2:
        raise ValueError('Too many values for p')

    main(args.type, args.p, args.q, args.train, args.validation, args.test, args.o, args.jobs, args.odd,
         args.max_length, args.min_length)
