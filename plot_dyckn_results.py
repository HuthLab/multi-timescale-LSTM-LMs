import numpy as np
from matplotlib.pyplot import *
import argparse


def load_summary(summary_file):
    f = np.load(summary_file, allow_pickle=True)
    lengths = np.array(f['summary'][()]['lengths'])
    ts = f['summary'][()]['timescales']
    metric = np.array(f['summary'][()]['metric'])
    max_len = np.max(lengths)

    print(np.mean(metric))

    max_ts = np.array([t.max() for t in ts])
    print(max_ts)
    print(max_len)
    bins = np.linspace(2, max_len, num=10)
    return lengths, ts, metric, max_len, max_ts, bins

# In order to generate this plot, you will need to run each model 20 times (with seeds 1 to 20).
# Use the "-o" option of the run_dyckn.py script to generate the right folder for each run.
# Once the models finish, this script will read all the results and will generate the timescales plot.

parser = argparse.ArgumentParser()
parser.add_argument('-b', default='./results/dyckn/Baseline_u256_l1_e2000_b32_s200_lr0.0001', type=str, help='Prefix (not incl. seed number) of Baseline model results')
parser.add_argument('-m', default='./results/dyckn/MTS_u256_l1_e2000_b32_s200_lr0.0001_sc1.00_a1.50', type=str, help='Prefix (not incl. seed number) of MTS model results')
args = parser.parse_args()
print(args)

baseline_prefix = args.b
mts_prefix = args.m

# Original run Baseline_u256_l1_e2000_b32_s200_lr0.0001_seed1
first_seed = 1
last_seed = 20
summary_group_1 = [f'{baseline_prefix}_seed{i}/summary.npz' for i in range(first_seed, last_seed+1)]
figure()
curves1 = []
for file in summary_group_1:
    print(file)
    lengths, ts, metric1, max_len, max_ts, bins = load_summary(file)
    vals1, _, _ = hist(max_ts[metric1], bins=bins, color='blue')
    vals3, _, _ = hist(max_ts[~metric1], bins=bins, color='cyan')
    curves1.append(vals1 / (vals1 + vals3))

first_seed = 1
last_seed = 20
summary_group_2 = [f'{mts_prefix}_seed{i}/summary.npz' for i in range(first_seed, last_seed+1)]
figure()
curves2 = []
for file in summary_group_2:
    print(file)
    lengths, ts, metric2, max_len, max_ts, bins = load_summary(file)
    vals2, _, _ = hist(max_ts[metric2], bins=bins, color='red')
    vals4, _, _ = hist(max_ts[~metric2], bins=bins, color='orange')
    curves2.append(vals2 / (vals2 + vals4))


baseline_curve = np.array(curves1).T
# curves = [curves2[i] + np.random.randn(curves2[i].shape[0])/10 for i in range(len(curves2))]
mts_curve = np.array(curves2).T

print(baseline_curve.shape)
figure()
plot(bins[1:], np.mean(baseline_curve,1), color='blue', label='Baseline', linewidth=4)
n=1 # std dev
fill_between(bins[1:], np.mean(baseline_curve,1)-np.std(baseline_curve,1)/np.sqrt(n), np.mean(baseline_curve,1)+np.std(baseline_curve,1)/np.sqrt(n),
             facecolor='purple', alpha=0.2)

plot(bins[1:], np.mean(mts_curve,1), color='red', label='Multi-timescale', linewidth=4)
n=1 # std dev
fill_between(bins[1:], np.mean(mts_curve,1)-np.std(mts_curve,1)/np.sqrt(n), np.mean(mts_curve,1)+np.std(mts_curve,1)/np.sqrt(n),
             facecolor='magenta', alpha=0.2)
grid()
legend(fontsize=12)
ylim([0., 1.05])
xlim([10, max(bins)])
xticks(fontsize=14)
yticks(fontsize=14)
ylabel('% Correct', fontsize=14)
xlabel('Max. Timescale in Sequence', fontsize=14)
tight_layout()
savefig('formal_timescales_histogram_variance.png')
savefig('formal_timescales_histogram_variance.eps')
