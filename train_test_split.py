import sys
import numpy as np
import argparse

p = argparse.ArgumentParser()
p.add_argument("-i", "--infile", help="input data file", type=argparse.FileType('r'), required=True)
p.add_argument("-o", "--outfile", help="output file prefix", type=str, required=True)
p.add_argument("-t", "--test_ratio", help="test ratio (default=0.3)", type=float, nargs='?', default=0.3)
args = p.parse_args()

data = np.genfromtxt(args.infile, dtype=int)
datasize = data[0]
data = data[1:]
test_index = np.random.choice(data.shape[0], size=int(args.test_ratio*data.shape[0]), replace=False)
train_index = np.setdiff1d(np.arange(data.shape[0]), test_index)

with open(args.outfile+'.train', 'w') as f:
    np.savetxt(f, np.vstack([datasize,data[train_index]]), fmt='%.0f')
with open(args.outfile+'.test', 'w') as f:
    np.savetxt(f, data[test_index], fmt='%.0f')
