import sys
import pickle
import argparse
import numpy as np
from pitf import PITF

def read_data(filepath):
    return np.genfromtxt(filepath, delimiter=' ', dtype=int)

p = argparse.ArgumentParser()
p.add_argument("-i", "--infile", help="input data file", type=argparse.FileType('r'), required=True)
p.add_argument("-m", "--modelfile", help="input model file", type=argparse.FileType('r'), required=True)
p.add_argument("-o", "--outfile", help="output file (default=STDOUT)", type=argparse.FileType('w'), nargs='?', default=sys.stdout)
args = p.parse_args()

data = read_data(args.infile)
model = pickle.load(args.modelfile)

predicted = model.predict2(data)
np.savetxt(args.outfile, predicted, fmt='%.0f')
