import sys
import argparse
import numpy as np

def read_data(filepath):
    return np.genfromtxt(filepath, delimiter=' ', dtype=int)

p = argparse.ArgumentParser()
p.add_argument("-a", "--answerfile", help="answer file (ground truth)", type=argparse.FileType('r'), required=True)
p.add_argument("-r", "--resultfile", help="result file", type=argparse.FileType('r'), required=True)
args = p.parse_args()

answer = read_data(args.answerfile)
predicted = read_data(args.resultfile)

print (answer==predicted).mean()
