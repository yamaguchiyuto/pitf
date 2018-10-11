Pairwise Interaction Tensor Factorization

## Example

```
% python train.py -i sample.train -k 32 -a 0.0001 -l 0.1 -m 100 -o sample.model
% python predict.py -i sample.test -m sample.model -o sample.result
% python evaluate.py -a sample.answer -r sample.result
```

## Usage

```
% python train.py -h
usage: train.py [-h] -i INFILE -o OUTFILE [-a [ALPHA]] [-l [LAMB]] [-k [K]]
                [-m [MAX_ITER]] [-v]

optional arguments:
  -h, --help            show this help message and exit
  -i INFILE, --infile INFILE
                        input tensor file
  -o OUTFILE, --outfile OUTFILE
                        output model file
  -a [ALPHA], --alpha [ALPHA]
                        alpha (default=0.05)
  -l [LAMB], --lamb [LAMB]
                        lambda (default=0.00005)
  -k [K]                k (default=10)
  -m [MAX_ITER], --max_iter [MAX_ITER]
                        max_iter (default=100)
  -v, --verbose         verbosity

% python predict.py -h
usage: predict.py [-h] -i INFILE -m MODELFILE [-o [OUTFILE]]

optional arguments:
  -h, --help            show this help message and exit
  -i INFILE, --infile INFILE
                        input data file
  -m MODELFILE, --modelfile MODELFILE
                        input model file
  -o [OUTFILE], --outfile [OUTFILE]
                        output file (default=STDOUT)
```

## Input data

```
% head -n10 sample.train
147541 147541 50 # data size for each mode
0 1 0
4 5 2
6 7 3
8 9 4
14 15 7
16 17 8
16 18 9
21 22 11
23 24 12

% head -n10 sample.test
13854 157 35
74543 39 39
94421 39598 41
132628 668 18
90426 3546 5
71342 334 28
12604 9177 8
48132 81345 16
36296 82 4
61936 129198 17
```

Each row indicates one data sample (zero-based indices).
Each column indicates the indices of corresponding mode.
e.g., the first row means X_{0,1,0} = 1
Sample data is available at [https://zenodo.org/record/13966#.WCHGjdzT2Vs](https://zenodo.org/record/13966#.WCHGjdzT2Vs).
