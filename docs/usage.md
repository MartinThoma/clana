# How to use clana with MNIST

## Prerequesites

Install `clana` and execute the example:

```
$ pip install clana
$ python mnist_example.py
```

This will generate the clana files.


## Usage

### distribution

```
$ clana distribution --gt gt-test.csv
11.35% 1 (1135 elements)
10.32% 2 (1032 elements)
10.28% 7 (1028 elements)
10.10% 3 (1010 elements)
10.09% 9 (1009 elements)
 9.82% 4 ( 982 elements)
 9.80% 0 ( 980 elements)
 9.74% 8 ( 974 elements)
 9.58% 6 ( 958 elements)
 8.92% 5 ( 892 elements)
```


### get-cm

This is an intermediate step required for the visualization.

```
$ clana get-cm --predictions train-pred.csv --gt gt-train.csv --n 10
2019-07-02 21:53:40,547 - root - INFO - cm was written to 'cm.json'
```
