How to use clana with MNIST
===========================

Prerequesites
-------------

Install ``clana`` and execute the example:

::

   $ pip install clana
   $ python mnist_example.py

This will generate the clana files.

Usage
-----

distribution
~~~~~~~~~~~~

::

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

get-cm
~~~~~~

This is an intermediate step required for the visualization.

::

   $ clana get-cm --predictions train-pred.csv --gt gt-train.csv --n 10
   2019-07-02 21:53:40,547 - root - INFO - cm was written to 'cm.json'

visualize
~~~~~~~~~

::

   $ clana visualize --cm cm.json
   Score: 12634
   2019-07-02 22:13:54,987 - root - INFO - n=10
   2019-07-02 22:13:54,987 - root - INFO - ## Starting Score: 12634.00
   2019-07-02 22:13:54,988 - root - INFO - Current: 12249.00 (best: 12249.00, hot_prob_thresh=100.0000%, step=0, swap=False)
   2019-07-02 22:13:54,988 - root - INFO - Current: 10457.00 (best: 10457.00, hot_prob_thresh=100.0000%, step=1, swap=False)
   2019-07-02 22:13:54,988 - root - INFO - Current: 10453.00 (best: 10453.00, hot_prob_thresh=100.0000%, step=3, swap=False)
   2019-07-02 22:13:54,988 - root - INFO - Current: 10340.00 (best: 10340.00, hot_prob_thresh=100.0000%, step=6, swap=True)
   2019-07-02 22:13:54,989 - root - INFO - Current: 10166.00 (best: 10166.00, hot_prob_thresh=100.0000%, step=14, swap=True)
   2019-07-02 22:13:54,989 - root - INFO - Current: 9644.00 (best: 9644.00, hot_prob_thresh=100.0000%, step=17, swap=True)
   2019-07-02 22:13:54,989 - root - INFO - Current: 9617.00 (best: 9617.00, hot_prob_thresh=100.0000%, step=19, swap=True)
   2019-07-02 22:13:54,990 - root - INFO - Current: 9528.00 (best: 9528.00, hot_prob_thresh=100.0000%, step=38, swap=False)
   2019-07-02 22:13:54,992 - root - INFO - Current: 9297.00 (best: 9297.00, hot_prob_thresh=100.0000%, step=86, swap=True)
   2019-07-02 22:13:54,993 - root - INFO - Current: 9092.00 (best: 9092.00, hot_prob_thresh=100.0000%, step=109, swap=True)
   2019-07-02 22:13:54,994 - root - INFO - Current: 9018.00 (best: 9018.00, hot_prob_thresh=100.0000%, step=123, swap=True)
   Score: 9018
   Perm: [0, 6, 5, 3, 8, 1, 2, 7, 9, 4]
   2019-07-02 22:13:55,029 - root - INFO - Classes: [0, 6, 5, 3, 8, 1, 2, 7, 9, 4]
   Accuracy: 94.34%
   2019-07-02 22:13:55,152 - root - INFO - Save figure at '/home/moose/confusion_matrix.tmp.pdf'
   2019-07-02 22:13:55,269 - root - INFO - Found threshold for local connection: 258
   2019-07-02 22:13:55,269 - root - INFO - Found 9 clusters
   2019-07-02 22:13:55,270 - root - INFO - silhouette_score=-0.0067092812311967
       1: [0]
       1: [6]
       1: [5]
       1: [3]
       1: [8]
       1: [1]
       1: [2]
       2: [7, 9]
       1: [4]
