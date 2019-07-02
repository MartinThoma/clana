The following file formats are used within ``clana``.

Label Format
============

The label file format is a text format. It is used to make sense of the
prediction. The order matters.

Specification
-------------

-  One label per line
-  It is a CSV file with ``;`` as the delimiter and ``"`` as the quoting
   character.
-  The first value is a short version of the label. It has to be unique
   over all short versions.
-  The second value is a long version of the label. It has to be unique
   over all long versions.

Example
-------

Computer Vision
~~~~~~~~~~~~~~~

::

   car;car
   cat;cat
   dog;dog
   mouse;mouse

mnist.csv:

::

   0;0
   1;1
   2;2
   3;3
   4;4
   5;5
   6;6
   7;7
   8;8
   9;9

Language Identification
~~~~~~~~~~~~~~~~~~~~~~~

::

   German;de
   English;en
   French;fr

Classification Dump Format
==========================

TODO: THIS IS WAY TOO BIG!

The classification dump format is a text format. It describes what the
output of a classifier for some inputs.

.. _specification-1:

Specification
-------------

The Classification Dump Format is a text format.

-  Each line contains exactly one output of the classifier for one
   input.
-  It is a CSV file with ``;`` as the delimiter and ``"`` as the quoting
   character.
-  The first value is an identifier for the input. It is no longer than
   60 characters.
-  The second and following values are the outputs for each label. Each
   of those values is a number in ``[0, 1]``.
-  The outputs are in the same order as in the related ``label.csv``
   file.

.. _example-1:

Example
-------

::

   identifier 1;0.1;0.3;0.6
   ident 2;0.8;0.1;0.1

Ground Truth Format
===================

The Ground Truth Format is a text file format. It is used to describe
the ground truth of data.

.. _specification-2:

Specification
-------------

-  Each line contains the ground truth of exactly one element.
-  It is a CSV file with ``;`` as the delimiter and ``"`` as the quoting
   character.
-  The first value is an identifier for the input. It is no longer than
   60 characters.
-  The second and following values are the outputs for each label. Each
   of those values is a number in ``[0, 1]``.
-  The outputs are in the same order as in the related ``label.csv``
   file.

.. _example-2:

Example
-------

::

   identifier 1;1;0;1
   identifier 1;0.5;0;0.5
