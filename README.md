Author: Tyson Loveless
Simulated Annealing implemented in Python 2.6.9

To run the program with some `graph.col` file in the form as described in the prompt, use:

`python anneal.py graph.col <seed> <method> [-st] [outputFileName]`

from the directory containing the file `anneal.py`.

Paramaters:
 - `<seed>` is a required random seed value
 - `<method>` (required) is one of `-penalty`, `-kempe`, or `-fixed_k`
 - `-st` is an optional to turn on stochastic tunneling
 - `outputFileName` is an optional name for the output.run and output.trace files

Required Libraries:
 - helper
 - random
 - copy
 - numpy
 - math
 - sys
