"""
This file implements helper functions used within the project that are not
specific to annealing or tunneling in general.
"""

import sys


def enum(**enums):
    return type('Enum', (), enums)


def read_params():
    """
    read_params -- This reads the params.in files which specifies the value of each Simulated Annealing parameter, as well
    as the value of the gamma parameter for stochastic tunneling.
    """
    f = open("anneal/params.in", "r")

    try:
        params = enum(INITPROB=float(f.readline().strip().split()[1]), FREEZE_LIM=int(f.readline().strip().split()[1]),
                      SIZEFACTOR=int(f.readline().strip().split()[1]), CUTOFF=int(f.readline().strip().split()[1]),
                      TEMPFACTOR=float(f.readline().strip().split()[1]), MINPERCENT=float(f.readline().strip().split()[1]),
                      GAMMA=float(f.readline().strip().split()[1]), KCOLORS=int(f.readline().strip().split()[1]))
    except ValueError:
        print "params.in has invalid format. Review README for correct format."
        exit(1)

    return params


def read_input():
    """
    read_input -- This reads the command line arguments passed, as:
    anneal <filename> <seed> <metaheuristic> [-st]
    where <metaheuristic> is one of: -penalty, -kempe, or -fixed_k
    """

    # input file

    with open(sys.argv[1]) as f:
        for line in f:
            if line.startswith("c"):
                continue
            elif line.startswith("\n"):
                continue
            elif line.startswith("p "):
                problem = line.strip().split(" ")
                break
            else:
                print "Invalid line.  Problem must be defined first."
                exit(1)
        nodes = []
        edges = []
        optionals = []
        for line in f:
            if line.startswith("c"):
                continue
            if line.startswith("v"):
                x = line.split(" ")
                y = x.__len__()
                optionals.append(("v", x, y))
            else:
                t, x, y = line.strip().split(" ")
            if t == "n":
                nodes.append((int(x), int(y)))
            elif t == "e":
                edges.append((int(x), int(y)))
            elif t == "d":
                optionals.append(("d", x, y))
            elif t == "x":
                optionals.append(("x", x, y))
            else:
                print "Invalid character.  Expecting \"c\", \"n\", \"d\", \"x\", \"v\", or \"e\"."
                exit(1)



    # a random number seed
    seed = sys.argv[2]

    penalty = kempe = fixed_k = False

    # the metaheuristic to use
    m = sys.argv[3]
    if m == "-penalty":
        penalty = True
    elif m == "-kempe":
        kempe = True
    elif m == "-fixed_k":
        fixed_k = True
    else:
        print "Invalid metaheuristic entered. Options are \"-penalty\", \"-kempe\", and \"-fixed_k\"."
        exit(1)
    # optional stochastic tunneling
    st = False
    try:
        if sys.argv[4] == "-st":
            st = True
            try:
                outfile = sys.argv[5]
            except IndexError:
                outfile = "anneal"
        else:
            outfile = sys.argv[4]
    except IndexError:
        outfile = "anneal"

    return problem, nodes, edges, optionals, seed, penalty, kempe, fixed_k, st, outfile
