import helper as help
import random
import copy
import numpy as np
import math
import sys

"""A solution S to the graph coloring problem is a mapping of all vertices V in G to colors 1...k where k <= |V|.
    We wish to find a legal solution, such that for each edge connecting vertices (u, v), the color of u != color of v.
    Furthermore, we wish to find a minimum value for k; that is, we wish to minimize the number of colors used."""

global problem, edges, penalty, kempe, fixed_k, st, seed, K, cStar, gamma, outfile

global trace
trace = ""

"""
Each parameter (penalty, kempe, fixed_k), defines the neighborhood structure of a solution
as well as the cost function for determining how good a solution is

The neighborhood structure defines the expected neighborhood size.  Specifically,
   a the size of a the neighborhood is the number of possible neighboring solutions S' to current S -- depending on the
    definition of a solution, this changes the expected neighborhood size

with -penalty: "Solutions" are any coloring (even illegal colorings), so that any color class
    may have "bad edges" represented (in which two neighboring vertices are in the same class)
    Then the cost function accounts for a "penalty" (hence the name) to convince the algorithm
    to converge upon a feasible solution (only legal colorings).
    Solution: any coloring of the |V| vertices into nonempty disjoint sets C_1, C_2,... C_k (1 <= k <= |V|)
        whether there are bad edges or not.
    Neighbors: any two solutions are neighbors if you can move 1 vertex from any non-empty class (C_i) (1 <= i <= k)
        and move it to some other class C_j (1 <= j <= k+1), (i!=j) (given current number k of non-empty classes)
    Cost:  Let P be a solution, and E_i (1 <= i <= k) be the set of edges with bad edges in C_i.  Then,
         cost(P) = -sum_{i=1}^k |C_i|^2  +  sum_{i=1}^k 2*|C_i| * |E_i|


with -kemp:  A kempe chain is a connected component induced by taking the union of two independent sets C and D in G
    Solution: any legal coloring of the |V| vertices into nonempty disjoint sets C_1, C_2, ... C_k (1 <+ k <= |V|).
    Neighbors: Supposed that C and D are disjoint independent sets in G, we find a kempe chain H of C union D so that
        the symmetric difference C/\H (where C/\D indicates (C-D)U(D-C)), and D/\H are themselves disjoint independent
         sets whose union is C union D.  We choose some color class C_i (1 <= i <= k) with vertex v and some other color
        class C_j (j != i in same range), and let H be the kempe chain from C_i union C_j containing vertex v. Repeat
        until we obtain C_i, v, C_j, and H such that H != C union D (when H is not full). We then get our new solution
        by replacing C_i with C_i/\H and C_j with C_j/\H.
    Cost: Let P be a feasible solution, then
         cost(P) = -sum_{i=1}^k |C_i|^2

with -fixed_k:  Instead of minimizing k in a legal coloring, we instead minimize the number of monochromatic edges
    in a not-necessarily-legal coloring with a fixed number of color classes.
    Solution: Given G and a number of colors K, solutions are all partitions of V into K sets (empty sets allowed),
    Neighbors: given a partition C_i (1 <= i <= K) with a bad vertex v (a vertex is bad if the endpoint of a bad edge),
        and choose a random new partition C_j (j!=i) to move it to.  Essentially, we have (K-1) choices for each "bad" vertex
    Cost: total number of edges that do not have endpoints in different classes (bad edges)

"""


"""
DsaturAlgorithm:
Greedy approach that takes a vertex v with highest degree H, assigns to a new color class while removing v and v's
    edges from G.  if the next highest degre vertex u has degree == H, then u goes to the same color class as v,
    otherwise, u gets a new color class and the process is repeated until all vertices are colored.
"""
def dsatur(Graph, V):
    # graph represented as adjacency matrix
    matrix = np.array(Graph)
    not_colored = V     # # of vertices to color
    S = []
    color = 1
    # get vertex with max degree
    v = matrix.sum(axis=0).argmax()
    max = matrix[:, v].sum()
    C = []
    C.append(v+1)
    not_colored -= 1
    # remove v's edges from graph
    matrix[:, v] = np.zeros(matrix.shape[1])
    matrix[v] = np.zeros(matrix.shape[0])
    while not_colored:
        # get next max degree vertex
        v = matrix.sum(axis=0).argmax()
        nextMax = matrix[:, v].sum()
        # if degree is not same as previous vertex, new color class
        if nextMax != max:
            max = nextMax
            color += 1
            S.append(C)
            C = []
            continue
        # else, add to same color class
        C.append(v+1)
        not_colored -= 1
        # remove v's edges from graph
        matrix[:, v] = np.zeros(matrix.shape[1])
        matrix[v] = np.zeros(matrix.shape[0])

    # S is now a valid (possibly not chromatic) coloring of G
    return S


'''
Reads a problem instance to setup an upperbound on the cost function, and to set up the adjacency representation of
    the input graph.  Reads length of |V|, |E|, and sets up an expected neighborhood size N
'''
def READ_INSTANCE(problem):
    global edges, penalty, kempe, cStar
    """This reads a given problem to return """
    V = int(problem[2])
    E = int(problem[3])
    N = int(math.ceil(float(E) / V * 2))

    # adjacency matrix
    Graph = [[0] * V for i in range(V)]
    edge_u = [int(x[0]) for x in edges]
    edge_v = [int(x[1]) for x in edges]
    for i in range(len(edge_u)):
        u = edge_u[i]-1
        v = edge_v[i]-1
        Graph[u][v] += 1

    # cStar starts off with a large upper bound
    if penalty or kempe:
        cStar = V**2
    else: #fixed_k
        cStar = max(E, V)/2
    return cStar, N, V, E, Graph


'''
returns a list of bad edges.  a "bad edge" is an edge such that both vertices on the edge are in the same color class
    i.e. an invalid coloring results because of such edges.
'''
def get_bad_edges(S):
    global edges
    # initialize to zeros
    bad = [[] for x in range(0, len(S))]

    #check each color class for bad edges
    for color_class in S:
        edges_in_class = []
        for vertex in color_class:
            edges_in_class += [edge for edge in edges if vertex in edge]
        #check if both vertices of edges are in color class
        for edge in edges_in_class:
            if edge[0] in color_class and edge[1] in color_class:
                bad[S.index(color_class)].append(edge)

    return bad


'''
Returns the objective function's value, which differs for our 3 parameters (penalty, kempe, and fixed_k)
    this calculates the cost function given for each of these in Johnson et. al (1991)
'''
def get_cost(S):
    global edges, penalty, kempe, st, cStar, gamma
    bad = get_bad_edges(S)
    cost = 0
    if penalty or kempe:
        '''
        Let S be a solution, and E_i (1 <= i <= k) be the set of edges with bad edges in C_i.  Then,
            cost(S) = -sum_{i=1}^k |C_i|^2  + (if penalty) sum_{i=1}^k 2*|C_i| * |E_i|
        Essentially, we're trying to make the sizes of color classes as large as possible. If penalty method, we try
            minimizing the number of bad edges as well.
        '''
        for i in range(0, len(S)):
            cost -= len(S[i])**2
            if penalty:
                cost += 2*len(S[i])*len(bad[i])
    else: #fixed_k
        '''
            fixed_k's cost function is simply the number of bad edges in the solution
        '''
        for i in range(0, len(bad)):
            cost += len(bad[i])
        # for loop above counts number of vertices (2 per edge), so division by 2 to get number of edges
        cost /= 2

    if st:
        if penalty or kempe:
            cost = cost*(1.0 - math.exp((cost-cStar)/gamma))
        else:
            cost = int(cost*(1.0 - math.exp((cost-cStar)/gamma)))
    return cost


'''
returns a starting place for our given annealing approach.   penalty and kempe chains start at a valid solution
    based on DSATUR greedy coloring algorithm while fixed_k uses the length of DSATUR to inform a random starting point
     for the size K
'''
def INITIAL_SOLUTION(V, Graph):
    global edges, penalty, kempe, K
    S = []
    if penalty or kempe:
        # initialize solution from DSATUR
        S = dsatur(Graph, V)
    else: # fixed_k:
        # use DSATUR to get a starting place
        #K = len(dsatur(Graph, V))
        S = [[] for x in range(0, K)]
        for i in range(1, V):
            color_class = random.randint(0, K-1)
            try:
                S[color_class].append(i)
            except Exception as e:
                print e

    cost = get_cost(S)

    return S, cost


'''
chooses a neighbor to change S to based on given annealing approach.
    notes for how a neighbor is chosen are inline for each approach
'''
def NEXT_CHANGE(S):
    global edges, penalty, kempe
    if penalty:
        '''
        any two solutions are neighbors if you can move 1 vertex from any non-empty class (C_i) (1 <= i <= k)
        and move it to some other class C_j (1 <= j <= k+1), (i!=j) (given current number k of non-empty classes)
        '''
        Sprime = copy.deepcopy(S)
        C_i = random.randint(0, len(Sprime)-1)
        C_j = random.randint(0, len(Sprime))
        while C_i == C_j:
            C_j = random.randint(0, len(Sprime))
        v = random.choice(Sprime[C_i])
        Sprime[C_i].remove(v)
        if C_j > len(Sprime)-1:
            C = [v]
            Sprime.append(C)
        else:
            Sprime[C_j].append(v)
        if Sprime[C_i].__len__() == 0:
            del Sprime[C_i]

    elif kempe:
        '''
        We choose some color class C_i (1 <= i <= k) with vertex v and some other color
        class C_j (j != i in same range), and let H be the kempe chain from C_i union C_j containing vertex v. Repeat
        until we obtain C_i, v, C_j, and H such that H != C union D (when H is not full). We then get our new solution
        by replacing C_i with C_i/\H and C_j with C_j/\H
        '''
        found_neighbor = False
        run_count = 0
        while not found_neighbor and run_count < int(problem[2]):
            run_count += 1
            Sprime = copy.deepcopy(S)
            C_i = random.randint(0, len(Sprime)-1)
            C_j = random.randint(0, len(Sprime)-1)
            while C_i == C_j or len(Sprime[C_j]) < 2:
                C_j = random.randint(0, len(Sprime) - 1)
            try:
                v = random.choice(Sprime[C_i])
            except Exception as e:
                print e
            C_I = copy.deepcopy(Sprime[C_i])
            C_J = copy.deepcopy(Sprime[C_j])
            union = set(C_I).union(set(C_J))
            bad = get_bad_edges([list(union)])[0]
            H = set()
            for edge in bad:
                if len(edge) > 0:
                    H.add(edge[0])
                    H.add(edge[1])
      #      H = set(x[0] and x[1] for xs in get_bad_edges([list(union)]) for x in xs)
            if len(H) == 0 and H != set(C_I).union(set(C_J)):
                Sprime[C_i] = list(union)
                del Sprime[C_j]
            elif len(H) > 0 and H == set(C_I).union(set(C_J)):
                continue
            else:
                Sprime[C_i] = list(set(C_I).symmetric_difference(H))
                Sprime[C_j] = list(set(C_J).symmetric_difference(H))
                if C_i > C_j:
                    if len(Sprime[C_i]) == 0:
                        del Sprime[C_i]
                    if len(Sprime[C_j]) == 0:
                        del Sprime[C_j]
                else:
                    if len(Sprime[C_j]) == 0:
                        del Sprime[C_j]
                    if len(Sprime[C_i]) == 0:
                        del Sprime[C_i]

            found_neighbor = True
        if not found_neighbor:
            Sprime = copy.deepcopy(S)

    else:
        Sprime = copy.deepcopy(S)
        '''
        given a partition C_i (1 <= i <= K) with a bad vertex v (a vertex is bad if the endpoint of a bad edge),
        and choose a random new partition C_j (j!=i) to move it to.  Essentially, we have (K-1) choices for each "bad" vertex
        '''
        bad = get_bad_edges(Sprime)
        bad_indices = []
        for i in range(0, len(bad)):
            if len(bad[i]) > 0:
                bad_indices.append(i)
        i = random.choice(bad_indices)
        v = bad[i][0][0]
        C_i = i
        C_j = random.randint(0, len(Sprime)-1)
        while C_i == C_j:
            C_j = random.randint(0, len(Sprime)-1)
        Sprime[C_j].append(v)
        Sprime[C_i].remove(v)

    cost = get_cost(Sprime)
    return Sprime, cost


'''
checks for a feasible correct coloring solution (one with no bad edges)
'''
def is_feasible(S):
    global penalty, kempe
    bad = get_bad_edges(S)
    for colorClass in bad:
        if len(colorClass) > 0:
            return False
    return True


'''
if this is a better, valid solution, then update and record the trace
'''
def CHANGE_SOL(Sprime, cprime, Sstar, iteration):
    global trace, fixed_k, cStar
    S = copy.deepcopy(Sprime)
    changed = False
    if is_feasible(Sprime) and ((fixed_k and cprime <= cStar) or cprime < cStar):
        if kempe:
            if len(Sprime) < len(Sstar[0]):
                changed = True
        else:
            changed = True
        _Sstar = copy.deepcopy(Sprime)
        _cStar = cprime
        trace += "Iteration: " + str(iteration) + "\nCost: " + str(_cStar) + "\nColoring:\n"
        for i in range(0, len(S)):
            trace += "\tColor " + str(i+1) + ": " + str(S[i]) + "\n"
        trace += "\n"
        '''
        for the "fixed_k" method, I chose to make it more of a "dynamic_k," as a "good guess" for k is not always
            available.  using DSATUR for an initial k starting point, I shrink the number of color classes to attempt
            to force a better solution.  This has proven quite effective in my testing.
        '''
        if fixed_k:
            remove = random.randint(0, len(S)-1)
            C_i = S[remove]
            for elem in C_i:
                new = random.randint(0, len(S)-1)
                while new == remove:
                    new = random.randint(0, len(S) - 1)
                S[new].append(elem)
            del S[remove]
    else:
        if len(Sstar) > 0:
            _Sstar = copy.deepcopy(Sstar[0])
        else:
            _Sstar = copy.deepcopy(Sstar)
        _cStar = cStar
    return S, _Sstar, _cStar, changed


'''
returns the best solution found
'''
def FINAL_SOLN(S):
    Sstar = [list(i[0]) for i in S]
    min_length = min(map(len, Sstar))
    found = False
    for sol in S:
        if len(sol[0]) == min_length:
            if found:
                if sol[1] < best_cost:
                    best_soln = sol[0]
                    best_cost = sol[1]
            else:
                best_soln = sol[0]
                best_cost = sol[1]
                found = True
    return best_soln


'''
the simulated annealing approach for an optimization problem, pseudocoe followed as given by Johnson et. al (1991)
'''
def simulate_annealing(x):
    global problem, trace, edges, penalty, kempe, fixed_k, st, seed, cStar, gamma, K, outfile
    problem, nodes, edges, optionals, seed, penalty, kempe, fixed_k, st, outfile = x
    random.seed(seed)
    params = help.read_params()
    gamma = params.GAMMA
    if fixed_k:
        K = params.KCOLORS
    print ("Coloring the graph " + str(sys.argv[1]) + " using the " + str(sys.argv[3]) + " method") + ("." if not st else " with stochastic tunneling turned on.")
    cStar, N, V, E, Graph = READ_INSTANCE(problem) # cStar initially set to based on heuristic as upper bound
    S, c = INITIAL_SOLUTION(V, Graph)

    # choose initial temperature > 0
    if penalty:
        T = 40
    elif kempe:
        T = 5
    else:
        T = 15
    if penalty or kempe:
        Sstar = [(copy.deepcopy(S), get_cost(S))]
    else:
        Sstar = []
    freezecount = 0
    iteration = 0

    if penalty or kempe:
        trace += "Iteration: 0" + "\nCost: " + str(get_cost(S)) + "\nColoring:\n"
        for i in range(0, len(S)):
            trace += "\tColor " + str(i + 1) + ": " + str(S[i]) + "\n"
        trace += "\n"

    while freezecount < params.FREEZE_LIM: # while not yet frozen
        changed = False
        cChanged = False
        changes = trials = 0
        while trials < params.SIZEFACTOR*N and changes < params.CUTOFF*N:
            trials += 1
            iteration += 1
            Sprime, cprime = NEXT_CHANGE(S)
            delta = cprime - c
            if delta <= 0: #downhill
                c = cprime
                S, _Sstar, cStar, changed = CHANGE_SOL(Sprime, cprime, (Sstar[-1] if len(Sstar)>0 else Sstar), iteration)
            else: #delta > 0
                r = random.random() # 0 <= r <= 1
                probability = math.exp(-delta / T) #e^(-delta/T)
                if r <= probability:  # (randomly accept this worse move)
                    c = cprime
                    S, _Sstar, cStar, changed = CHANGE_SOL(Sprime, cprime, (Sstar[-1] if len(Sstar)>0 else Sstar), iteration)
            if changed:
                cChanged = True
                changes += 1
                Sstar.append((copy.deepcopy(_Sstar), cStar))

        T = params.TEMPFACTOR * T
        if cChanged: #c* was changed
            freezecount = 0
        if changes/trials < params.MINPERCENT:
            freezecount += 1

    if Sstar:
        return FINAL_SOLN(Sstar)

    else:
        run = open(outfile + ".run", "w")
        run.write("no solution found")

        # for anneal.trace
        traceFile = open(outfile + ".trace", "w")
        traceFile.write("no solution found")
        exit(0)


'''
because it's weird not having a main function.
'''
def main():
    global trace
    # read input file
    x = help.read_input()

    S = simulate_annealing(x)

    #the rest is for output files
    cost = get_cost(S)
    num = len(S)

    # for anneal.run
    output = "Objective function value: " + str(cost) + "\n"
    output += "Colors used: " + str(num) + "\n"
    output += "Color Assignment:\n"
    for i in range(0, len(S)):
        output += "\tColor " + str(i+1) + ": " + str(S[i]) + "\n"

    run = open(outfile + ".run", "w")
    run.write(output)

    # for anneal.trace
    traceFile = open(outfile + ".trace", "w")
    traceFile.write(trace)


'''run program'''
main()
