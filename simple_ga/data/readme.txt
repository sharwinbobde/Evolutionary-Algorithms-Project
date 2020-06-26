All edges have edge weights randomly generated following a Beta
distribution with parameters  alpha = 100,  beta = 1 and scaled to the
range of [1;5].

File names: n<XXX>i<YY>.txt
<XXX>: Number of variables (nodes)
<YY>: Instance number (per dimensionality)
Best known value: n<XXX>i<YY>.bkv

File format:
<l,e>: The first line defines the number of nodes/variables l and the number of edges e.
<e1,e2,w>: All other lines define an edge from e1 (1-based) to e2, with weight w.

set0a: fully connected graphs.
set0b: 2D square-grid graphs.
set0c: 3D square-torus graphs.
set0d: Geometric, randomly distributed in the 1000xunit box.
       Distances are floor(Euclidean).
set0e: Same as set0d, but with floor(sqrt(l)) nearest neighbors.
