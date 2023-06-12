#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 01:25:09 2023

@author: ckuemme1
"""
# %% Import packages
import numpy as np, matplotlib.pyplot as plt
from juliacall import Main as jl
from juliacall import Pkg as jlPkg

def batch(n):
    x = np.sort(np.random.uniform(-1, 1, (1, n)))
    y = np.sin(x * 10) + np.random.normal(0, 0.1, (1, n))
    return x, y


# %% Test juliacall
jlPkg.add("Flux")
jl.seval("using Flux")
model = jl.Chain(
    jl.Dense(1, 10, jl.relu),
    jl.Dense(10, 10, jl.relu),
    jl.Dense(10, 10, jl.relu),
    jl.Dense(10, 1),
)
loss = jl.seval("m -> (x, y) -> Flux.Losses.mse(m(x), y)")(model)
x, y = batch(400)
lss = loss(x, y)
print("loss =", loss(x, y))

# %% Define graph, adjacecy matrix A, incidence matrix and Laplacian L
import networkx as nx
import numpy as np
from scipy.sparse.linalg import norm as spnorm

n = 10000
m = 5

G = nx.barabasi_albert_graph(n, m, seed=100)
# Gdi = nx.DiGraph(G,oriented=True)
# Cdi = nx.incidence_matrix(Gdi,oriented=True)
C = nx.incidence_matrix(
    G, oriented=True
)  # incidence matrix with 1's everywhere instead of 1's and -1's.
A = nx.adjacency_matrix(G)
L = nx.laplacian_matrix(G)
Ltest = C @ C.T
spnorm(L - Ltest, "fro") / spnorm(L, "fro")  # check Laplacian
# %% Define demand vector
d = np.zeros((n,))
d[0] = 1
d[-1] = -1

# %% Use dense numpy solver
import time

Ldense = L.toarray()
st = time.time()
xdense = np.linalg.solve(Ldense, d)
residual = L @ xdense - d

et = time.time()
elapsed_time = et - st
print(
    "Solver: Dense Numpy \nResidual norm:",
    np.linalg.norm(residual),
    "Time:",
    elapsed_time,
)
# %% Use scipy.sparse spsolve
# import scipy

# st = time.time()
# xsparse = scipy.sparse.linalg.spsolve(L, d)
# residual_sp = L @ xsparse - d
# et = time.time()
# elapsed_time = et - st
# print(
#     "Solver: Spsolve \nResidual norm:", np.linalg.norm(residual), "Time:", elapsed_time
# )


# %% Define function to use approxchol_lap from Laplacians.jl
def solve_LaplaciansJulia(A, d, tol=1e-14, timing=False):
    from juliacall import Main as jl
    from juliacall import convert as jlconvert

    jl.seval("using Laplacians")
    jl.seval("using SparseArrays")
    if timing:
        import time

        st = time.time()
    Acoo = A.tocoo()
    i = Acoo.row
    j = Acoo.col
    v = Acoo.data
    i_jul = jlconvert(T=jl.Vector[jl.Int64], x=i + 1)
    j_jul = jlconvert(T=jl.Vector[jl.Int64], x=j + 1)
    v_jul = jlconvert(T=jl.Vector[jl.Float64], x=v)
    AA = jl.SparseArrays.sparse(i_jul, j_jul, v_jul, n, n)
    if timing:
        time_prep = time.time()
    solver = jl.Laplacians.approxchol_lap(AA, verbose=True, tol=tol)
    if timing:
        time_chol = time.time()
    x = solver(d)
    if timing:
        time_solved = time.time()
        residual_Laplacians = jl.Laplacians.lap(AA) * x - d
        et = time.time()
        elapsed_time = et - st
        print(
            "Solver: Laplacians.jl Approxchol \nResidual norm:",
            np.linalg.norm(residual_Laplacians),
            "\nTime for preparation:",
            time_prep - st,
            "\nTime for Approx. Cholesky:",
            time_chol - time_prep,
            "\nTime for PCG:",
            time_solved - time_chol,
            "\nTotal time:",
            elapsed_time,
        )
    return x


# %% Use Laplacian solver
xLap = solve_LaplaciansJulia(A, d, timing=True)

# %% Compute flow
f = C.T @ xdense
# fsparse = C.T @ xsparse

fLap = C.T @ xLap

# %%
