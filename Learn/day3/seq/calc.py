#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 11:19:08 2023

@author: vep
"""
import numpy as np

def argmin(Z) :
    kl = np.argmin(Z)
    k = np.floor(kl/Z.shape[0]).astype('int')
    l = np.floor(kl-k*Z.shape[0]).astype('int')
    return k,l

def seq_loops(mtx,rhs) :
    """Function doing the job."""
    mtx = mtx.reshape(mtx.shape, order = 'F')
    mtx = np.swapaxes(mtx,0,2) # (x,y,d,s) -> (d,y,x,s)
    shape = [mtx.shape[0],mtx.shape[1],mtx.shape[2]]
    S_d = np.full(shape, 0.0)
    for d in range(0,mtx.shape[0]) :
        for s in range(0,mtx.shape[3]) :
            for j in range(0,mtx.shape[2]) :
                for i in range(0,mtx.shape[1]) :
                    S_d[d,i,j] += mtx[d,i,j,s]

    shape = [mtx.shape[1],mtx.shape[2]]
    S = np.full(shape, 0.0)
    R = np.full(shape, 0.0)

    for d in range(0,mtx.shape[0]) :
        for j in range(0,mtx.shape[2]) :
            for i in range(0,mtx.shape[1]) :
                S[i,j] += S_d[d,i,j] * S_d[d,i,j]
                R[i,j] += S_d[d,i,j]*rhs[d]

    X = np.full(shape, 0.0)
    for j in range(0,mtx.shape[2]) :
        for i in range(0,mtx.shape[1]) :
            if S[i,j] != 0 :
                X[i,j] = R[i,j] / S[i,j]
            else :
                X[i,j] = 0.0

    F = np.full(shape, 0.0)
    for d in range(0,mtx.shape[0]) :
        for j in range(0,mtx.shape[2]) :
            for i in range(0,mtx.shape[1]) :
                F[i,j] += (S_d[d,i,j]*X[i,j] - rhs[d])**2

    ij = np.argmin(F)
    i = np.floor(ij/10).astype('int')
    j = np.floor(ij-i*10).astype('int')

    print(i,j, X[i,j], F[i,j])
    return (X,F)
    
def numpy_lib(mtx,rhs) :
    """Function doing the job."""
    mtx = np.swapaxes(mtx,0,2) # (x,y,d,s) -> (d,y,x,s)

    row = mtx.shape[1]*mtx.shape[2]
    sha = (mtx.shape[1],mtx.shape[2])
    
    S_d = np.sum(mtx,3).reshape([mtx.shape[0],row])

    S = np.sum(S_d*S_d,0)
    R = np.sum((S_d.T*rhs),1)
    X = R.copy()

    mask = S > 0
    np.divide(R, S, out=X, where=mask)

    F = np.sum(((S_d*X).T - rhs)**2,1).reshape(sha)

    i,j = argmin(F)
    
    X = X.reshape(sha)
    print(i,j, X[i,j], F[i,j])
    return (X,F)
    

