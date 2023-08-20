#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 11:19:08 2023

@author: vep
"""
import math
import numpy as np
from scipy.optimize import lsq_linear  #method = 'trf' or 'bvls'
from package.draw.draw import show_steps

def calculateAndDraw(mtx,rhs) :
    """Function doing the job."""
    #nonlocal mtx, rhs
    mtx = mtx.reshape(mtx.shape, order = 'F')
    shape = mtx.shape # need it later for drawing
    mtx = np.swapaxes(mtx,0,2) # (x,y,d,s) -> (d,y,x,s)
    row = np.prod(mtx.shape[1:4])
    mtx = mtx.reshape( (mtx.shape[0], row), order = 'F')
    mtx0 = mtx
    rhs0 = rhs

    for r in np.linspace(-1.0,1.0,5) :
        for d in range(1,mtx.shape[0]) :
            mtx[d,:] = mtx0[d,:] - r*mtx0[d-1,:]
            rhs[d]   = rhs0[d] - r*rhs0[d-1]
        mtx[0,:] = math.sqrt(1.0 - r*r) * mtx0[0,:]
        rhs[0]   = math.sqrt(1.0 - r*r) * rhs0[0]
        result = lsq_linear(mtx,rhs,
                            bounds=(0.0, +np.inf),
                            method = 'bvls')
        x, y = result.x, result.unbounded_sol[0]
        x = x.reshape(shape[0],shape[1],shape[3])
        y = y.reshape(shape[0],shape[1],shape[3])
        x[0,:,:] = x[0,:,:] * (1.0 - r)
        y[0,:,:] = y[0,:,:] * (1.0 - r)
        s = f"nonnegative solution, r={r:4.1f}"
        t = f"unconstrained, r={r:4.1f}"
        show_steps(s,x,t,y, sameaxs = True)
