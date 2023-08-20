#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 11:15:29 2023

@author: vep
"""
import numpy as np

def readFiles(mtxFile,rhsFile) :
    """Function reading files."""
    #nonlocal mtx, rhs
    mtx = np.load(mtxFile)
    assert mtx.ndim == 4, "Dimension of the input array inconsistent with the specification"
    rhs = np.loadtxt(rhsFile)
    #plt.plot(np.linspace(0, 10, 10), rhs, 'o', label='rhs')
    assert rhs.size == mtx.shape[0], "RHS size do not match"
    return (mtx,rhs)
