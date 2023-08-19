#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 21:46:35 2023

@author: vep
"""
import math
import numpy as np
from scipy.optimize import lsq_linear  #method = 'trf' or 'bvls'
import matplotlib.pyplot as plt
from IPython.display import clear_output

class Arrays :
    """ class """
    def __init__(self,mtxFile,rhsFile) :
        self.mtx = np.load(mtxFile)
        assert self.mtx.ndim == 4,"Dimension of the input array inconsistent with the specification"
        self.rhs = np.loadtxt(rhsFile)
        assert self.rhs.size == self.mtx.shape[0], "RHS size do not match"
        #plt.plot(np.linspace(0, 10, 10), rhs, 'o', label='rhs')

    def calculateAndDraw(self) :
        self.mtx = self.mtx.reshape(self.mtx.shape, order = 'F')
        shape = self.mtx.shape # need it later for drawing
        self.mtx = np.swapaxes(self.mtx,0,2) # (x,y,d,s) -> (d,y,x,s)
        row = np.prod(self.mtx.shape[1:4])
        self.mtx = self.mtx.reshape( (self.mtx.shape[0], row), order = 'F')
        mtx0 = self.mtx
        rhs0 = self.rhs

        for r in np.linspace(-1.0,1.0,5) :
            for d in range(1,self.mtx.shape[0]) :
                self.mtx[d,:] = mtx0[d,:] - r*mtx0[d-1,:]
                self.rhs[d]   = rhs0[d] - r*rhs0[d-1]

            self.mtx[0,:] = math.sqrt(1.0 - r*r) * mtx0[0,:]
            self.rhs[0]   = math.sqrt(1.0 - r*r) * rhs0[0]
            result = lsq_linear(self.mtx,self.rhs,
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

def get_levels(d_l,d_r, saxs = True) :
    """Function returning levelset."""
    if saxs :
        pmin, pmax = (max(np.ndarray.min(d_l), np.ndarray.min(d_r)),
                      min(np.ndarray.max(d_l), np.ndarray.max(d_r)))
        levelsx = np.linspace(pmin, pmax, 15)
        levelsy = levelsx
    else :
        pmin, pmax = np.ndarray.min(d_l), np.ndarray.max(d_l)
        levelsx = np.linspace(pmin, pmax, 15)
        pmin, pmax = np.ndarray.min(d_r), np.ndarray.max(d_r)
        levelsy = np.linspace(pmin, pmax, 15)
    return levelsx,levelsy

def show_steps(title_l,data_l,title_r,data_r, sameaxs = True) :
    """Function drawing result by first index."""
    plt.style.use('_mpl-gallery-nogrid')
    xp, yp = (np.meshgrid(np.linspace(0, data_l.shape[1], data_l.shape[1]),
                          np.linspace(0, data_l.shape[1], data_l.shape[1])))
    lx,ly = get_levels(data_l,data_r,sameaxs)
    for k in range(0,data_l.shape[0]) :
        clear_output(wait=True)
        fig, axs = plt.subplots(1, 2, figsize=(8, 5))
        axs[0].set_title(title_l)
        axs[1].set_title(title_r)
        axs[0].invert_yaxis()
        axs[1].invert_yaxis()
        cf  = axs[0].contourf(xp,yp,np.flip(data_l[k,:,:],1).T, levels=lx)
        cf1 = axs[1].contourf(xp,yp,np.flip(data_r[k,:,:],1).T, levels=ly)
        fig.colorbar(cf)
        fig.colorbar(cf1)
        fig.tight_layout()
        plt.show()
        #input("Press Enter to continue...")

def runExample(mtxFileName,rhsFileName) :
    a = Arrays(mtxFileName,rhsFileName)
    a.calculateAndDraw()

def testRun() :
    runExample("../../../Data/Anomaly/A.npy","../../../Data/Anomaly/n.csv")

if __name__ == "__main__" :
    import sys
    runExample(sys.argv[1],sys.argv[2])
