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

class TwoPanels :
    def __init__(self,data_l,data_r, saxs = True) :
        self.data_l = data_l
        self.data_r = data_r
        self.sameaxs = saxs
        if saxs :
            pmin, pmax = (max(np.ndarray.min(data_l), np.ndarray.min(data_r)),
                          min(np.ndarray.max(data_l), np.ndarray.max(data_l)))
            self.levelsx = np.linspace(pmin, pmax, 15)
            self.levelsy = self.levelsx
        else :
            pmin, pmax = np.ndarray.min(data_l), np.ndarray.max(data_l)
            self.levelsx = np.linspace(pmin, pmax, 15)
            pmin, pmax = np.ndarray.min(data_l), np.ndarray.max(data_l)
            self.levelsy = np.linspace(pmin, pmax, 15)
        

    def show_steps(self, title_l, title_r) :
        """Function drawing result by first index."""
        plt.style.use('_mpl-gallery-nogrid')
        xp, yp = (np.meshgrid(np.linspace(0, self.data_l.shape[1], self.data_l.shape[1]),
                          np.linspace(0, self.data_l.shape[1], self.data_l.shape[1])))
        for k in range(0,self.data_l.shape[0]) :
            clear_output(wait=True)
            fig, axs = plt.subplots(1, 2, figsize=(8, 5))
            axs[0].set_title(title_l)
            axs[1].set_title(title_r)
            axs[0].invert_yaxis()
            axs[1].invert_yaxis()
            cf  = axs[0].contourf(xp,yp,np.flip(self.data_l[k,:,:],1).T, levels=self.levelsx)
            cf1 = axs[1].contourf(xp,yp,np.flip(self.data_r[k,:,:],1).T, levels=self.levelsy)
            fig.colorbar(cf)
            fig.colorbar(cf1)
            fig.tight_layout()
            plt.show()
            #input("Press Enter to continue...")


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
            two_p = TwoPanels(x,y,True)
            two_p.show_steps(s,t)


#def runExample(mtxFileName,rhsFileName) :
#    a = Arrays(mtxFileName,rhsFileName)
#    a.calculateAndDraw()

#def testRun() :
#    runExample("../../../Data/Anomaly/A.npy","../../../Data/Anomaly/n.csv")

#if __name__ == "__main__" :
#    import sys
#    testRun()
    #runExample(sys.argv[1],sys.argv[2])
