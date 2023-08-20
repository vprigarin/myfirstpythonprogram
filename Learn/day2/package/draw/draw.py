#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 11:17:28 2023

@author: vep
"""
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output

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

def show(title_l,data_l,title_r,data_r) :
    """Function drawing result by first index."""
    plt.style.use('_mpl-gallery-nogrid')
    xp, yp = (np.meshgrid(np.linspace(0, data_l.shape[0], data_l.shape[0]),
                          np.linspace(0, data_r.shape[1], data_r.shape[1])))
    lx,ly = get_levels(data_l,data_r,False)
    clear_output(wait=True)
    fig, axs = plt.subplots(1, 2, figsize=(8, 5))
    axs[0].set_title(title_l)
    axs[1].set_title(title_r)
    #axs[0].invert_yaxis()
    #axs[1].invert_yaxis()
    #cf  = axs[0].contourf(xp,yp,np.flip(data_l,1).T, levels=lx)
    #cf1 = axs[1].contourf(xp,yp,np.flip(data_r,1).T, levels=ly)
    cf  = axs[0].contourf(xp,yp,data_l, levels=lx)
    cf1 = axs[1].contourf(xp,yp,data_r, levels=ly)
    fig.colorbar(cf)
    fig.colorbar(cf1)
    fig.tight_layout()
    plt.show()
    #input("Press Enter to continue...")
