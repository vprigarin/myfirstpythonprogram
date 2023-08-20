#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 10:39:54 2023

@author: vep
"""
import sys

def runExample(mtxFileName,rhsFileName) :
    """Function main."""
    sys.path.append('../day2')
    from package.read.readFiles import readFiles
    from package.draw.draw import show
    from seq.calc import seq_loops
    from seq.calc import numpy_lib
    mtx, rhs = readFiles(mtxFileName,rhsFileName)
    X,F = seq_loops(mtx,rhs)
    X,F = numpy_lib(mtx,rhs)
    show(" F ",F," X ",X)
   
def testRun() :
    """Function test."""
    runExample("../../Data/Anomaly/A.npy","../../Data/Anomaly/n.csv")


if __name__ == "__main__" :
    import sys    
    if len(sys.argv) == 1 :
        testRun()
    else :
        runExample(sys.argv[1],sys.argv[2])
