#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 10:39:54 2023

@author: vep
"""
import package.read.readFiles
import package.calc.calc

def runExample(mtxFileName,rhsFileName) :
    """Function main."""
    mtx, rhs = package.read.readFiles.readFiles(mtxFileName,rhsFileName)
    package.calc.calc.calculateAndDraw(mtx,rhs)
   
def testRun() :
    """Function test."""
    runExample("../../Data/Anomaly/A.npy","../../Data/Anomaly/n.csv")


if __name__ == "__main__" :
    import sys
    
    if len(sys.argv) == 1 :
        testRun()
    else :
        runExample(sys.argv[1],sys.argv[2])
