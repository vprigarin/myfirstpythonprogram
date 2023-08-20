#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 10:39:54 2023

@author: vep
"""
import arrays

def runExample(mtxFileName,rhsFileName) :
    a = arrays.Arrays(mtxFileName,rhsFileName)
    a.calculateAndDraw()

def testRun() :
    runExample("../../../Data/Anomaly/A.npy","../../../Data/Anomaly/n.csv")

if __name__ == "__main__" :
    import sys
    
    if len(sys.argv) == 1 :
        testRun()
    else :
        runExample(sys.argv[1],sys.argv[2])
