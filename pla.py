#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 14:48:05 2021

@author: nicholaslillis
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
from matplotlib import cm
import matplotlib.lines as mlines
from mpl_toolkits.mplot3d import Axes3D


def plot(df,w1,w2,w0):
    #Plotter

    colors = pd.Series(['r' if label > 0 else 'b' for label in df["Label"]])
    ax = df.plot(x="X", y="Y", kind='scatter', c=colors)
    # Get scatter plot boundaries to define line boundaries
    xmin, xmax = ax.get_xlim()

    # Compute and draw line. ax + by + c = 0  =>  y = -a/b*x - c/b

    line_start = (xmin, xmax)
    line_end = ((-w1/w2)*xmin - w0/w2, (-w1/w2)*xmax - w0/w2)
    line = mlines.Line2D(line_start, line_end, color='red')
    ax.add_line(line)

    title = 'Scatter of feature %s vs %s' %(str("X"), str("Y"))
    ax.set_title(title)

    plt.show()
    

def func(b,w1,w2,const,x,y):
    return b*const+w1*x+w2*y         

if __name__ == '__main__':

    df = pd.read_csv("data1.csv",names = ["X","Y","Label"])
    arr = df.to_numpy()
    out_filename = 'results1.csv'    
    outfile = open(out_filename, "w")
    num = arr.shape[0]
    w1List = []
    w2List = []
    w0List = []
    w1 = 0
    w2 = 0
    w0 = 0
    
    converge = 1
    while converge !=0:
        converge = 0
        for i in range(num):
            x1 = arr[i,0]
            x2 = arr[i,1]
            y = arr[i,2]
            if (y*func(w0,w1,w2,1,x1,x2))<=0:
                w0 = w0 + y
                w1 = w1 + y*x1
                w2 = w2 + y*x2
                converge += 1
        w1List.append(w1)
        w2List.append(w2)
        w0List.append(w0)
        
    d = {'col1': w1List , 'col2': w2List, 'col3': w0List}
    df2 = pd.DataFrame(data=d)    
    df2.to_csv(outfile, header=False,index=False)
    
    #plot(df,w1,w2,w0)










            
            
            



#for items in data:     