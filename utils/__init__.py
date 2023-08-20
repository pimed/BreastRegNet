"""
    Author: Negar Golestani
    Created: August 2023
"""


import os
import numpy as np
import math

from options.config import RESULTS_ROOTDIR








###################################################################################################
class clockwise_angle_and_distance():
    def __init__(self, origin):
        self.origin = origin
    # ------------------------------------------------------------------------------------
    def __call__(self, point, refvec = [0, 1]):
        if self.origin is None:
            raise NameError("clockwise sorting needs an origin. Please set origin.")
        
        vector = [point[0]-self.origin[0], point[1]-self.origin[1]]
        lenvector = np.linalg.norm(vector[0] - vector[1])
        if lenvector == 0: return - math.pi, 0

        normalized = [vector[0]/lenvector, vector[1]/lenvector]
        dotprod  = normalized[0]*refvec[0] + normalized[1]*refvec[1] # x1*x2 + y1*y2
        diffprod = refvec[1]*normalized[0] - refvec[0]*normalized[1] # x1*y2 - y1*x2
        angle = math.atan2(diffprod, dotprod)

        if angle < 0:
            return 2*math.pi+angle, lenvector

        return angle, lenvector
###################################################################################################
def merge_contours(cnts):
    if len(cnts) > 1: 
        list_of_pts = np.vstack(cnts).reshape(-1,2)
        clock_ang_dist = clockwise_angle_and_distance( np.mean(list_of_pts, axis=0) ) # set origin
        cnts = sorted(list_of_pts, key=clock_ang_dist) # use to sort

    return np.array(cnts, dtype=np.int32).reshape((-1,1,2))
###################################################################################################



####################################################################################################################################
def get_sortedResultFolders():
    name_list = os.listdir(RESULTS_ROOTDIR)

    sorting_list = list()
    for name in name_list:
        try:  num = int(name.split('_')[0].strip().split('+')[0].split('-')[-1])
        except: num = 0
        sorting_list.append(num)

    name_list_sorted = [fn for _, fn in sorted(zip(sorting_list, name_list))]
    return name_list_sorted
####################################################################################################################################
def side_by_side(*args, space=10):

    txtLines_list, maxL_list = list(), list()
    Nlines = 0 

    for txt in args:
        txtLines = txt.split('\n') 
        maxL = np.max([len(line) for line in txtLines])

        txtLines_list.append(txtLines)
        maxL_list.append(maxL)
        Nlines = max(Nlines, len(txtLines))

    txt = ''
    for idx in range(Nlines):     
        for i, (txtLines, maxL) in enumerate(zip(txtLines_list, maxL_list)):            
            line = txtLines[idx] if len(txtLines)>idx else '' 
            if i > 0: txt += ' '*space 
            txt += line + ' '*(maxL-len(line))
        txt += '\n'

    return txt  
####################################################################################################################################

