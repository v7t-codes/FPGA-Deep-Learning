##################################
#author          :Vishwajit
#date            :2017 09
#owner           :Cogknit Semantics Pvt Ltd
##################################


import numpy as np

with open('peephole_i_c_fw_layer1.txt') as infile:
	polyShape = [map(float, line.split()) for line in infile]		

polyShape = np.array(polyShape)
np.save('___filename', polyShape)