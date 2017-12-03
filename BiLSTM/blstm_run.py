##################################
#author          :Vishwajit
#date            :2017 09
#owner           :Cogknit Semantics Pvt Ltd
##################################

import numpy as np
import pyopencl as cl
import os 
from get_params import layer1, layer2
from lstm_parser import LSTMprop

os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
os.environ['PYOPENCL_CTX']='0'

input_mat = np.load('test_input_features.npy')
params = np.load('BiLSTM_parameters.npy').item()

##FCNN Params
fw = params['weights']
fb = params['biases']

###define layer1 and layer2 objects
no_layers = 2
	
L_0 = layer1()
L_0_0 = L_0.forward()
L_0_1 = L_0.backward()

L_1 = layer2()
L_1_0 = L_1.forward()
L_1_1 = L_1.backward()

out = LSTMprop(input_mat, L_0_0)

print out.shape