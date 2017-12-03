##################################
#author          :Vishwajit
#date            :2017 09
#owner           :Cogknit Semantics Pvt Ltd
##################################

import numpy as np
import pyopencl as cl
import os 
from util import hadamard
# linear_mult , 
# sigmoid, tanh
params = np.load('BiLSTM_parameters.npy').item()

##FCNN Params
fw = params['weights']
fb = params['biases']

##to verify structure using pure python uncomment following lines##
# hadamard = np.multiply

def sigmoid(A):
	return 1/(1+ np.exp(-A))

linear_mult = np.dot 
tanh = np.tanh

## load params
from get_params import layer1, layer2

## load input
input_mat = np.load('test_input_features.npy')

## ocl device and compiler params
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
os.environ['PYOPENCL_CTX']='0'

###define layer1 and layer2 objects
no_layers = 2

L_0 = layer1()
L_0_0 = L_0.forward()
L_0_1 = L_0.backward()

L_1 = layer2()
L_1_0 = L_1.forward()
L_1_1 = L_1.backward()

def LSTMprop(x, param, timesteps):

	for t in range(timesteps):

		if t==0:
			z=t
		else:
			z= t-1
		print t
		i_1 = linear_mult(param.Wix, x[t])
		i_2 = linear_mult(param.Wih, param.hidden_state[z])
		i_3 = hadamard(param.Wic, param.cell_state[z])
		param.i[t] = sigmoid(i_1 + i_2 + i_3 + param.bi)

		f_1 = linear_mult(param.Wfx, x[t])
		f_2 = linear_mult(param.Wfh, param.hidden_state[z])
		f_3 = hadamard(param.Wfc, param.cell_state[z])
		param.f[t] = sigmoid(f_1 + f_2 + f_3 + param.bf)

		c_1 = linear_mult(param.Wcx, x[t])
		c_2 = linear_mult(param.Wch, param.hidden_state[z])
		c_3 = tanh(c_1 + c_2 + param.bc)
		param.cell_state[t] = hadamard(param.i[t], c_3) + hadamard(param.f[t], param.cell_state[z])

		o_1 = linear_mult(param.Wox, x[t])
		o_2 = linear_mult(param.Woh, param.hidden_state[z])
		o_3 = hadamard(param.Woc, param.cell_state[t])
		param.output[t] = sigmoid(o_1 + o_2 + o_3 + param.bo)

		param.hidden_state[t] = hadamard(param.output[t], tanh(param.cell_state[t]))

	out = param.output
	return out

count = 0
out1 = np.zeros(shape = L_0_0.output.shape)
out2 = np.zeros(shape = L_0_0.output.shape)

for k in range(no_layers):

	if k == 0:
		x = input_mat
	else :
		x = np.concatenate((out1, out2[::-1]), axis = 1)

	# forward and backward
	for j in range(2):

		count = 2*k + j
		if count == 0:
			param = L_0_0

		elif count == 1:
			param = L_0_1

		elif count == 2:
			param = L_1_0

		else:
			param = L_1_1

		if j == 1:
			out2 = LSTMprop(x[::-1], param, timesteps =207)
		else :
			out1 = LSTMprop(x, param, timesteps = 207)
		
		count +=1

print 'param_hidden shape', param.hidden_state[0].shape
LSTM_out = np.concatenate((out1, out2[::-1]), axis = 1)

print 'LSTM out:',LSTM_out.shape
print LSTM_out[0].shape

final_out = linear_mult(fw, LSTM_out[0].T) + fb

print 'final_out:',final_out.shape
print np.exp(final_out)/np.sum(np.exp(final_out[0]), axis = 0)
print np.argmax(final_out)