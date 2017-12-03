##################################
#author          :Vishwajit
#date            :2017 09
#owner           :Cogknit Semantics Pvt Ltd
##################################

import numpy as np
import pyopencl as cl
import os 

"""
LSTM layer
"""
# class LSTM:

# 	def __init__(self,):
# n= total number of time steps
# xs = row corresponds data at different times t
# no of columns of xs is length of input

####### FUNCS 
# it = σ(Wix*xt+  Wih*ht−1 + Wic*ct−1 + bi)
# ft = σ(Wfx*xt + Wfh*ht−1 + Wfc*ct−1 + bf)
# ct = ft (dot) ct−1 + it (dot) φ(Wcx*xt + Wch*ht−1 + bc)
# ot = σ(Wox*xt + Woh*ht−1 + Woc*ct + bo) 
# ht = ot (dot) φ(ct)
#######

kernelsource= open('matmul.cl').read()

def forward(self, n, source, gi, gf, go, gix, gfx , gox, cix, state, output, WGI,WGF,WGO,WCI,WIP,WFP,WOP):
	
	#forward sequence activation
	for i in range(n):
		prev = np.zeros(ns) if t==0 else output[t-1]
		source[t,0]=1 
		source[t, 1:1+ni] = xs[t]
		source[t,1+ni:] = prev 
		x_shape = source[t].shape 
		# compute gate values
		gix[t] = mat_multiply_cl(input_A =WGI, input_B=source[t], input_A_shape=WGI.shape, input_B_shape= x_shape ,output_buffer= )
		gfx[t] = mat_multiply_cl(input_A =WGF, input_B=source[t], input_A_shape=WGF.shape, input_B_shape= x_shape ,output_buffer= )
		gox[t] = mat_multiply_cl(input_A =WGO, input_B=source[t], input_A_shape=WGO.shape, input_B_shape= x_shape ,output_buffer= )
		cix[t] = mat_multiply_cl(input_A =WCI, input_B=source[t], input_A_shape=WCI.shape, input_B_shape= x_shape ,output_buffer= )

		if t>0:
			gix[t] += WIP*state[t-1]
			gfx[t] += WFP*state[t-1]

		#define out_size as bytes
		gi[t] = sigmoid_cl(input_= , input_shape = ,output_buffer =)
		gf[t] = sigmoid_cl(input_= , input_shape= , output_buffer =)
		ci[t] = tanh_cl(input_=, input_shape= , output_buffer =)

		if t>0: 
			state[t] += gf[t]*state[t-1]
			gox[t] += WOP*state[t]

		go[t] = sigmoid_cl(input_= , input_shape= , output_buffer =)
		output = tanh_cl(input_= , input_shape= , output_buffer =)*go[t]

def vectorize(X):
	return X.ravel()

def get_work_sizes(dim1, dim2):
	max_local_size = 
	return

def sigmoid_cl(X):
	# define kernel params
	return 1 / (1 + np.exp(X))

def tanh_cl(X):
	# define kernel params 
	return np.tanh(X)

def mat_multiply_cl(input_A, input_B, input_A_shape, input_B_shape, output_buffer):

	if input_A.shape != input_A_shape:
		print "shapes don't match\n"
		exit 0

	A = vectorize(input_A)
	B = vectorize(input_B)
	C = output_buffer.flatten()

	#specify dimensions 
	M = A.shape[0]
	N = B.shape[1]
	K = A.shape[1]

	# create device memory
	d_A = cl.Buffer(context, cl.mem_flags.READ_ONLY|cl.mem_flags.COPY_HOST_PTR, hostbuf=A)
	d_B = cl.Buffer(context, cl.mem_flags.READ_ONLY|cl.mem_flags.COPY_HOST_PTR, hostbuf=B)
	d_C = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, C.nbytes)

	program = cl.Program(context, kernelsource).build()
	multiplier = program.mmul
	multiplier.set_scalar_arg_dtypes([np.int32, np.int32, np.int32, None, None , None])

	global_ws, local_ws = get_work_sizes(M, N)

	multiplier(queue, global_ws, local_ws, M, N, K, d_A, d_B, d_C)
	queue.finish()
	cl.enqueue_copy(queue, C, d_C)

	return C.reshape(M, N)


