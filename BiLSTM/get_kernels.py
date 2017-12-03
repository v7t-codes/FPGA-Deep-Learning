
##################################
#author          :Vishwajit
#date            :2017 09
#owner           :Cogknit Semantics Pvt Ltd
##################################

import pyopencl as cl
import numpy as np
import os

os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
os.environ['PYOPENCL_CTX']='0'


def linear_kernel(context):
	#load kernelsource 
	kernelsource= open('../kernels/linear_mmul.cl').read()
	#define , build and return
	functor = cl.Program(context, kernelsource).build()
	mult = functor.linear_mmul
	mult.set_scalar_arg_dtypes([np.int32, np.int32, np.int32, None, None, None])
	return mult

def opt_kernel(context):
	#load kernelsource 
	kernelsource= open('../kernels/opt_mmul.cl').read()
	#define , build and return
	functor = cl.Program(context, kernelsource).build()
	mult = functor.linear_mmul
	mult.set_scalar_arg_dtypes([np.int32, np.int32, np.int32, None, None, None])
	return mult

def sigmoid_kernel(context):
	#load kernelsource 
	kernelsource= open('../kernels/sigmoid.cl').read()
	#define , build and return
	functor = cl.Program(context, kernelsource).build()
	sigm = functor.sigmoid
	sigm.set_scalar_arg_dtypes([None, None])
	return sigm

def tanh_kernel(context):
	#load kernelsource 
	kernelsource= open('../kernels/tanh.cl').read()
	#define , build and return
	functor = cl.Program(context, kernelsource).build()
	tanh_ = functor.tanh_
	tanh_.set_scalar_arg_dtypes([ None, None])
	return tanh_


def hamadard_kernel(context):
	kernelsource= open('../kernels/hadamard.cl').read()
	#define , build and return
	functor = cl.Program(context, kernelsource).build()
	hadamard = functor.hadamard
	hadamard.set_scalar_arg_dtypes([None, None, None])
	return hadamard
