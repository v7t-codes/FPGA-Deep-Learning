
##################################
#author          :Vishwajit
#date            :2017 09
#owner           :Cogknit Semantics Pvt Ltd
##################################

# import get_kernels as get
import numpy as np
import pyopencl as cl
import os

os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
os.environ['PYOPENCL_CTX']='0'

context = cl.create_some_context()

kernelsource1= open('../kernels/linear_mmul.cl').read()
#define , build and return
functor1 = cl.Program(context, kernelsource1).build()
mult = functor1.linear_mmul
mult.set_scalar_arg_dtypes([np.int32, np.int32, np.int32, None, None, None])

kernelsource2= open('../kernels/hadamard.cl').read()
#define , build and return
functor2 = cl.Program(context, kernelsource2).build()
hadamard = functor2.hadamard
hadamard.set_scalar_arg_dtypes([None, None, None])

kernelsource3= open('../kernels/tanh.cl').read()
#define , build and return
functor3 = cl.Program(context, kernelsource3).build()
tanh_ = functor3.tanh_
tanh_.set_scalar_arg_dtypes([ None, None])

kernelsource4 = open('../kernels/sigmoid.cl').read()
#define , build and return
functor4 = cl.Program(context, kernelsource4).build()
sigm = functor4.sigmoid
sigm.set_scalar_arg_dtypes([None, None])

def vectorize(X):
	return X.ravel()

###getting and running the built kernels

def linear_mult(A, B, kernel = mult):

	queue = cl.CommandQueue(context)

	a1, a2= A.shape

	b1= len(B)
	C= np.zeros(shape = (a1, 1)).astype(np.float32)

 	assert a2 == b1, "MAYDAY! the matrix shapes are not compatible"

 	#host memory
	h_A = (vectorize(A)).astype(np.float32)
	h_B = (vectorize(B)).astype(np.float32)
	h_C = (vectorize(C)).astype(np.float32)

	# print h_A.shape
	# print h_B.shape 
	# print h_C.shape

	#device memory
	d_A = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_A)
	d_B = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_B)
	d_C = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, h_C.nbytes)

	kernel(queue, (a1,1), (a1/5,1), a1, 1, b1, d_A, d_B, d_C)
	queue.finish()
	cl.enqueue_copy(queue, h_C, d_C)

	del queue
	# del kernel
	return h_C

def hadamard(A, B, kernel = hadamard):
	queue = cl.CommandQueue(context)
	# kernel= get.hamadard_kernel(context)

	C= np.zeros(shape = (len(B),)).astype(np.float32)
 	#host memory
	h_A = (vectorize(A)).astype(np.float32)
	h_B = (vectorize(B)).astype(np.float32)
	h_C = (vectorize(C)).astype(np.float32)

	a1= len(h_A)
	b1= len(h_B)
	print a1

	assert a1 == b1, "MAYDAY! hadamard demands you to pass vectors with same size"

	#device memory
	d_A = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_A)
	d_B = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_B)
	d_C = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, h_C.nbytes)

	kernel(queue, (a1,), (a1/5,), d_A, d_B, d_C)
	queue.finish()
	cl.enqueue_copy(queue, h_C, d_C)

	del queue
	# del kernel

	return h_C

def sigmoid(A, kernel = sigm):

	queue = cl.CommandQueue(context)
	# kernel= get.sigmoid_kernel(context)
	a1= len(A.ravel())
	C= np.zeros(shape = (a1,)).astype(np.float32)

 	#host memory
	h_A = (vectorize(A)).astype(np.float32)
	h_C = (vectorize(C)).astype(np.float32)
	
	#device memory
	d_A = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_A)
	d_C = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, h_C.nbytes)

	kernel(queue, (a1,), (a1/5,), d_A, d_C)
	queue.finish()
	cl.enqueue_copy(queue, h_C, d_C)

	# del queue
	# del kernel

	return h_C

def tanh(A, kernel = tanh_):

	queue = cl.CommandQueue(context)
	# kernel= get.tanh_kernel(context)

	a1= len(A.ravel())
	C= np.zeros(shape = (a1,)).astype(np.float32)
 	
 	#host memory
	h_A = (vectorize(A)).astype(np.float32)
	h_C = (vectorize(C)).astype(np.float32)

	#device memory
	d_A = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_A)
	d_C = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, h_C.nbytes)

	kernel(queue, (a1,), (a1/5,), d_A, d_C)
	queue.finish()
	cl.enqueue_copy(queue, h_C, d_C)
	
	del queue
	# del kernel

	return h_C