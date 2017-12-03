##################################
#author          :Vishwajit
#date            :2017 09
#owner           :Cogknit Semantics Pvt Ltd
##################################


import numpy as np
import pyopencl as cl 
import os 

os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
os.environ['PYOPENCL_CTX']='0'

M= 320
K= 360
N= 1

TSM = 128                		# The tile-size in dimension M
TSN = 128                		# The tile-size in dimension N
TSK = 16                 		# The tile-size in dimension K
WPTM = 8                 		# The work-per-thread in dimension M
WPTN = 8                 		# The work-per-thread in dimension N
RTSM = (TSM/WPTM)        		# The reduced tile-size in dimension M
RTSN = (TSN/WPTN)               # The reduced tile-size in dimension N
LPTA = ((TSK*TSM)/(RTSM*RTSN))  # Loads-per-thread for A
LPTB = ((TSK*TSN)/(RTSM*RTSN))  # Loads-per-thread for Buffer

# def closestMultiple(a, b):
# 	remainder = a % b
# 	if remainder == 0:
# 		return a
# 	else:
# 		return a - remainder + b

#create and host memory vars 

# A = M x K
# B = K x N 
# C = M x N

context = cl.create_some_context()
queue = cl.CommandQueue(context)

A = np.ones((M, K)).astype(np.float32)
B = np.ones((K, N)).astype(np.float32)
C = np.empty((M, N)).astype(np.float32)

h_A = A.ravel()
h_B = B.ravel()
h_C = C.ravel()

#create and device memory vars

d_A = cl.Buffer(context, cl.mem_flags.READ_ONLY|cl.mem_flags.COPY_HOST_PTR, hostbuf=h_A)
d_B = cl.Buffer(context, cl.mem_flags.READ_ONLY|cl.mem_flags.COPY_HOST_PTR, hostbuf=h_B)
d_C = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, h_C.nbytes)

#read kernel 
kernelsource = open("linear_mmul.cl").read()
program = cl.Program(context, kernelsource).build()

multiplier = program.linear_mmul
multiplier.set_scalar_arg_dtypes([np.int32, np.int32, np.int32, None, None , None])

#work_sizes
local_ws = (32,1)#(RTSM, RTSN)
global_ws = (320,1)#(M/WPTM, N/WPTN)

multiplier(queue, global_ws, local_ws, M, N, K, d_A, d_B, d_C)
queue.finish()

cl.enqueue_copy(queue, h_C, d_C)

print h_C

print h_C.shape 

print 'Verified and working:::>' , np.max(h_C) == K