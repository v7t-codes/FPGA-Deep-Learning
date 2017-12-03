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

context = cl.create_some_context()

kernelsource= open('lstm.cl').read()
#define , build and return
functor = cl.Program(context, kernelsource).build()
prop = functor.lstm_prop
prop.set_scalar_arg_dtypes([np.int32, np.int32, np.int32,None, None, None,None, None, None,None, None, None, None, None,
 	None,None, None, None,None, None, None,None,None, None])
context = cl.create_some_context()

def vectorize(A):
	return np.float32(A.ravel())

def LSTMprop(x, param, kernel = prop, context= context):
	queue = cl.CommandQueue(context)

	#Host variables
	h_input = vectorize(x)

	h_Wix = vectorize(param.Wix)
	h_Wih = vectorize(param.Wih)
	h_Wic = vectorize(param.Wic)
	h_Bi = vectorize(param.bi)

	h_Wfx = vectorize(param.Wfx)
	h_Wfh = vectorize(param.Wfh)
	h_Wfc = vectorize(param.Wfc)
	h_Bf = vectorize(param.bf)

	h_Wcx = vectorize(param.Wcx)
	h_Wch = vectorize(param.Wch)
	h_Bc = vectorize(param.bc)

	h_Wox = vectorize(param.Wox)
	h_Woh = vectorize(param.Woh)
	h_Woc = vectorize(param.Woc)
	h_Bo = vectorize(param.bo)

	h_H = vectorize(param.hidden_state)
	h_I = vectorize(param.i)
	h_F = vectorize(param.f)
	h_C = vectorize(param.cell_state)

	h_in_width = x.shape[1]
	h_timesteps = x.shape[0]
	h_out_width = 320

	h_out = np.ndarray(shape = (h_timesteps*h_out_width,))

	#device variables

	d_input = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_input)
	d_Wix = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_Wix)
	d_Wih = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_Wih)
	d_Wic = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_Wic)
	d_Bi = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_Bi)

	d_Wfx = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_Wfx)
	d_Wfh = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_Wfh)
	d_Wfc = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_Wfc)
	d_Bf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_Bf)

	d_Wcx = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_Wcx)
	d_Wch = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_Wch)
	d_Bc = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_Bc)

	d_Wox = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_Wox)
	d_Woh = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_Woh)
	d_Woc = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_Woc)
	d_Bo = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_Bo)

	d_H = cl.Buffer(context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_H)
	d_I= cl.Buffer(context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_I)
	d_F = cl.Buffer(context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_F)
	d_C = cl.Buffer(context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_C)

	d_out = cl.Buffer(context, cl.mem_flags.READ_WRITE, h_out.nbytes)

	glb_ws = (np.int(1),)
	lcl_ws = None

	kernel(queue, (1,), None, h_in_width, h_timesteps, h_out_width, d_input, d_Wix, d_Wih, d_Wic, d_Bi, d_Wfx, d_Wfh, d_Wfc, d_Bf, d_Wcx, d_Wch, d_Bc, d_Wox, d_Woh, d_Woc, d_Bo, d_out, d_H, d_I, d_F, d_C)

	queue.finish()

	cl.enqueue_copy(queue, h_out, d_out)

	assert len(h_out) == h_timesteps*h_out_width, "OOPS... Output size error"

	return h_out