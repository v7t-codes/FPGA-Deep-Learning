// ##################################
// #author          :Vishwajit
// #date            :2017 09
// #owner           :Cogknit Semantics Pvt Ltd
// ##################################

__kernel void tanh_(__global const float *A, 
			  __global float *C){
	// use in-built exp function
	int gid = get_global_id(0);
	C[gid] = tanh(A[gid]);
}