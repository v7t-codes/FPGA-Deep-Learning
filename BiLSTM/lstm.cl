// ##################################
// #author          :Vishwajit
// #date            :2017 09
// #owner           :Cogknit Semantics Pvt Ltd
// ##################################

// LSTM propagation kernel in C

//refer BiLSTM_inference 1 to understand breakdown and structure

// ARGUMENTS: 
// 0. Input matrix (X)

//INPUT
// 1. Wix
// 2. Wih
// 3. Wic
// 4. Bi

// FORGET
// 5. Wfx
// 6. Wfh
// 7. Wfc
// 8. Bf

//CELL STATE
// 9. Wcx
// 10. Wch
// 11. Bc 

//OUTPUT
// 12. Wox
// 13. Woh
// 14. Woc
// 15. Bo
// 16. Output Buffer

// float sigmoid(float a){

// 	return 1 / (1+ exp(-a));
// }

// float tanh_(float a){
	
// 	return tanh(a);
// }

__kernel void lstm_prop(
	int in_width,
	int timesteps, 
	int out_width,
	const __global float* input,
	const __global float* Wix,
	const __global float* Wih,
	const __global float* Wic,
	const __global float* Bi,
	const __global float* Wfx,
	const __global float* Wfh,
	const __global float* Wfc,
	const __global float* Bf,
	const __global float* Wcx,
	const __global float* Wch,
	const __global float* Bc,
	const __global float* Wox,
	const __global float* Woh,
	const __global float* Woc,
	const __global float* Bo,
	__global float* outbuff,
	__global float* H,
	__global float* I,
	__global float* F,
	__global float* C)
{
	int id = get_global_id(0);
	
	int z;
	
	for(int gid= 0; gid<timesteps ;gid++){

		if(gid==0){
			z= gid;
		}
		else {
			z= gid - 1;
		}

		// float I[timesteps][out_width];
		// float F[timesteps][out_width];
		// float C[timesteps][out_width];
		// float H[timesteps][out_width];

		float i1[320], i2[320], i3[320], f1[320], f2[320], f3[320];
		float c1[320], c2[320] ,c3[320], o1[320], o2[320], o3[320];

		// initialise hidden and cell state matrices
		
		// input multiplications ow = out_width, iw = 360
		for(int i=0 ; i< out_width; i++){
			for(int j=0; j< in_width; j++){
				i1[i] += Wix[i*in_width + j] * input[gid*timesteps + j];
				f1[i] += Wfx[i*in_width + j] * input[gid*timesteps + j];
				c1[i] += Wcx[i*in_width + j] * input[gid*timesteps + j];
				o1[i] += Wox[i*in_width + j] * input[gid*timesteps + j];
			}
		}

		// hidden multiplications
		for(int i=0 ; i< out_width ; i++){
			for(int j= 0; j<out_width; j++){	
				i2[i] += Wih[i*out_width + j] * H[z*timesteps + j]; 
				f2[i] += Wfh[i*out_width + j] * H[z*timesteps + j];
				c2[i] += Wch[i*out_width + j] * H[z*timesteps + j];
				o2[i] += Woh[i*out_width + j] * H[z*timesteps + j];
			}
		}
		
		//elementwise multiplications
		for(int i=0; i< out_width; i++){
			i3[i] = Wic[i] * C[z*timesteps + i]; 
			f3[i] = Wfc[i] * C[z*timesteps + i]; 
		} 

		// tanhs and sigs
		for(int i= 0; i< out_width; i++){
			I[gid*timesteps + i] = 1/(1+exp(-(i1[i] + i2[i] + i3[i] + Bi[i])));
			F[gid*timesteps + i] = 1/(1+exp(-(f1[i] + f2[i] + f3[i] + Bf[i])));
			c3[i] = tanh(c1[i] + c2[i] + Bc[i]);
		}

		for(int i=0; i<out_width ; i++){
			C[gid*timesteps + i] = I[gid*timesteps + i] * c3[i] + F[gid*timesteps + i]* C[z*timesteps + i];
		}

		for(int i=0; i< out_width; i++){
			o3[i] = Woc[i]* C[gid*timesteps + i];
		}

		for(int i= 0; i< out_width; i++){
			outbuff[gid*timesteps + i] = 1/(1+exp(-(o1[i] + o2[i] + o3[i] + Bo[i])));
		}

		for(int i=0; i< out_width ; i++){
			H[gid*timesteps + i] = outbuff[gid*timesteps + i] * tanh(C[gid*timesteps + i]);
		}
	}
}