
// ##################################
// #author          :Vishwajit
// #date            :2017 09
// #owner           :Cogknit Semantics Pvt Ltd
// ##################################
__kernel void linear_mmul(const int M, const int N, const int K,
    
                      __global const float* A,
                      __global const float* B,
                      __global float* C) {
    
    // Thread identifiers
    const int globalRow = get_global_id(0); // Row ID of C (0..M)
    const int globalCol = get_global_id(1); // Col ID of C (0..N)
    
    // for(int x =0 ; x< 320;x++){
    //   printf("%f\n",A[x] );
    // }

    //printf("(%d, %d)\n",globalRow, globalCol);
    // Compute a single element (loop over K)
    float acc = 0.0;
    for (int k=0; k<K; k++) {
        acc += A[k*M + globalRow] * B[globalCol*K + k];
        //printf("%f * %f\n",A[k*M + globalRow],B[globalCol*K + k] );
    }
    // Store the result
    C[globalCol*M + globalRow] = acc;
    //printf("%f\n",acc );
}