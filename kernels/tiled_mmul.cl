// ##################################
// #author          :Vishwajit
// #date            :2017 09
// #owner           :Cogknit Semantics Pvt Ltd
// ##################################


__kernel void mmul(const int M, const int N, const int K,
                      const __global float* A,
                      const __global float* B,
                      __global float* C) {
    #define TSM 128                
    #define TSN 128                
    #define TSK 16                 
    #define WPTM 8                 
    #define WPTN 8                 
    #define RTSM (TSM/WPTM)        
    #define RTSN (TSN/WPTN)        
    #define LPTA ((TSK*TSM)/(RTSM*RTSN)) 
    #define LPTB ((TSK*TSN)/(RTSM*RTSN)) 

    
    const int tidm = get_local_id(0); 
    const int tidn = get_local_id(1); 
    const int offsetM = TSM*get_group_id(0); 
    const int offsetN = TSN*get_group_id(1); 
 
    // Local memory to fit a tile of A and B
    __local float Asub[TSK][TSM];
    __local float Bsub[TSN][TSK+2];
 
    // Allocate register space
    float Areg;
    float Breg[WPTN];
    float acc[WPTM][WPTN];
 
    // Initialise the accumulation registers
    for (int wm=0; wm<WPTM; wm++) {
        for (int wn=0; wn<WPTN; wn++) {
            acc[wm][wn] = 0.0f;
        }
    }
    
    // Loop over all tiles
    int numTiles = K/TSK;
    for (int t=0; t<numTiles; t++) {
 
        // Load one tile of A and B into local memory
        for (int la=0; la<LPTA; la++) {
            int tid = tidn*RTSM + tidm;
            int id = la*RTSN*RTSM + tid;
            int row = id % TSM;
            int col = id / TSM;
            int tiledIndex = TSK*t + col;
            Asub[col][row] = A[tiledIndex*M + offsetM + row];
            Bsub[row][col] = B[tiledIndex*N + offsetN + row];
        }
        
        // Synchronise to make sure the tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);
 
        // Loop over the values of a single tile
        for (int k=0; k<TSK; k++) {
 
            // Cache the values of Bsub in registers
            for (int wn=0; wn<WPTN; wn++) {
                int col = tidn + wn*RTSN;
                Breg[wn] = Bsub[col][k];
            }
 
            // Perform the computation
            for (int wm=0; wm<WPTM; wm++) {
                int row = tidm + wm*RTSM;
                Areg = Asub[k][row];
                for (int wn=0; wn<WPTN; wn++) {
                    acc[wm][wn] += Areg * Breg[wn];
                }
            }
        }
        // Synchronise before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }
 
    // Store the final results in C
    for (int wm=0; wm<WPTM; wm++) {
        int globalRow = offsetM + tidm + wm*RTSM;
        for (int wn=0; wn<WPTN; wn++) {
            int globalCol = offsetN + tidn + wn*RTSN;
            C[globalCol*M + globalRow] = acc[wm][wn];
        }
    }
}