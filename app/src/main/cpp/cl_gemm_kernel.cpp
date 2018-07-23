#ifdef USE_HALF
const char* cl_gemm_str= R"CLSRC(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void float2half(const __global float* in,
						__global half* out,
						const int M, const int N)
{
	const int globalRow = get_global_id(0); // Row ID of C (0..M)
	const int globalCol = get_global_id(1); // Col ID of C (0..N)
	if (globalCol < N && globalRow < M)
	{
		int idx = globalRow * N + globalCol;
		vstore_half(in[idx], 0, out + idx);
	}
}

__kernel void half2float(const __global half* in,
						__global float* out,
						const int M, const int N)
{
	const int globalRow = get_global_id(0); // Row ID of C (0..M)
	const int globalCol = get_global_id(1); // Col ID of C (0..N)
	if (globalCol < N && globalRow < M)
	{
		int idx = globalRow * N + globalCol;
		out[idx] = vload_half(0, in + idx);
	}
}
__kernel void myGEMM1TransHalf(const int M, const int N, const int K,
                      const __global float8* A,
                      const __global half8* B,
                      __global float* C)
{
    const int globalRow = get_global_id(0); // Row ID of C (0..M)
    const int globalCol = get_global_id(1); // Col ID of C (0..N)
    if (globalCol < N && globalRow < M)
    {
        half8 acc = { 0, 0, 0, 0, 0, 0, 0, 0};
        half8 t8;
        for (int k=0; k<K; k++)
        {
            vstore_half8(A[globalRow * K + k], 0, (half*)&t8);
            acc += t8 * B[globalCol*K + k];
        }
        
        half tf = acc.s0+acc.s1+acc.s2+acc.s3 +acc.s4+acc.s5+acc.s6+acc.s7;
        C[globalRow*N + globalCol] = vload_half(0, &tf);
    }
}
/*
__kernel void myGEMM1TransHalf(const int M, const int N, const int K,
                      const __global float8* A,
                      const __global half8* B,
                      __global float* C)
{
    const int globalRow = get_global_id(0); // Row ID of C (0..M)
    const int globalCol = get_global_id(1) * 2; // Col ID of C (0..N)
	if (globalCol + 1 < N && globalRow < M)
	{
		half8 acc0 = { 0, 0, 0, 0, 0, 0, 0, 0};
		half8 acc1 = { 0, 0, 0, 0, 0, 0, 0, 0};
		//half8 acc2 = { 0, 0, 0, 0, 0, 0, 0, 0};
		//half8 acc3 = { 0, 0, 0, 0, 0, 0, 0, 0};
		half8 t8;
		//half8 tb8;
		for (int k=0; k<K; k++)
		{
			vstore_half8(A[globalRow * K + k], 0, (half*)&t8);
			acc0 += t8 * B[globalCol*K + k];
			acc1 += t8 * B[(globalCol + 1)*K + k];
			//acc2 += t8 * B[(globalCol + 2)*K + k];
			//acc3 += t8 * B[(globalCol + 3)*K + k];
		}
		
		//half tf0 = acc0.s0+acc0.s1+acc0.s2+acc0.s3 +acc0.s4+acc0.s5+acc0.s6+acc0.s7;
		//half tf1 = acc1.s0+acc1.s1+acc1.s2+acc1.s3 +acc1.s4+acc1.s5+acc1.s6+acc1.s7;
		//C[globalRow*N + globalCol] = vload_half(0, &tf0);
		//C[globalRow*N + globalCol + 1] = vload_half(0, &tf1);
		
		//half tf2 = acc2.s0+acc2.s1+acc2.s2+acc2.s3 +acc2.s4+acc2.s5+acc2.s6+acc2.s7;
		//half tf3 = acc3.s0+acc3.s1+acc3.s2+acc3.s3 +acc3.s4+acc3.s5+acc3.s6+acc3.s7;
		//float4 temp4;
		//temp4.s0=vload_half(0, &tf0);
		//temp4.s1=vload_half(0, &tf1);
		//temp4.s2=vload_half(0, &tf2);
		//temp4.s3=vload_half(0, &tf3);
		//vstore4( temp4, 0, C+globalRow*N+globalCol);
		
		half2 tf;
		//tf.s0 = dot(acc0, (half8)(1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0));
		//tf.s1 = dot(acc1, (half8)(1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0));
		tf.s0 = acc0.s0+acc0.s1+acc0.s2+acc0.s3 +acc0.s4+acc0.s5+acc0.s6+acc0.s7;
		tf.s1 = acc1.s0+acc1.s1+acc1.s2+acc1.s3 +acc1.s4+acc1.s5+acc1.s6+acc1.s7;
		float2 temp2 = convert_float2(tf);
		vstore2( temp2, 0, C+globalRow*N+globalCol);	
	}
}
*/
/*
__kernel void myGEMM1TransHalf(const int M, const int N, const int K,
                      const __global float16* A,
                      const __global half16* B,
                      __global float* C)
{
    const int globalRow = get_global_id(0); // Row ID of C (0..M)
    const int globalCol = get_global_id(1); // Col ID of C (0..N)
	if (globalCol < N && globalRow < M)
	{
		int offset_A = globalRow * K;
		int offset_B = globalCol * K;
		//half16 acc = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
		half16 t16;
		half tf = 0;
		for (int k=0; k<K; k++)
		{
			vstore_half16(A[offset_A + k], 0, (half*)&t16);
			tf += dot(t16, B[offset_B + k]);
		}
		
		//half tf = acc.s0+acc.s1+acc.s2+acc.s3 +acc.s4+acc.s5+acc.s6+acc.s7;
		
		//half tf = acc.s0 +acc.s1+acc.s2+acc.s3 +acc.s4+acc.s5+acc.s6+acc.s7;
		//half tf2 = acc.s8+acc.s9 +acc.sa +acc.sb +acc.sc+acc.sd+acc.se+acc.sf;
		//tf += tf2;
		C[globalRow*N + globalCol] = vload_half(0, &tf);
	}
}
*/
/*
__kernel void myGEMM1TransHalf_v0(const int M, const int N, const int K,
                      const __global half8* A,
                      const __global half8* B,
                      __global float* C)
{
    const int globalRow = get_global_id(0); // Row ID of C (0..M)
    const int globalCol = get_global_id(1); // Col ID of C (0..N)
	if (globalCol < N && globalRow < M)
	{
		half8 acc = { 0, 0, 0, 0, 0, 0, 0, 0};
		
		for (int k=0; k<K; k++)
		{
			acc += A[globalRow * K + k] * B[globalCol*K + k];
		}
		
		half tf = acc.s0+acc.s1+acc.s2+acc.s3 +acc.s4+acc.s5+acc.s6+acc.s7;
		C[globalRow*N + globalCol] = vload_half(0, &tf);
	}
}
*/
/*
__kernel void myGEMM1TransHalf_v1(const int M, const int N, const int K,
                      const __global half8* A,
                      const __global half8* B,
                      __global half* C)
{
    const int globalRow = get_global_id(0); // Row ID of C (0..M)
    const int globalCol = get_global_id(1); // Col ID of C (0..N)
	if (globalCol < N && globalRow < M)
	{
		half8 acc = { 0, 0, 0, 0, 0, 0, 0, 0};
		for (int k=0; k<K; k++)
		{
			acc += A[globalRow * K + k] * B[globalCol*K + k];
		}
		
		half sum = acc.s0+acc.s1+acc.s2+acc.s3 +acc.s4+acc.s5+acc.s6+acc.s7
		C[globalRow*N + globalCol] = sum;//vload_half(0, &sum);
	}
}*/


__kernel void myGEMM1Col(const int M, const int N, const int K,
                      const __global float* A,
                      const __global float* B,
                      __global float* C) 
{
    
    // Thread identifiers
    const int globalRow = get_global_id(0); // Row ID of C (0..M)
    const int globalCol = get_global_id(1); // Col ID of C (0..N)
 
    // Compute a single element (loop over K)
    float acc = 0.0f;
    for (int k=0; k<K; k++) {
        acc += A[k*M + globalRow] * B[globalCol*K + k];
    }
 
    // Store the result
    C[globalCol*M + globalRow] = acc;
}
/*
__kernel void myGEMM1(const int M, const int N, const int K,
                      const __global float* A,
                      const __global float* B,
                      __global float* C) 
{
    
    // Thread identifiers
    const int globalRow = get_global_id(0); // Row ID of C (0..M)
    const int globalCol = get_global_id(1); // Col ID of C (0..N)
 
    // Compute a single element (loop over K)
	if (globalCol < N && globalRow < M)
	{
		float acc = 0.0f;
		for (int k=0; k<K; k++) {
			acc += A[globalRow * K + k] * B[k * N + globalCol];
		}
 
		// Store the result
		C[globalRow*N + globalCol] = acc;
	}
}
*/

__kernel void myGEMM1(const int M, const int N, const int K,
                      const __global float4* A,
                      const __global float* B,
                      __global float* C) 
{
    
    // Thread identifiers
    const int globalRow = get_global_id(0); // Row ID of C (0..M)
    const int globalCol = get_global_id(1); // Col ID of C (0..N)

    // Compute a single element (loop over K)
	if (globalCol < N && globalRow < M)
	{
		float4 acc = { 0.0f, 0.0f, 0.0f, 0.0f };
		float4 tb;
		for (int k=0; k<K; k++)
		{
			tb.s0 = B[4*k * N + globalCol];
			tb.s1 = B[(4*k+1) * N + globalCol];
			tb.s2 = B[(4*k+2) * N + globalCol];
			tb.s3 = B[(4*k+3) * N + globalCol];
			acc += A[globalRow * K + k] * tb;
		}
 
		// Store the result
		C[globalRow*N + globalCol] = acc.s0+acc.s1+acc.s2+acc.s3;
	}
}

__kernel void myGEMM1Trans(const int M, const int N, const int K,
                      const __global float* A,
                      const __global float* B,
                      __global float* C) 
{
    // Thread identifiers
    const int globalRow = get_global_id(0); // Row ID of C (0..M)
    const int globalCol = get_global_id(1); // Col ID of C (0..N)
	if (globalCol < N && globalRow < M)
	{
		float acc = 0.0f;
		for (int k=0; k<K; k++)
		{
			acc += A[globalRow * K + k] * B[globalCol*K + k];
		}
 
		C[globalRow*N + globalCol] = acc;
	}


}

__kernel void myGEMM1TransVec(const int M, const int N, const int K,
                      const __global float4* A,
                      const __global float4* B,
                      __global float* C) 
{
    // Thread identifiers
    const int globalRow = get_global_id(0); // Row ID of C (0..M)
    const int globalCol = get_global_id(1); // Col ID of C (0..N)
	float4 c0 = { 0.0f, 0.0f, 0.0f, 0.0f };
	if (globalCol < N && globalRow < M)
	{
		float4 acc = { 0.0f, 0.0f, 0.0f, 0.0f };
		for (int k=0; k<K; k++)
		{
			acc += mad(A[globalRow * K + k], B[globalCol*K + k], c0);
		}
 
		C[globalRow*N + globalCol] = acc.s0+acc.s1+acc.s2+acc.s3;
	}
}

/*
__kernel void myGEMM1TransVec(const int M, const int N, const int K,
                      const __global float4* A,
                      const __global float4* B,
                      __global float* C) 
{
    // Thread identifiers
    const int globalRow = get_global_id(0); // Row ID of C (0..M)
    const int globalCol = get_global_id(1); // Col ID of C (0..N)
	if (globalCol < N && globalRow < M)
	{
		float4 acc = { 0.0f, 0.0f, 0.0f, 0.0f };
		for (int k=0; k<K; k++)
		{
			acc += A[globalRow * K + k] * B[globalCol*K + k];
		}
 
		C[globalRow*N + globalCol] = acc.s0+acc.s1+acc.s2+acc.s3;
	}
}
*/


)CLSRC";

#else

const char* cl_gemm_str= R"CLSRC(
#pragma OPENCL EXTENSION cl_arm_printf : enable
__kernel void myGEMM1Col(const int M, const int N, const int K,
                      const __global float* A,
                      const __global float* B,
                      __global float* C) 
{
    
    // Thread identifiers
    const int globalRow = get_global_id(0); // Row ID of C (0..M)
    const int globalCol = get_global_id(1); // Col ID of C (0..N)
 
    // Compute a single element (loop over K)
    if (globalRow < M && globalCol < N) {
        float acc = 0.0f;
        for (int k=0; k<K; k++) {
            acc += A[k*M + globalRow] * B[globalCol*K + k];
        }
 
        // Store the result
        C[globalCol*M + globalRow] = acc;
    }
}

__kernel void myGEMM1Row(const int M, const int N, const int K,
                      const __global float* A,
                      const __global float* B,
                      __global float* C)
{

    // Thread identifiers
    const int globalRow = get_global_id(0); // Row ID of C (0..M)
    const int globalCol = get_global_id(1); // Col ID of C (0..N)

    //printf("GEMM_ROW: %d %d\n", globalRow, globalCol);
    // Compute a single element (loop over K)
    if (globalRow < M && globalCol < N) {
        float acc = 0.0f;
        for (int k=0; k<K; k++) {
            acc += A[globalRow * K + k] * B[k * N + globalCol];
        }

        // Store the result
        C[globalRow * N + globalCol] = acc;
    }
}


__kernel void myGEMM1(const int M, const int N, const int K,
                      const __global float4* A,
                      const __global float* B,
                      __global float* C) 
{
    
    // Thread identifiers
    const int globalRow = get_global_id(0); // Row ID of C (0..M)
    const int globalCol = get_global_id(1); // Col ID of C (0..N)

    // Compute a single element (loop over K)
	if (globalCol < N && globalRow < M)
	{
		float4 acc = { 0.0f, 0.0f, 0.0f, 0.0f };
		float4 tb;
		for (int k=0; k<K; k++)
		{
			tb.s0 = B[4*k * N + globalCol];
			tb.s1 = B[(4*k+1) * N + globalCol];
			tb.s2 = B[(4*k+2) * N + globalCol];
			tb.s3 = B[(4*k+3) * N + globalCol];
			acc += A[globalRow * K + k] * tb;
		}
 
		// Store the result
		C[globalRow*N + globalCol] = acc.s0+acc.s1+acc.s2+acc.s3;
	}
}

__kernel void myGEMM1Trans(const int M, const int N, const int K,
                      const __global float* A,
                      const __global float* B,
                      __global float* C) 
{
    // Thread identifiers
    const int globalRow = get_global_id(0); // Row ID of C (0..M)
    const int globalCol = get_global_id(1); // Col ID of C (0..N)
	if (globalCol < N && globalRow < M)
	{
		float acc = 0.0f;
		for (int k=0; k<K; k++)
		{
			acc += A[globalRow * K + k] * B[globalCol*K + k];
		}
 
		C[globalRow*N + globalCol] = acc;
	}


}

__kernel void myGEMM1TransVec(const int M, const int N, const int K,
                      const __global float4* A,
                      const __global float4* B,
                      __global float* C) 
{
    // Thread identifiers
    const int globalRow = get_global_id(0); // Row ID of C (0..M)
    const int globalCol = get_global_id(1); // Col ID of C (0..N)
	float4 c0 = { 0.0f, 0.0f, 0.0f, 0.0f };
    printf("Hello xshuo for log opencl kernel");
    printf("%d %d %d", M, N, K);
	if (globalCol < N && globalRow < M)
	{
		float4 acc = { 0.0f, 0.0f, 0.0f, 0.0f };
		for (int k=0; k<K; k++)
		{
			acc += mad(A[globalRow * K + k], B[globalCol*K + k], c0);
		}
 
		C[globalRow*N + globalCol] = acc.s0+acc.s1+acc.s2+acc.s3;
	}
}
)CLSRC";

#endif
