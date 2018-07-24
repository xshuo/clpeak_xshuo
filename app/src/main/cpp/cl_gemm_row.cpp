//
// Created by xieshuo on 2018/7/23.
//
#include "include/clpeak.h"
#define MAX_M 1024
#define MAX_K 1024
#define MAX_N 1024

#define TEST_CNT 3

int
clPeak::runGemmRow(cl::CommandQueue &queue, cl::Program &prog, device_info_t &devInfo) {
    Timer timer;
    float timed;
    float* input_A = NULL;
    float* input_B = NULL;
    float* output_C = NULL;
    cl::NDRange globalSize, localSize;

    LOGD("Gemm Row Major format");
    cl::Context ctx = queue.getInfo<CL_QUEUE_CONTEXT>();
    try {
        input_A = new float[MAX_M * MAX_K];
        input_B = new float[MAX_K * MAX_N];
        output_C = new float[MAX_M * MAX_N];
        populate(input_A, MAX_M * MAX_K);
        populate(input_B, MAX_K * MAX_N);
        memset(output_C, 0, sizeof(output_C));

        //for (int i = 0; i < MAX_M * MAX_N; i++) {
        //    output_C[i] = i;
        //}
        //log->log2File(output_C, MAX_M, MAX_N);
        //log->log2File(input_A, MAX_M, MAX_K);

        //test on cpu first
        //for (int i = 0; i < MAX_M; i++) {
        //    for (int j = 0; j < MAX_N; j++) {
        //        float tmp = 0;
        //        for (int k = 0; k < MAX_K; k++) {
        //            tmp += input_A[i * MAX_K + k] * input_B[k * MAX_K + j];
        //        }
        //        output_C[i * MAX_N + j] = tmp;
        //    }
        //}
        //log->log2File(output_C, MAX_M, MAX_N);


        cl::Buffer BufA = cl::Buffer(ctx, CL_MEM_READ_ONLY, MAX_M * MAX_K * sizeof(float));
        cl::Buffer BufB = cl::Buffer(ctx, CL_MEM_READ_ONLY, MAX_K * MAX_N * sizeof(float));
        cl::Buffer BufC = cl::Buffer(ctx, CL_MEM_WRITE_ONLY, MAX_M * MAX_N * sizeof(float));

        for (int i = 0; i < TEST_CNT; i++) {
            timer.start();
            queue.enqueueWriteBuffer(BufA, CL_TRUE, 0, MAX_M * MAX_K * sizeof(float), input_A);
            queue.finish();
            timed = timer.stopAndTime();
            LOGD("m*k enqueue write buffer: %lf ms", timed / 1e6f);

            timer.start();
            queue.enqueueWriteBuffer(BufB, CL_TRUE, 0, MAX_K * MAX_N * sizeof(float), input_B);
            queue.finish();
            timed = timer.stopAndTime();
            LOGD("k*n enqueue write buffer: %lf ms", timed / 1e6f);

            cl::Kernel kernel_gemm(prog, "myGEMM1Row");
            kernel_gemm.setArg(0, MAX_M);
            kernel_gemm.setArg(1, MAX_N);
            kernel_gemm.setArg(2, MAX_K);
            kernel_gemm.setArg(3, BufA);
            kernel_gemm.setArg(4, BufB);
            kernel_gemm.setArg(5, BufC);

            globalSize = cl::NDRange(MAX_M, MAX_N);
            localSize = cl::NullRange;

            int dim = globalSize.dimensions();
            LOGI("global dim: %u", dim);

            timer.start();
            queue.enqueueNDRangeKernel(kernel_gemm, cl::NullRange, globalSize, localSize);
            queue.finish();
            timed = timer.stopAndTime();
            LOGI("gemm compute: %lf ms", timed / 1e6f);

            timer.start();
            queue.enqueueReadBuffer(BufC, CL_TRUE, 0, MAX_M * MAX_N * sizeof(float), output_C);
            queue.finish();
            timed = timer.stopAndTime();
            log->log2File(output_C, MAX_M, MAX_N);
            LOGI("gemm compute read buffer m*n: %lf ms", timed / 1e6f);
            memcpy(input_A, output_C, sizeof(output_C));
        }

        if (input_A) delete[] input_A;
        if (input_B) delete[] input_B;
        if (output_C) delete[] output_C;
    } catch (cl::Error error) {
        stringstream ss;
        ss << error.what() << " (" << error.err() << ")" NEWLINE
           << TAB TAB TAB "Tests skipped" NEWLINE;
        log->print(ss.str());

        if (input_A) delete[] input_A;
        if (input_B) delete[] input_B;
        if (output_C) delete[] output_C;
        return -1;
    }
    return 0;
}

