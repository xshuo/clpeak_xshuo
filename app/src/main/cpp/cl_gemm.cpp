//
// Created by xieshuo on 2018/7/19.
//
#include "include/clpeak.h"
//#define MAX_M 512
//#define MAX_K 2048
//#define MAX_N 40960
#define MAX_M 2
#define MAX_K 8
#define MAX_N 2
#define TEST_CNT 1

int
clPeak::runGemmTest(cl::CommandQueue &queue, cl::Program &prog, device_info_t &devInfo) {
    Timer timer;
    float timed;
    float* input_A = NULL;
    float* input_B = NULL;
    float* output_C = NULL;
    cl::NDRange globalSize, localSize;

    cl::Context ctx = queue.getInfo<CL_QUEUE_CONTEXT>();
    try {
        input_A = new float[MAX_M * MAX_K] {
                1, 2, 3, 4, 5, 6, 7, 8,
                8, 7, 6, 5, 4, 3, 2, 1
        };
        input_B = new float[MAX_K * MAX_N] {
                1, 2,
                3, 4,
                5, 6,
                7, 8,
                8, 7,
                6, 5,
                4, 3,
                2, 1
        };
        output_C = new float[MAX_M * MAX_N] {
                0, 0,
                0, 0
        };
        //populate(input_A, MAX_M * MAX_K);
        //populate(input_B, MAX_K * MAX_N);
        for (int i = 0; i < 16; i++) {
            LOGD("%lf", input_A[i]);
        }
        for (int i = 0; i < 4; i++) {
            LOGD("%lf", output_C[i]);
        }
        cl::Buffer BufA = cl::Buffer(ctx, CL_MEM_READ_ONLY, MAX_M * MAX_K * sizeof(float));
        cl::Buffer BufB = cl::Buffer(ctx, CL_MEM_READ_ONLY, MAX_K * MAX_N * sizeof(float));
        cl::Buffer BufC = cl::Buffer(ctx, CL_MEM_WRITE_ONLY, MAX_M * MAX_N * sizeof(float));

        timer.start();
        for (int i = 0; i < TEST_CNT; i++) {
            queue.enqueueWriteBuffer(BufA, CL_TRUE, 0, MAX_M * MAX_K * sizeof(float), input_A);
            queue.finish();
        }
        timed = timer.stopAndTime();
        LOGD("m*k enqueue write buffer: %lf", timed / TEST_CNT);

        timer.start();
        for (int i = 0; i < TEST_CNT; i++) {
            queue.enqueueWriteBuffer(BufB, CL_TRUE, 0, MAX_K * MAX_N * sizeof(float), input_B);
            queue.finish();
        }
        timed = timer.stopAndTime();
        LOGD("k*n enqueue write buffer: %lf", timed / TEST_CNT);

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
        for (int i = 0; i < TEST_CNT; i++) {
            queue.enqueueNDRangeKernel(kernel_gemm, cl::NullRange, globalSize, localSize);
            queue.finish();
        }
        timed = timer.stopAndTime();
        LOGI("gemm compute: %lf", timed / 10);

        timer.start();
        for (int i = 0; i < TEST_CNT; i++) {
            queue.enqueueReadBuffer(BufC, CL_TRUE, 0, MAX_M * MAX_N * sizeof(float), output_C);
            queue.finish();
        }
        timed = timer.stopAndTime();
        LOGI("gemm compute read buffer m*n: %lf", timed / TEST_CNT);
        for (int i = 0; i < 4; i++) {
            LOGD("%lf", output_C[i]);
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

