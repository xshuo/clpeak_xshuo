#ifndef CLPEAK_HPP
#define CLPEAK_HPP

#define __CL_ENABLE_EXCEPTIONS

#include <CL/cl.hpp>

#include <iostream>
#include <stdio.h>
#include <iomanip>
#include <string.h>
#include <sstream>
#include "common.h"
#include "logger.h"

#include <android/log.h>
#define LOG_TAG "XSHUO_CLPEAK"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,LOG_TAG,__VA_ARGS__)
#define LOGD(...) __android_log_print(ANDROID_LOG_INFO,LOG_TAG,__VA_ARGS__)

#define BUILD_OPTIONS           " -cl-mad-enable "
#define BUILD_GEMM_OPTIONS "-cl-strict-aliasing -cl-mad-enable -cl-no-signed-zeros -cl-fast-relaxed-math"
/*********************************
* cl_arm_printf extension
*********************************/
#define CL_PRINTF_CALLBACK_ARM                      0x40B0
#define CL_PRINTF_BUFFERSIZE_ARM                    0x40B1


using namespace std;

class clPeak
{
public:

  bool forcePlatform, forceDevice, useEventTimer;
  bool isGlobalBW, isComputeSP, isComputeDP, isComputeInt, isTransferBW, isKernelLatency;
  int specifiedPlatform, specifiedDevice;
  logger *log;

  clPeak();
  ~clPeak();

  int parseArgs(int argc, char **argv);
  void static print_callback(const char *buffer, size_t length, size_t final, void *user_data);

  // Return avg time in us
  float run_kernel(cl::CommandQueue &queue, cl::Kernel &kernel, cl::NDRange &globalSize, cl::NDRange &localSize, int iters);

  int runGemmTest(cl::CommandQueue &queue, cl::Program &prog, device_info_t &devInfo);
  int runGemmRow(cl::CommandQueue &queue, cl::Program &prog, device_info_t &devInfo);

  int runGlobalBandwidthTest(cl::CommandQueue &queue, cl::Program &prog, device_info_t &devInfo);

  int runComputeSP(cl::CommandQueue &queue, cl::Program &prog, device_info_t &devInfo);

  int runComputeHP(cl::CommandQueue &queue, cl::Program &prog, device_info_t &devInfo);

  int runComputeDP(cl::CommandQueue &queue, cl::Program &prog, device_info_t &devInfo);

  int runComputeInteger(cl::CommandQueue &queue, cl::Program &prog, device_info_t &devInfo);

  int runTransferBandwidthTest(cl::CommandQueue &queue, cl::Program &prog, device_info_t &devInfo);

  int runKernelLatency(cl::CommandQueue &queue, cl::Program &prog, device_info_t &devInfo);

  int runAll();

  int runUnitTest();
};

#endif  // CLPEAK_HPP
