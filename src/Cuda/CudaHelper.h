#ifndef CUDA_HELPER_H
#define CUDA_HELPER_H

#include <string>
#include <vector>
#include <iostream>
#include <sstream>
#include <climits>
#include <algorithm>
#include <fstream>

#include <cuda.h>
#include <cuda_runtime_api.h>

bool cudaPrintErrorExit(cudaError msg_id, const char *msg); // check error
bool cudaPrintError(cudaError err_num, const char *text, std::ostream *error_stream = &std::cerr);
bool cudaPrintInfo(cudaError err_num, const char *text, std::ostream *error_stream = &std::cerr);

double getCudaEventTime(cudaEvent_t start, cudaEvent_t stop);

#endif