#ifndef USE_PYBIND
#ifdef USE_OPENCL
#include "OpenCL/CTCOpenCL.h"
#endif
#ifdef USE_CUDA
#include "Cuda/CTCCuda.h"
#endif
#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#define _USE_MATH_DEFINES
#include <math.h>
#include "TensorND.h"

template <typename T>
void test(CTC<T> &ctc, unsigned int iterations)
{
	//TensorND<T, 3> grads_h("grads.csv");
	//TensorND<T, 1> loss_h("loss.csv");
	//TensorND<T, 1> ll_backward_h("ll_backward.csv");
	TensorND<T, 3> forward("forward.csv");
	TensorND<T, 2> forward_start("forward_start.csv");
	TensorND<T, 2> forward_end("forward_end.csv");
	TensorND<T, 3> backward("backward.csv");
	TensorND<T, 2> backward_start("backward_start.csv");
	TensorND<T, 2> backward_end("backward_end.csv");
	TensorND<T, 3> logits("logits.csv");
	TensorND<int, 2> labels("labels.csv");
	//TensorND<T, 3> alphas_h("alphas.csv");
	//TensorND<T, 3> betas_h("betas.csv");
	//TensorND<T, 3> full_probs_h("full_probs.csv");
	//TensorND<T, 3> probs_h("probs.csv");

	std::array<size_t, 3> part_sizes = { labels.sizes[0], logits.sizes[1], logits.sizes[2] };
	std::array<size_t, 3> full_sizes = logits.sizes;
	std::array<size_t, 1> batch_sizes = { labels.sizes[1] };

	TensorND<T, 3> grads_d(full_sizes);
	TensorND<T, 1> loss_d(batch_sizes);
	TensorND<T, 1> ll_backward_d(batch_sizes);
	TensorND<T, 3> alphas_d(part_sizes);
	TensorND<T, 3> betas_d(part_sizes);
	TensorND<T, 3> full_probs_d(full_sizes);
	TensorND<T, 3> probs_d(part_sizes);

	double time;
	for (int i = 0; i < iterations; i++)
	{
		ctc.calcCTC(grads_d, loss_d, forward, forward_start, forward_end, backward, backward_start, backward_end, logits, labels, 10, &time);
		std::cerr << "Compute time " << i << " iteration: " << time << std::endl;
	}
	//ctc->copyFromDeviceDebug(alphas_d, betas_d, probs_d, full_probs_d, ll_backward_d);

	grads_d.save("grads_d.csv");
	/*alphas_d.save("alphas_d.csv");
	betas_d.save("betas_d.csv");
	probs_d.save("probs_d.csv");
	full_probs_d.save("full_probs_d.csv");
	ll_backward_d.save("ll_backward_d.csv");*/
	loss_d.save("loss_d.csv");
}

#ifdef USE_CUDA
template <typename T>
void test_cuda(unsigned int iterations)
{
	CTC<T>* ctc = new CTCCuda<T>(true);
	test(*ctc, iterations);
	delete ctc;
}
#endif

#ifdef USE_OPENCL
template <typename T>
void test_opencl(unsigned int iterations)
{
	CTC<T>* ctc = new CTCOpenCL<T>(true);
	test(*ctc, iterations);
	delete ctc;
}
#endif

int main(int argc, char* argv[])
{
#ifdef USE_CUDA
	std::cerr << std::endl << "Test cuda float" << std::endl;
	test_cuda<float>(10);
	std::cerr << std::endl << "Test cuda double" << std::endl;
	test_cuda<double>(10);
#endif
#ifdef USE_OPENCL
	std::cerr << std::endl << "Test opencl float" << std::endl;
	test_opencl<float>(10);
	std::cerr << std::endl << "Test opencl double" << std::endl;
	test_opencl<double>(10);
#endif
}

#endif