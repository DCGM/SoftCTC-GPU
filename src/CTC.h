#ifndef CTC_H
#define CTC_H

#define _USE_MATH_DEFINES
#include <math.h>
#include <iostream>
#include <array>

#include "TensorND.h"

#ifdef USE_PYBIND
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "PyBindHelper.h"

namespace py = pybind11;
#endif

#ifdef USE_TORCH
#include <torch/torch.h>
#ifdef USE_PYBIND
#include <torch/extension.h>
#endif
#endif

template <typename T>
class CTC {

public:
	CTC();
	virtual ~CTC();

#ifdef USE_TORCH
	bool calcCTCTorch(torch::Tensor& grads, torch::Tensor& loss,
		const torch::Tensor& forward, const torch::Tensor& forward_start, const torch::Tensor& forward_end,
		const torch::Tensor& backward, const torch::Tensor& backward_start, const torch::Tensor& backward_end,
		const torch::Tensor& logits, const torch::Tensor& labels,
		unsigned int norm_step, bool zero_infinity = false);
#endif

#ifdef USE_PYBIND

	bool calcCTCPy(py::array_t<T, py::array::c_style>& grads, py::array_t<T, py::array::c_style>& loss,
		const py::array_t<T, py::array::c_style | py::array::forcecast>& forward, const py::array_t<T, py::array::c_style | py::array::forcecast>& forward_start, const py::array_t<T, py::array::c_style | py::array::forcecast>& forward_end,
		const py::array_t<T, py::array::c_style | py::array::forcecast>& backward, const py::array_t<T, py::array::c_style | py::array::forcecast>& backward_start, const py::array_t<T, py::array::c_style | py::array::forcecast>& backward_end,
		const py::array_t<T, py::array::c_style | py::array::forcecast>& logits, const py::array_t<int, py::array::c_style | py::array::forcecast>& labels, 
		unsigned int norm_step, bool zero_infinity = false);
#endif	

	bool calcCTC(std::vector<T>& grads, std::vector<T>& loss, 
				 const std::vector<T> &forward, const std::vector<T> &forward_start, const std::vector<T> &forward_end,
				 const std::vector<T> &backward, const std::vector<T> &backward_start, const std::vector<T> &backward_end,
				 const std::vector<T> &logits, const std::vector<int> &labels,
				 unsigned int time_size, unsigned int vector_size, unsigned int logit_size, unsigned int batch_size, unsigned int norm_step, bool zero_infinity = false);
	bool calcCTC(TensorND<T, 3>& grads, TensorND<T, 1>& loss,
		const TensorND<T, 3>& forward, const TensorND<T, 2>& forward_start, const TensorND<T, 2>& forward_end,
		const TensorND<T, 3>& backward, const TensorND<T, 2>& backward_start, const TensorND<T, 2>& backward_end,
		const TensorND<T, 3>& logits, const TensorND<int, 2>& labels, unsigned int norm_step, bool zero_infinity = false);
	bool isValid();
	virtual void printDeviceInfo() { return; };

	bool copyFromDeviceDebug(TensorND<T, 3>& forward, TensorND<T, 2>& forward_start, TensorND<T, 2>& forward_end,
		TensorND<T, 3>& backward, TensorND<T, 2>& backward_start, TensorND<T, 2>& backward_end,
		TensorND<T, 3>& logits, TensorND<int, 2>& labels, 
		TensorND<T, 3>& grads, TensorND<T, 1>& loss, TensorND<T, 3>& alphas, TensorND<T, 3>& betas, TensorND<T, 3>& probs, TensorND<T, 3>& full_probs, TensorND<T, 1>& ll_backward);
	bool copyFromDeviceDebug(std::vector<T>& forward, std::vector<T>& forward_start, std::vector<T>& forward_end,
		std::vector<T>& backward, std::vector<T>& backward_start, std::vector<T>& backward_end,
		std::vector<T>& logits, std::vector<int>& labels,
		std::vector<T>& grads, std::vector<T>& loss, std::vector<T>& alphas, std::vector<T>& betas, std::vector<T>& probs, std::vector<T>& full_probs, std::vector<T>& ll_backward);

protected:
	virtual bool copyFromDeviceDebugCore(T* forward, T* forward_start, T* forward_end,
		T* backward, T* backward_start, T* backward_end,
		T* logits, int* labels,
		T* grads, T* loss, T* alphas, T* betas, T* probs, T* full_probs, T* ll_backward) { return false; };
	virtual bool calcCTCCore(T *grads, T *loss, 
		const T *forward, const T *forward_start, const T *forward_end,
		const T *backward, const T *backward_start, const T *backward_end,
		const T *logits, const int *labels,
		unsigned int time_size, unsigned int vector_size, unsigned int logit_size, unsigned int batch_size, unsigned int norm_step, bool zero_infinity = false, bool native = false) {return false;};

	unsigned int time_size;
	unsigned int vector_size;
	unsigned int logit_size;
	unsigned int batch_size;

	bool valid;
	bool native_cuda;
};

template <typename T>
CTC<T>::CTC()
{
	this->valid = false;
	this->time_size = 0;
	this->batch_size = 0;
	this->vector_size = 0;
	this->logit_size = 0;
	this->native_cuda = false;
}

template <typename T>
CTC<T>::~CTC()
{
}

template <typename T>
bool CTC<T>::isValid()
{
	return valid;
}

template <typename T>
bool CTC<T>::copyFromDeviceDebug(TensorND<T, 3>& forward, TensorND<T, 2>& forward_start, TensorND<T, 2>& forward_end, 
	TensorND<T, 3>& backward, TensorND<T, 2>& backward_start, TensorND<T, 2>& backward_end, 
	TensorND<T, 3>& logits, TensorND<int, 2>& labels, 
	TensorND<T, 3>& grads, TensorND<T, 1>& loss, TensorND<T, 3>& alphas, TensorND<T, 3>& betas, TensorND<T, 3>& probs, TensorND<T, 3>& full_probs, TensorND<T, 1>& ll_backward)
{
	std::array<size_t, 3> part_sizes = { vector_size, batch_size, time_size };
	std::array<size_t, 3> full_sizes = { logit_size, batch_size, time_size };
	std::array<size_t, 3> connection_sizes = { vector_size, vector_size, batch_size };
	std::array<size_t, 2> label_sizes = { vector_size, batch_size };
	std::array<size_t, 1> batch_sizes = { batch_size };
	alphas.resize(part_sizes);
	betas.resize(part_sizes);
	probs.resize(part_sizes);
	full_probs.resize(full_sizes);
	ll_backward.resize(batch_sizes);
	grads.resize(full_sizes);
	loss.resize(batch_sizes);
	forward.resize(connection_sizes);
	forward_start.resize(label_sizes);
	forward_end.resize(label_sizes);
	backward.resize(connection_sizes);
	backward_start.resize(label_sizes);
	backward_end.resize(label_sizes);
	logits.resize(full_sizes);
	labels.resize(label_sizes);
	bool result = copyFromDeviceDebugCore(forward.data.data(), forward_start.data.data(), forward_end.data.data(), backward.data.data(), backward_start.data.data(), backward_end.data.data(), logits.data.data(), labels.data.data(), grads.data.data(), loss.data.data(), alphas.data.data(), betas.data.data(), probs.data.data(), full_probs.data.data(), ll_backward.data.data());
	return result;
}

template <typename T>
bool CTC<T>::copyFromDeviceDebug(std::vector<T>& forward, std::vector<T>& forward_start, std::vector<T>& forward_end,
	std::vector<T>& backward, std::vector<T>& backward_start, std::vector<T>& backward_end,
	std::vector<T>& logits, std::vector<int>& labels,
	std::vector<T>& grads, std::vector<T>& loss, std::vector<T>& alphas, std::vector<T>& betas, std::vector<T>& probs, std::vector<T>& full_probs, std::vector<T>& ll_backward)
{
	size_t part_sizes = vector_size + batch_size + time_size;
	size_t full_sizes = logit_size + batch_size + time_size;
	size_t connection_sizes = vector_size + vector_size + batch_size;
	size_t label_sizes = vector_size + batch_size;
	size_t batch_sizes = batch_size;
	alphas.resize(part_sizes);
	betas.resize(part_sizes);
	probs.resize(part_sizes);
	full_probs.resize(full_sizes);
	ll_backward.resize(batch_sizes);
	grads.resize(full_sizes);
	loss.resize(batch_sizes);
	forward.resize(connection_sizes);
	forward_start.resize(label_sizes);
	forward_end.resize(label_sizes);
	backward.resize(connection_sizes);
	backward_start.resize(label_sizes);
	backward_end.resize(label_sizes);
	logits.resize(full_sizes);
	labels.resize(label_sizes);
	return copyFromDeviceDebugCore(forward.data(), forward_start.data(), forward_end.data(), backward.data(), backward_start.data(), backward_end.data(), logits.data(), labels.data(), grads.data(), loss.data(), alphas.data(), betas.data(), probs.data(), full_probs.data(), ll_backward.data());
}

template <typename T>
bool CTC<T>::calcCTC(std::vector<T>& grads, std::vector<T>& loss,
	const std::vector<T>& forward, const std::vector<T>& forward_start, const std::vector<T>& forward_end,
	const std::vector<T>& backward, const std::vector<T>& backward_start, const std::vector<T>& backward_end,
	const std::vector<T>& logits, const std::vector<int>& labels, unsigned int time_size, unsigned int vector_size, unsigned int logit_size, unsigned int batch_size, unsigned int norm_step, bool zero_infinity)
{
	grads.resize(time_size * batch_size * logit_size);
	loss.resize(batch_size);
	return this->calcCTCCore(grads.data(), loss.data(), forward.data(), forward_start.data(), forward_end.data(), backward.data(), backward_start.data(), backward_end.data(), logits.data(), labels.data(), time_size, vector_size, logit_size, batch_size, norm_step, zero_infinity);
}

template <typename T>
bool CTC<T>::calcCTC(TensorND<T, 3>& grads, TensorND<T, 1>& loss,
	const TensorND<T, 3>& forward, const TensorND<T, 2>& forward_start, const TensorND<T, 2>& forward_end,
	const TensorND<T, 3>& backward, const TensorND<T, 2>& backward_start, const TensorND<T, 2>& backward_end,
	const TensorND<T, 3>& logits, const TensorND<int, 2>& labels, unsigned int norm_step, bool zero_infinity)
{
	unsigned int time_size = logits.sizes[2];
	unsigned int vector_size = forward.sizes[1];
	unsigned int logit_size = logits.sizes[0];
	unsigned int batch_size = logits.sizes[1];
	std::array<size_t, 3> full_sizes = { logit_size, batch_size, time_size };
	std::array<size_t, 3> connection_sizes = { vector_size, vector_size, batch_size };
	std::array<size_t, 2> label_sizes = { vector_size, batch_size };
	std::array<size_t, 1> batch_sizes = { batch_size };

	if ((forward.sizes != connection_sizes) || (forward_start.sizes != label_sizes) || (forward_end.sizes != label_sizes) ||
		(backward.sizes != connection_sizes) || (backward_start.sizes != label_sizes) || (backward_end.sizes != label_sizes) ||
		(logits.sizes != full_sizes) || (labels.sizes != label_sizes))
	{
		std::cerr << "Error: CTC<T>::calcCTC: Invalid tensor sizes." << std::endl;
		return false;
	}
	grads.resize(full_sizes);
	loss.resize(batch_sizes);
	return this->calcCTCCore(grads.data.data(), loss.data.data(), forward.data.data(), forward_start.data.data(), forward_end.data.data(), backward.data.data(), backward_start.data.data(), backward_end.data.data(), logits.data.data(), labels.data.data(), time_size, vector_size, logit_size, batch_size, norm_step, zero_infinity);
}


#ifdef USE_TORCH
template <typename T>
bool CTC<T>::calcCTCTorch(torch::Tensor& grads, torch::Tensor& loss,
	const torch::Tensor& forward, const torch::Tensor& forward_start, const torch::Tensor& forward_end,
	const torch::Tensor& backward, const torch::Tensor& backward_start, const torch::Tensor& backward_end,
	const torch::Tensor& logits, const torch::Tensor& labels,
	unsigned int norm_step, bool zero_infinity)
{
	bool use_cuda_buffer = native_cuda && grads.device().is_cuda() && loss.device().is_cuda() &&
		forward.device().is_cuda() && forward_start.device().is_cuda() && forward_end.device().is_cuda() &&
		backward.device().is_cuda() && backward_start.device().is_cuda() && backward_end.device().is_cuda() &&
		logits.device().is_cuda() && labels.device().is_cuda();
	
	if ((!use_cuda_buffer) && ((!grads.device().is_cpu()) || (!loss.device().is_cpu()) ||
		(!forward.device().is_cpu()) || (!forward_start.device().is_cpu()) || (!forward_end.device().is_cpu()) ||
		(!backward.device().is_cpu()) || (!backward_start.device().is_cpu()) || (!backward_end.device().is_cpu()) ||
		(!logits.device().is_cpu()) || (!labels.device().is_cpu())))
	{
		std::cerr << "Error: CTC<T>::calcCTCTorch: Unsupported buffer types." << std::endl;
		std::cerr << "Buffers types (1 for GPU type) " << std::endl <<
			" grads: " << grads.device().is_cuda() << " loss: " << loss.device().is_cuda() << std::endl <<
			" forward: " << forward.device().is_cuda() << " forward_start: " << forward_start.device().is_cuda() << " forward_end: " << forward_end.device().is_cuda() << std::endl <<
			" backward: " << backward.device().is_cuda() << " backward_start: " << backward_start.device().is_cuda() << " backward_end: " << backward_end.device().is_cuda() << std::endl <<
			" logits: " << logits.device().is_cuda() << " labels: " << labels.device().is_cuda() << std::endl;
		return false;
	}
	if ((!grads.is_contiguous()) || (!loss.is_contiguous()) ||
		(!forward.is_contiguous()) || (!forward_start.is_contiguous()) || (!forward_end.is_contiguous()) ||
		(!backward.is_contiguous()) || (!backward_start.is_contiguous()) || (!backward_end.is_contiguous()) ||
		(!logits.is_contiguous()) || (!labels.is_contiguous()))
	{
		std::cerr << "Error: CTC<T>::calcCTCTorch: Buffers is not contiguous." << std::endl;
		return false;
	}
	if ((logits.dim() != 3) || (forward.dim() != 3) || (forward_start.dim() != 2) || (loss.dim() != 1))
	{
		std::cerr << "Error: CTC<T>::calcCTCTorch: Invalid tensor dimensions." << std::endl;
		return false;
	}
	unsigned int time_size = logits.size(0);
	unsigned int vector_size = forward.size(1);
	unsigned int logit_size = logits.size(2);
	unsigned int batch_size = logits.size(1);

	if ((forward.size(0) != batch_size) || (forward.size(2) != vector_size) || (forward_start.size(1) != vector_size) || (loss.size(0) != batch_size) ||
		(!grads.is_same_size(logits)) ||
		(!forward.is_same_size(backward)) ||
		(!forward_start.is_same_size(backward_start)) || (!forward_end.is_same_size(backward_end)) || (!forward_start.is_same_size(forward_end)) || (!forward_start.is_same_size(labels)))
	{
		std::cerr << "Error: CTC<T>::calcCTCTorch: Invalid tensor sizes." << std::endl;
		return false;
	}

	bool result = this->calcCTCCore(grads.data_ptr<T>(), loss.data_ptr<T>(), forward.data_ptr<T>(), forward_start.data_ptr<T>(), forward_end.data_ptr<T>(), backward.data_ptr<T>(), backward_start.data_ptr<T>(), backward_end.data_ptr<T>(), logits.data_ptr<T>(), labels.data_ptr<int>(), time_size, vector_size, logit_size, batch_size, norm_step, zero_infinity, use_cuda_buffer);

	return result;
}
#endif

#ifdef USE_PYBIND
template <typename T>
bool CTC<T>::calcCTCPy(py::array_t<T, py::array::c_style>& grads, py::array_t<T, py::array::c_style>& loss,
	const py::array_t<T, py::array::c_style | py::array::forcecast>& forward, const py::array_t<T, py::array::c_style | py::array::forcecast>& forward_start, const py::array_t<T, py::array::c_style | py::array::forcecast>& forward_end,
	const py::array_t<T, py::array::c_style | py::array::forcecast>& backward, const py::array_t<T, py::array::c_style | py::array::forcecast>& backward_start, const py::array_t<T, py::array::c_style | py::array::forcecast>& backward_end,
	const py::array_t<T, py::array::c_style | py::array::forcecast>& logits, const py::array_t<int, py::array::c_style | py::array::forcecast>& labels,
	unsigned int norm_step, bool zero_infinity)
{
	if ((logits.ndim() != 3) || (forward.ndim() != 3))
	{
		std::cerr << "Error: CTC<T>::calcCTC: Invalid tensor dimensions." << std::endl;
		return false;
	}
	unsigned int time_size = logits.shape(0);
	unsigned int vector_size = forward.shape(1);
	unsigned int logit_size = logits.shape(2);
	unsigned int batch_size = logits.shape(1);
	std::vector<size_t> part_sizes = { time_size, batch_size, vector_size };
	std::vector<size_t> full_sizes = { time_size, batch_size, logit_size };
	std::vector<size_t> connection_sizes = { batch_size, vector_size, vector_size };
	std::vector<size_t> label_sizes = { batch_size, vector_size };
	std::vector<size_t> batch_sizes = { batch_size };

	if ((forward != connection_sizes) || (forward_start != label_sizes) || (forward_end != label_sizes) ||
		(backward != connection_sizes) || (backward_start != label_sizes) || (backward_end != label_sizes) ||
		(logits != full_sizes) || (labels != label_sizes))
	{
		std::cerr << "Error: CTC<T>::calcCTC: Invalid tensor sizes." << std::endl;
		return false;
	}
	grads.resize(full_sizes);
	loss.resize(batch_sizes);
	bool result = this->calcCTCCore(grads.mutable_data(), loss.mutable_data(), forward.data(), forward_start.data(), forward_end.data(), backward.data(), backward_start.data(), backward_end.data(), logits.data(), labels.data(),  time_size, vector_size, logit_size, batch_size, norm_step, zero_infinity);

	return result;
}

#endif

#endif
