#ifndef CTC_CUDA_H
#define CTC_CUDA_H

#define _USE_MATH_DEFINES
#include <math.h>
#include <iostream>
#include <array>

#include "CudaHelper.h"

#include "../CTC.h"

template <typename T>
class CTCCuda : public CTC<T> {

public:
	CTCCuda(bool split_forward_backward = false, bool compile_static_matrix = true, bool sync_native = false);
	~CTCCuda();

	void printDeviceInfo();

protected:
	bool createBuffers(unsigned int time_size, unsigned int vector_size, unsigned int logit_size, unsigned int batch_size, unsigned int norm_step, bool act_use_native);
	bool copyToDevice(const T* forward, const T* forward_start, const T* forward_end,
		const T* backward, const T* backward_start, const T* backward_end,
		const T* logits, const int* labels, cudaStream_t& stream);
	bool copyFromDevice(T* grads, T* loss, cudaStream_t& stream);
	bool copyFromDeviceDebugCore(T* forward, T* forward_start, T* forward_end,
		T* backward, T* backward_start, T* backward_end,
		T* logits, int* labels,
		T* grads, T* loss, T* alphas, T* betas, T* probs, T* full_probs, T* ll_backward);
	bool calcCTCCore(T *grads, T *loss,
		const T *forward, const T *forward_start, const T *forward_end,
		const T *backward, const T *backward_start, const T *backward_end,
		const T *logits, const int *labels,
		unsigned int time_size, unsigned int vector_size, unsigned int logit_size, unsigned int batch_size, unsigned int norm_step, bool zero_infinity = false, bool native = false);
	bool deleteBuffers();
	void setBuffers(T* grads, T* loss, const T* forward, const T* forward_start, const T* forward_end,
		const T* backward, const T* backward_start, const T* backward_end,
		const T* logits, const int* labels);
	bool initCuda();
	void initBuffers();

	cudaStream_t stream;
#ifdef USE_PROFILLING
	cudaEvent_t ev_copy_from_start, ev_copy_from_stop, ev_copy_to_start, ev_copy_to_stop;
	cudaEvent_t ev_init_start, ev_init_stop, ev_forward_backward_start, ev_forward_backward_stop, ev_forward_start, ev_forward_stop, ev_backward_start, ev_backward_stop, ev_finalize_start, ev_finalize_stop, ev_set_zero_to_invalid_start, ev_set_zero_to_invalid_stop;
#endif

	unsigned int* d_validity_of_batch_ids;
	T *d_grads;
	T *d_loss;
	T *d_forward;
	T *d_forward_start;
	T *d_forward_end;
	T *d_backward;
	T *d_backward_start;
	T *d_backward_end;
	T *d_logits;
	int *d_labels;
	T *d_alphas;
	T *d_betas;
	T *d_full_probs;
	T *d_probs;
	T* d_ll_backward;

	T* d_act_grads;
	T* d_act_loss;
	const T* d_act_forward;
	const T* d_act_forward_start;
	const T* d_act_forward_end;
	const T* d_act_backward;
	const T* d_act_backward_start;
	const T* d_act_backward_end;
	const T* d_act_logits;
	const int* d_act_labels;

	bool split_forward_backward;
	bool compile_static_matrix;
	bool use_native;
	bool sync_native;
};

template <typename T>
CTCCuda<T>::CTCCuda(bool split_forward_backward, bool compile_static_matrix, bool sync_native) : CTC<T>(), split_forward_backward(split_forward_backward), compile_static_matrix(compile_static_matrix), sync_native(sync_native)
{
	initBuffers();
	this->valid = initCuda();
	this->native_cuda = true;
	this->use_native = false;
}

template <typename T>
CTCCuda<T>::~CTCCuda()
{
	deleteBuffers();
#ifdef USE_PROFILLING
	cudaEventDestroy(ev_copy_from_start);
	cudaEventDestroy(ev_copy_from_stop);
	cudaEventDestroy(ev_copy_to_start);
	cudaEventDestroy(ev_copy_to_stop);
	cudaEventDestroy(ev_init_start);
	cudaEventDestroy(ev_init_stop);
	cudaEventDestroy(ev_forward_backward_start);
	cudaEventDestroy(ev_forward_backward_stop);
	cudaEventDestroy(ev_forward_start);
	cudaEventDestroy(ev_forward_stop);
	cudaEventDestroy(ev_backward_start);
	cudaEventDestroy(ev_backward_stop);
	cudaEventDestroy(ev_finalize_start);
	cudaEventDestroy(ev_finalize_stop);
	cudaEventDestroy(ev_set_zero_to_invalid_start);
	cudaEventDestroy(ev_set_zero_to_invalid_stop);
#endif
	cudaStreamDestroy(stream);
}

template <typename T>
void CTCCuda<T>::initBuffers()
{
	d_grads = NULL;
	d_loss = NULL;
	d_ll_backward = NULL;
	d_forward = NULL;
	d_forward_start = NULL;
	d_forward_end = NULL;
	d_backward = NULL;
	d_backward_start = NULL;
	d_backward_end = NULL;
	d_logits = NULL;
	d_labels = NULL;
	d_alphas = NULL;
	d_betas = NULL;
	d_full_probs = NULL;
	d_probs = NULL;
	d_validity_of_batch_ids = NULL;
}

template <typename T>
void CTCCuda<T>::printDeviceInfo()
{
	if (!this->isValid())
	{
		std::cerr << "Error: CTC is not in valid state. Cannot print device info." << std::endl;
		return;
	}
}

template <typename T>
bool CTCCuda<T>::deleteBuffers()
{
	cudaFree(d_grads);
	cudaFree(d_loss);
	cudaFree(d_ll_backward);
	cudaFree(d_forward);
	cudaFree(d_forward_start);
	cudaFree(d_forward_end);
	cudaFree(d_backward);
	cudaFree(d_backward_start);
	cudaFree(d_backward_end);
	cudaFree(d_logits);
	cudaFree(d_labels);
	cudaFree(d_alphas);
	cudaFree(d_betas);
	cudaFree(d_full_probs);
	cudaFree(d_probs);
	cudaFree(d_validity_of_batch_ids);

	d_grads = NULL;
	d_loss = NULL;
	d_ll_backward = NULL;
	d_forward = NULL;
	d_forward_start = NULL;
	d_forward_end = NULL;
	d_backward = NULL;
	d_backward_start = NULL;
	d_backward_end = NULL;
	d_logits = NULL;
	d_labels = NULL;
	d_alphas = NULL;
	d_betas = NULL;
	d_full_probs = NULL;
	d_probs = NULL;
	d_validity_of_batch_ids = NULL;

	return true;
}

template <typename T>
bool CTCCuda<T>::createBuffers(unsigned int time_size, unsigned int vector_size, unsigned int logit_size, unsigned int batch_size, unsigned int norm_step, bool act_use_native)
{
	if ((time_size != this->time_size) || (vector_size != this->vector_size) || (logit_size != this->logit_size) || (this->batch_size != batch_size) || (act_use_native != use_native))
	{
		this->deleteBuffers();
		this->time_size = time_size;
		this->vector_size = vector_size;
		this->logit_size = logit_size;
		this->batch_size = batch_size;

		unsigned int part_sizes = time_size * vector_size * batch_size;
		unsigned int full_sizes = time_size * logit_size * batch_size;
		unsigned int label_sizes = vector_size * batch_size;
		unsigned int connection_sizes = vector_size * vector_size * batch_size;
		if (act_use_native)
		{
			if ((!cudaPrintError(cudaMalloc(&d_ll_backward, batch_size * sizeof(T)), "Create buffer d_ll_backward")) ||
				(!cudaPrintError(cudaMalloc(&d_validity_of_batch_ids, batch_size * sizeof(unsigned int)), "Create buffer d_validity_of_batch_ids")) ||
				(!cudaPrintError(cudaMalloc(&d_alphas, part_sizes * sizeof(T)), "Create buffer d_alphas")) ||
				(!cudaPrintError(cudaMalloc(&d_betas, part_sizes * sizeof(T)), "Create buffer d_betas")) ||
				(!cudaPrintError(cudaMalloc(&d_full_probs, full_sizes * sizeof(T)), "Create buffer d_full_probs")) ||
				(!cudaPrintError(cudaMalloc(&d_probs, part_sizes * sizeof(T)), "Create buffer d_probs")))
			{
				std::cerr << "Error: Unable to create buffers." << std::endl;
				return false;
			}
			use_native = true;
		}
		else
		{
			if ((!cudaPrintError(cudaMalloc(&d_grads, full_sizes * sizeof(T)), "Create buffer d_grads")) ||
				(!cudaPrintError(cudaMalloc(&d_loss, batch_size * sizeof(T)), "Create buffer d_loss")) ||
				(!cudaPrintError(cudaMalloc(&d_ll_backward, batch_size * sizeof(T)), "Create buffer d_ll_backward")) ||
				(!cudaPrintError(cudaMalloc(&d_validity_of_batch_ids, batch_size * sizeof(unsigned int)), "Create buffer d_validity_of_batch_ids")) ||
				(!cudaPrintError(cudaMalloc(&d_forward, connection_sizes * sizeof(T)), "Create buffer d_forward")) ||
				(!cudaPrintError(cudaMalloc(&d_forward_start, label_sizes * sizeof(T)), "Create buffer d_forward_start")) ||
				(!cudaPrintError(cudaMalloc(&d_forward_end, label_sizes * sizeof(T)), "Create buffer d_forward_end")) ||
				(!cudaPrintError(cudaMalloc(&d_backward, connection_sizes * sizeof(T)), "Create buffer d_backward")) ||
				(!cudaPrintError(cudaMalloc(&d_backward_start, label_sizes * sizeof(T)), "Create buffer d_backward_start")) ||
				(!cudaPrintError(cudaMalloc(&d_backward_end, label_sizes * sizeof(T)), "Create buffer d_backward_end")) ||
				(!cudaPrintError(cudaMalloc(&d_logits, full_sizes * sizeof(T)), "Create buffer d_logits")) ||
				(!cudaPrintError(cudaMalloc(&d_labels, label_sizes * sizeof(int)), "Create buffer d_labels")) ||
				(!cudaPrintError(cudaMalloc(&d_alphas, part_sizes * sizeof(T)), "Create buffer d_alphas")) ||
				(!cudaPrintError(cudaMalloc(&d_betas, part_sizes * sizeof(T)), "Create buffer d_betas")) ||
				(!cudaPrintError(cudaMalloc(&d_full_probs, full_sizes * sizeof(T)), "Create buffer d_full_probs")) ||
				(!cudaPrintError(cudaMalloc(&d_probs, part_sizes * sizeof(T)), "Create buffer d_probs")))
			{
				std::cerr << "Error: Unable to create buffers." << std::endl;
				return false;
			}
		}
	}
	return true;
}

template <typename T>
void CTCCuda<T>::setBuffers(T* grads, T* loss, const T* forward, const T* forward_start, const T* forward_end,
	const T* backward, const T* backward_start, const T* backward_end,
	const T* logits, const int* labels)
{
	if (use_native)
	{
		d_act_grads = grads;
		d_act_loss = loss;
		d_act_forward = forward;
		d_act_forward_start = forward_start;
		d_act_forward_end = forward_end;
		d_act_backward = backward;
		d_act_backward_start = backward_start;
		d_act_backward_end = backward_end;
		d_act_logits = logits;
		d_act_labels = labels;
	}
	else
	{
		d_act_grads = d_grads;
		d_act_loss = d_loss;
		d_act_forward = d_forward;
		d_act_forward_start = d_forward_start;
		d_act_forward_end = d_forward_end;
		d_act_backward = d_backward;
		d_act_backward_start = d_backward_start;
		d_act_backward_end = d_backward_end;
		d_act_logits = d_logits;
		d_act_labels = d_labels;
	}
}

template <typename T>
bool CTCCuda<T>::copyToDevice(const T* forward, const T* forward_start, const T* forward_end,
	const T* backward, const T* backward_start, const T* backward_end,
	const T* logits, const int* labels, cudaStream_t& stream)
{
	unsigned int full_sizes = this->time_size * this->logit_size * this->batch_size;
	unsigned int label_sizes = this->vector_size * this->batch_size;
	unsigned int connection_sizes = this->vector_size * this->vector_size * this->batch_size;

	if((!cudaPrintError(cudaMemcpyAsync(d_forward, forward, connection_sizes * sizeof(T), cudaMemcpyHostToDevice, stream), "Copy buffer d_forward")) ||
	   (!cudaPrintError(cudaMemcpyAsync(d_forward_start, forward_start, label_sizes * sizeof(T), cudaMemcpyHostToDevice, stream), "Copy buffer d_forward_start")) ||
	   (!cudaPrintError(cudaMemcpyAsync(d_forward_end, forward_end, label_sizes * sizeof(T), cudaMemcpyHostToDevice, stream), "Copy buffer d_forward_end")) ||
	   (!cudaPrintError(cudaMemcpyAsync(d_backward, backward, connection_sizes * sizeof(T), cudaMemcpyHostToDevice, stream), "Copy buffer d_backward")) ||
	   (!cudaPrintError(cudaMemcpyAsync(d_backward_start, backward_start, label_sizes * sizeof(T), cudaMemcpyHostToDevice, stream), "Copy buffer d_backward_start")) ||
	   (!cudaPrintError(cudaMemcpyAsync(d_backward_end, backward_end, label_sizes * sizeof(T), cudaMemcpyHostToDevice, stream), "Copy buffer d_backward_end")) ||
	   (!cudaPrintError(cudaMemcpyAsync(d_logits, logits, full_sizes * sizeof(T), cudaMemcpyHostToDevice, stream), "Copy buffer d_logits")) ||
	   (!cudaPrintError(cudaMemcpyAsync(d_labels, labels, label_sizes * sizeof(int), cudaMemcpyHostToDevice, stream), "Copy buffer d_labels")))
	{
		std::cerr << "Error: CTCCuda<T>::copyToDevice: while copy data to gpu." << std::endl;
		return false;
	}
	return true;
}


template <typename T>
bool CTCCuda<T>::copyFromDeviceDebugCore(T* forward, T* forward_start, T* forward_end,
	T* backward, T* backward_start, T* backward_end,
	T* logits, int* labels,
	T* grads, T* loss, T *alphas, T* betas, T* probs, T* full_probs, T* ll_backward)
{
	size_t part_sizes = this->time_size * this->vector_size * this->batch_size;
	size_t label_sizes = this->vector_size * this->batch_size;
	size_t connection_sizes = this->vector_size * this->vector_size * this->batch_size;
	size_t full_sizes = this->time_size * this->logit_size * this->batch_size;
	cudaMemcpyKind copy_type = cudaMemcpyDeviceToHost;
	if ((!cudaPrintError(cudaMemcpyAsync(forward, d_act_forward, connection_sizes * sizeof(T), copy_type, stream), "Copy buffer d_act_forward")) ||
		(!cudaPrintError(cudaMemcpyAsync(forward_start, d_act_forward_start, label_sizes * sizeof(T), copy_type, stream), "Copy buffer d_act_forward_start")) ||
		(!cudaPrintError(cudaMemcpyAsync(forward_end, d_act_forward_end, label_sizes * sizeof(T), copy_type, stream), "Copy buffer d_act_forward_end")) ||
		(!cudaPrintError(cudaMemcpyAsync(backward, d_act_backward, connection_sizes * sizeof(T), copy_type, stream), "Copy buffer d_act_backward")) ||
		(!cudaPrintError(cudaMemcpyAsync(backward_start, d_act_backward_start, label_sizes * sizeof(T), copy_type, stream), "Copy buffer d_act_backward_start")) ||
		(!cudaPrintError(cudaMemcpyAsync(backward_end, d_act_backward_end, label_sizes * sizeof(T), copy_type, stream), "Copy buffer d_act_backward_end")) ||
		(!cudaPrintError(cudaMemcpyAsync(logits, d_act_logits, full_sizes * sizeof(T), copy_type, stream), "Copy buffer d_act_logits")) ||
		(!cudaPrintError(cudaMemcpyAsync(labels, d_act_labels, label_sizes * sizeof(int), copy_type, stream), "Copy buffer d_act_labels")) ||
		(!cudaPrintError(cudaMemcpyAsync(alphas, d_alphas, part_sizes * sizeof(T), copy_type, stream), "Copy buffer d_alphas")) ||
		(!cudaPrintError(cudaMemcpyAsync(betas, d_betas, part_sizes * sizeof(T), copy_type, stream), "Copy buffer d_betas")) ||
		(!cudaPrintError(cudaMemcpyAsync(probs, d_probs, part_sizes * sizeof(T), copy_type, stream), "Copy buffer d_probs")) ||
		(!cudaPrintError(cudaMemcpyAsync(full_probs, d_full_probs, full_sizes * sizeof(T), copy_type, stream), "Copy buffer d_full_probs")) ||
		(!cudaPrintError(cudaMemcpyAsync(ll_backward, d_ll_backward, this->batch_size * sizeof(T), copy_type, stream), "Copy buffer d_ll_backward")) ||
		(!cudaPrintError(cudaMemcpyAsync(grads, d_act_grads, full_sizes * sizeof(T), copy_type, stream), "Copy buffer d_grads_act")) ||
		(!cudaPrintError(cudaMemcpyAsync(loss, d_act_loss, this->batch_size * sizeof(T), copy_type, stream), "Copy buffer d_loss_act")))
	{
		std::cerr << "Error: CTCCuda<T>::copyFromDeviceDebug: while copy data from gpu." << std::endl;
		return false;
	}
	return cudaPrintError(cudaStreamSynchronize(stream), "Error: CTCCuda<T>::copyFromDeviceDebug sync.");
}

template <typename T>
bool CTCCuda<T>::copyFromDevice(T* grads, T* loss, cudaStream_t& stream)
{
	unsigned int full_sizes = this->time_size * this->logit_size * this->batch_size;
	if ((!cudaPrintError(cudaMemcpyAsync(grads, d_grads, full_sizes * sizeof(T), cudaMemcpyDeviceToHost, stream), "Copy buffer d_grads")) ||
		(!cudaPrintError(cudaMemcpyAsync(loss, d_loss, this->batch_size * sizeof(T), cudaMemcpyDeviceToHost, stream), "Copy buffer d_loss")))
	{
		std::cerr << "Error: CTCCuda<T>::copyFromDevice: while copy data from gpu." << std::endl;
		return false;
	}
	return true;
}

template <typename T>
bool CTCCuda<T>::initCuda()
{
	bool result = (cudaPrintError(cudaStreamCreate(&stream), "Error while creating stream.")); 
#ifdef USE_PROFILLING
	result = result && (cudaPrintError(cudaEventCreate(&ev_copy_from_start), "Error while creating event.")) &&
		(cudaPrintError(cudaEventCreate(&ev_copy_from_stop), "Error while creating event.")) &&
		(cudaPrintError(cudaEventCreate(&ev_copy_to_start), "Error while creating event.")) &&
		(cudaPrintError(cudaEventCreate(&ev_copy_to_stop), "Error while creating event.")) &&
		(cudaPrintError(cudaEventCreate(&ev_init_start), "Error while creating event.")) &&
		(cudaPrintError(cudaEventCreate(&ev_init_stop), "Error while creating event.")) &&
		(cudaPrintError(cudaEventCreate(&ev_forward_backward_start), "Error while creating event.")) &&
		(cudaPrintError(cudaEventCreate(&ev_forward_backward_stop), "Error while creating event.")) &&
		(cudaPrintError(cudaEventCreate(&ev_forward_start), "Error while creating event.")) &&
		(cudaPrintError(cudaEventCreate(&ev_forward_stop), "Error while creating event.")) &&
		(cudaPrintError(cudaEventCreate(&ev_backward_start), "Error while creating event.")) &&
		(cudaPrintError(cudaEventCreate(&ev_backward_stop), "Error while creating event.")) &&
		(cudaPrintError(cudaEventCreate(&ev_finalize_start), "Error while creating event.")) &&
		(cudaPrintError(cudaEventCreate(&ev_finalize_stop), "Error while creating event.")) &&
		(cudaPrintError(cudaEventCreate(&ev_set_zero_to_invalid_start), "Error while creating event.")) &&
		(cudaPrintError(cudaEventCreate(&ev_set_zero_to_invalid_stop), "Error while creating event."));
#endif
	return result;
}

#endif
