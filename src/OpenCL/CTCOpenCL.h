#ifndef CTC_OPENCL_H
#define CTC_OPENCL_H

#define _USE_MATH_DEFINES
#include <math.h>
#include <iostream>
#include <array>

#include <CL/cl.hpp>

#include "OclHelper.h"

#include "../CTC.h"

template <typename T>
class CTCOpenCL : public CTC<T> {

public:
	CTCOpenCL(bool split_forward_backward = false, bool device_from_stdin = false);
	~CTCOpenCL();

	void printDeviceInfo();

protected:
	bool createBuffers(unsigned int time_size, unsigned int vector_size, unsigned int logit_size, unsigned int batch_size, unsigned int norm_step);
	bool copyToDevice(const T* forward, const T* forward_start, const T* forward_end,
		const T* backward, const T* backward_start, const T* backward_end,
		const T* logits, const int* labels);
	bool copyFromDevice(T* grads, T* loss);
	bool copyFromDeviceDebugCore(T* forward, T* forward_start, T* forward_end,
		T* backward, T* backward_start, T* backward_end,
		T* logits, int* labels,
		T* grads, T* loss, T* alphas, T* betas, T* probs, T* full_probs, T* ll_backward);
	bool calcCTCCore(T *grads, T *loss,
		const T *forward, const T *forward_start, const T *forward_end,
		const T *backward, const T *backward_start, const T *backward_end,
		const T *logits, const int *labels,
		unsigned int time_size, unsigned int vector_size, unsigned int logit_size, unsigned int batch_size, unsigned int norm_step, bool zero_infinity = false, bool native = false);

	bool initCL(bool device_from_stdin);
	void initBuffers();

	cl::CommandQueue queue;
	cl::Context context;
	cl::Program ctc_main_program;
	cl::Program ctc_helper_program;
	cl::Kernel init_kernel;
	cl::Kernel forward_kernel;
	cl::Kernel backward_kernel;
	cl::Kernel forward_backward_kernel;
	cl::Kernel finalize_kernel;
	cl::Kernel set_zero_to_invalid_kernel;
	cl::Device device;

	GpuImageData<T> *d_grads;
	GpuImageData<T> *d_loss;
	GpuImageData<T> *d_ll_backward;
	GpuImageData<T> *d_forward;
	GpuImageData<T> *d_forward_start;
	GpuImageData<T> *d_forward_end;
	GpuImageData<T> *d_backward;
	GpuImageData<T> *d_backward_start;
	GpuImageData<T> *d_backward_end;
	GpuImageData<T> *d_logits;
	GpuImageData<unsigned int>* d_validity_of_batch_ids;
	GpuImageData<int> *d_labels;
	GpuImageData<T> *d_alphas;
	GpuImageData<T> *d_betas;
	GpuImageData<T> *d_full_probs;
	GpuImageData<T> *d_probs;

	bool split_forward_backward;

#ifdef USE_PROFILLING
	cl::Event ev_init, ev_forward_backward, ev_finalize, ev_forward, ev_backward, ev_set_zero_to_invalid;
	cl::Event ev_copy_grads, ev_copy_loss, ev_copy_forward, ev_copy_forward_start, ev_copy_forward_end, ev_copy_backward, ev_copy_backward_start, ev_copy_backward_end, ev_copy_logits, ev_copy_labels;
#endif
} ;

template <typename T>
CTCOpenCL<T>::CTCOpenCL(bool split_forward_backward, bool device_from_stdin) : CTC<T>(), split_forward_backward(split_forward_backward)
{
	initBuffers();
	this->valid = initCL(device_from_stdin);
 }

template <typename T>
CTCOpenCL<T>::~CTCOpenCL()
{
	delete d_grads;
	delete d_loss;
	delete d_ll_backward;
	delete d_forward;
	delete d_forward_start;
	delete d_forward_end;
	delete d_backward;
	delete d_backward_start;
	delete d_backward_end;
	delete d_logits;
	delete d_labels;
	delete d_alphas;
	delete d_betas;
	delete d_full_probs;
	delete d_probs;
	delete d_validity_of_batch_ids;
}

template <typename T>
void CTCOpenCL<T>::initBuffers()
{
	d_grads = nullptr;
	d_loss = nullptr;
	d_ll_backward = nullptr;
	d_forward = nullptr;
	d_forward_start = nullptr;
	d_forward_end = nullptr;
	d_backward = nullptr;
	d_backward_start = nullptr;
	d_backward_end = nullptr;
	d_logits = nullptr;
	d_labels = nullptr;
	d_alphas = nullptr;
	d_betas = nullptr;
	d_full_probs = nullptr;
	d_probs = nullptr;
	d_validity_of_batch_ids = nullptr;
}

template <typename T>
void CTCOpenCL<T>::printDeviceInfo()
{
	if (!this->isValid())
	{
		std::cerr << "Error: CTC is not in valid state. Cannot print device info." << std::endl;
		return;
	}
	oclPrintDeviceInfo(device, std::cerr);
}

template <typename T>
bool CTCOpenCL<T>::createBuffers(unsigned int time_size, unsigned int vector_size, unsigned int logit_size, unsigned int batch_size, unsigned int norm_step)
{
	this->time_size = time_size;
	this->vector_size = vector_size;
	this->logit_size = logit_size;
	this->batch_size = batch_size;

	unsigned int part_sizes = time_size * vector_size * batch_size;
	unsigned int full_sizes = time_size * logit_size * batch_size;
	unsigned int label_sizes = vector_size * batch_size;
	unsigned int connection_sizes = vector_size * vector_size * batch_size;
	if (((d_grads == NULL) && ((d_grads = getGpuImageData<typename std::remove_pointer<decltype(d_grads)>::type::TemplateT>(BUF_TYPE_LINEAR, full_sizes, 1, context, CL_MEM_READ_WRITE)) == NULL)) ||
		((d_loss == NULL) && ((d_loss = getGpuImageData<typename std::remove_pointer<decltype(d_loss)>::type::TemplateT>(BUF_TYPE_LINEAR, batch_size, 1, context, CL_MEM_READ_WRITE)) == NULL)) ||
		((d_ll_backward == NULL) && ((d_ll_backward = getGpuImageData<typename std::remove_pointer<decltype(d_ll_backward)>::type::TemplateT>(BUF_TYPE_LINEAR, batch_size, 1, context, CL_MEM_READ_WRITE)) == NULL)) ||
		((d_validity_of_batch_ids == NULL) && ((d_validity_of_batch_ids = getGpuImageData<typename std::remove_pointer<decltype(d_validity_of_batch_ids)>::type::TemplateT>(BUF_TYPE_LINEAR, batch_size, 1, context, CL_MEM_READ_WRITE)) == NULL)) ||
		((d_forward == NULL) && ((d_forward = getGpuImageData<typename std::remove_pointer<decltype(d_forward)>::type::TemplateT>(BUF_TYPE_LINEAR, connection_sizes, 1, context, CL_MEM_READ_WRITE)) == NULL)) ||
		((d_forward_start == NULL) && ((d_forward_start = getGpuImageData<typename std::remove_pointer<decltype(d_forward_start)>::type::TemplateT>(BUF_TYPE_LINEAR, label_sizes, 1, context, CL_MEM_READ_WRITE)) == NULL)) ||
		((d_forward_end == NULL) && ((d_forward_end = getGpuImageData<typename std::remove_pointer<decltype(d_forward_end)>::type::TemplateT>(BUF_TYPE_LINEAR, label_sizes, 1, context, CL_MEM_READ_WRITE)) == NULL)) ||
		((d_backward == NULL) && ((d_backward = getGpuImageData<typename std::remove_pointer<decltype(d_backward)>::type::TemplateT>(BUF_TYPE_LINEAR, connection_sizes, 1, context, CL_MEM_READ_WRITE)) == NULL)) ||
		((d_backward_start == NULL) && ((d_backward_start = getGpuImageData<typename std::remove_pointer<decltype(d_backward_start)>::type::TemplateT>(BUF_TYPE_LINEAR, label_sizes, 1, context, CL_MEM_READ_WRITE)) == NULL)) ||
		((d_backward_end == NULL) && ((d_backward_end = getGpuImageData<typename std::remove_pointer<decltype(d_backward_end)>::type::TemplateT>(BUF_TYPE_LINEAR, label_sizes, 1, context, CL_MEM_READ_WRITE)) == NULL)) ||
		((d_logits == NULL) && ((d_logits = getGpuImageData<typename std::remove_pointer<decltype(d_logits)>::type::TemplateT>(BUF_TYPE_LINEAR, full_sizes, 1, context, CL_MEM_READ_WRITE)) == NULL)) ||
		((d_labels == NULL) && ((d_labels = getGpuImageData<typename std::remove_pointer<decltype(d_labels)>::type::TemplateT>(BUF_TYPE_LINEAR, label_sizes, 1, context, CL_MEM_READ_WRITE)) == NULL)) ||
		((d_alphas == NULL) && ((d_alphas = getGpuImageData<typename std::remove_pointer<decltype(d_alphas)>::type::TemplateT>(BUF_TYPE_LINEAR, part_sizes, 1, context, CL_MEM_READ_WRITE)) == NULL)) ||
		((d_betas == NULL) && ((d_betas = getGpuImageData<typename std::remove_pointer<decltype(d_betas)>::type::TemplateT>(BUF_TYPE_LINEAR, part_sizes, 1, context, CL_MEM_READ_WRITE)) == NULL)) ||
		((d_full_probs == NULL) && ((d_full_probs = getGpuImageData<typename std::remove_pointer<decltype(d_full_probs)>::type::TemplateT>(BUF_TYPE_LINEAR, full_sizes, 1, context, CL_MEM_READ_WRITE)) == NULL)) ||
		((d_probs == NULL) && ((d_probs = getGpuImageData<typename std::remove_pointer<decltype(d_probs)>::type::TemplateT>(BUF_TYPE_LINEAR, part_sizes, 1, context, CL_MEM_READ_WRITE)) == NULL)))
	{
		std::cerr << "Error: Unable to create buffers." << std::endl;
		return false;
	}

	// resize buffers to points_count with reserve aligned to group size
	d_grads->resize(full_sizes, 1, queue);
	d_loss->resize(batch_size, 1, queue);
	d_ll_backward->resize(batch_size, 1, queue);
	d_validity_of_batch_ids->resize(batch_size, 1, queue);
	d_forward->resize(connection_sizes, 1, queue);
	d_forward_start->resize(label_sizes, 1, queue);
	d_forward_end->resize(label_sizes, 1, queue);
	d_backward->resize(connection_sizes, 1, queue);
	d_backward_start->resize(label_sizes, 1, queue);
	d_backward_end->resize(label_sizes, 1, queue);
	d_logits->resize(full_sizes, 1, queue);
	d_labels->resize(label_sizes, 1, queue);
	d_alphas->resize(part_sizes, 1, queue);
	d_betas->resize(part_sizes, 1, queue);
	d_full_probs->resize(full_sizes, 1, queue);
	d_probs->resize(part_sizes, 1, queue);
	cl::LocalSpaceArg l_vector_arg_size = cl::Local(vector_size * sizeof(T) * 2);
	cl::LocalSpaceArg l_logit_arg_size = cl::Local(logit_size * sizeof(T));

	if (!oclPrintError(init_kernel.setArg(0, *d_full_probs->data), "Error while setting init full_probs kernel arg.") ||
		!oclPrintError(init_kernel.setArg(1, *d_probs->data), "Error while setting init probs kernel arg.") ||
		!oclPrintError(init_kernel.setArg(2, *d_labels->data), "Error while setting init labels kernel arg.") ||
		!oclPrintError(init_kernel.setArg(3, *d_logits->data), "Error while setting init logits kernel arg.") ||
		!oclPrintError(init_kernel.setArg(4, vector_size), "Error while setting init vector_size kernel arg.") ||
		!oclPrintError(init_kernel.setArg(5, logit_size), "Error while setting init logit_size kernel arg.") ||
		!oclPrintError(init_kernel.setArg(6, l_logit_arg_size), "Error while setting init local_memory kernel arg.")) return false;

	if (!oclPrintError(finalize_kernel.setArg(0, *d_grads->data), "Error while setting finalize grads kernel arg.") ||
		!oclPrintError(finalize_kernel.setArg(2, *d_alphas->data), "Error while setting finalize alphas kernel arg.") ||
		!oclPrintError(finalize_kernel.setArg(3, *d_betas->data), "Error while setting finalize betas kernel arg.") ||
		!oclPrintError(finalize_kernel.setArg(4, *d_probs->data), "Error while setting finalize probs kernel arg.") ||
		!oclPrintError(finalize_kernel.setArg(5, *d_full_probs->data), "Error while setting finalize full_probs kernel arg.") ||
		!oclPrintError(finalize_kernel.setArg(6, *d_labels->data), "Error while setting finalize d_labels kernel arg.") ||
		!oclPrintError(finalize_kernel.setArg(7, vector_size), "Error while setting finalize vector_size kernel arg.") ||
		!oclPrintError(finalize_kernel.setArg(8, logit_size), "Error while setting finalize logit_size kernel arg.") ||
		!oclPrintError(finalize_kernel.setArg(9, l_logit_arg_size), "Error while setting finalize local_memory kernel arg.")) return false;

	if (!oclPrintError(set_zero_to_invalid_kernel.setArg(0, *d_grads->data), "Error while setting zero grads kernel arg.") ||
		!oclPrintError(set_zero_to_invalid_kernel.setArg(1, *d_validity_of_batch_ids->data), "Error while setting zero validity kernel arg.") ||
		!oclPrintError(set_zero_to_invalid_kernel.setArg(2, logit_size), "Error while setting zero logit_size kernel arg.") ||
		!oclPrintError(set_zero_to_invalid_kernel.setArg(3, time_size), "Error while setting zero time_size kernel arg.")) return false;

	if (split_forward_backward)
	{
		if (!oclPrintError(forward_kernel.setArg(0, *d_loss->data), "Error while setting forward loss kernel arg.") ||
			!oclPrintError(forward_kernel.setArg(2, *d_alphas->data), "Error while setting forward alphas kernel arg.") ||
			!oclPrintError(forward_kernel.setArg(3, *d_probs->data), "Error while setting forward probs kernel arg.") ||
			!oclPrintError(forward_kernel.setArg(4, *d_forward->data), "Error while setting forward forward kernel arg.") ||
			!oclPrintError(forward_kernel.setArg(5, *d_forward_start->data), "Error while setting forward forward_start kernel arg.") ||
			!oclPrintError(forward_kernel.setArg(6, *d_forward_end->data), "Error while setting forward forward_end kernel arg.") ||
			!oclPrintError(forward_kernel.setArg(7, time_size), "Error while setting forward time_size kernel arg.") ||
			!oclPrintError(forward_kernel.setArg(8, vector_size), "Error while setting forward vector_size kernel arg.") ||
			!oclPrintError(forward_kernel.setArg(9, norm_step), "Error while setting forward norm_step kernel arg.") ||
			!oclPrintError(forward_kernel.setArg(10, l_vector_arg_size), "Error while setting forward local_memory kernel arg.")) return false;

		if (!oclPrintError(backward_kernel.setArg(0, *d_ll_backward->data), "Error while settingi backward ll_backward kernel arg.") ||
			!oclPrintError(backward_kernel.setArg(1, *d_betas->data), "Error while setting backward betas kernel arg.") ||
			!oclPrintError(backward_kernel.setArg(2, *d_probs->data), "Error while setting backward probs kernel arg.") ||
			!oclPrintError(backward_kernel.setArg(3, *d_backward->data), "Error while setting backward backward kernel arg.") ||
			!oclPrintError(backward_kernel.setArg(4, *d_backward_start->data), "Error while setting backward backward_start kernel arg.") ||
			!oclPrintError(backward_kernel.setArg(5, *d_backward_end->data), "Error while setting backward backward_end kernel arg.") ||
			!oclPrintError(backward_kernel.setArg(6, time_size), "Error while setting backward time_size kernel arg.") ||
			!oclPrintError(backward_kernel.setArg(7, vector_size), "Error while setting backward vector_size kernel arg.") ||
			!oclPrintError(backward_kernel.setArg(8, norm_step), "Error while setting backward norm_step kernel arg.") ||
			!oclPrintError(backward_kernel.setArg(9, l_vector_arg_size), "Error while setting backward local_memory kernel arg.")) return false;
	} 
	else
	{
		if (!oclPrintError(forward_backward_kernel.setArg(0, *d_loss->data), "Error while setting forward_backward loss kernel arg.") ||
			!oclPrintError(forward_backward_kernel.setArg(1, *d_ll_backward->data), "Error while setting forward_backward ll_backward kernel arg.") ||
			!oclPrintError(forward_backward_kernel.setArg(3, *d_alphas->data), "Error while setting forward_backward alphas kernel arg.") ||
			!oclPrintError(forward_backward_kernel.setArg(4, *d_betas->data), "Error while setting forward_backward betas kernel arg.") ||
			!oclPrintError(forward_backward_kernel.setArg(5, *d_probs->data), "Error while setting forward_backward probs kernel arg.") ||
			!oclPrintError(forward_backward_kernel.setArg(6, *d_forward->data), "Error while setting forward_backward forward kernel arg.") ||
			!oclPrintError(forward_backward_kernel.setArg(7, *d_forward_start->data), "Error while setting forward_backward forward_start kernel arg.") ||
			!oclPrintError(forward_backward_kernel.setArg(8, *d_forward_end->data), "Error while setting forward_backward forward end kernel arg.") ||
			!oclPrintError(forward_backward_kernel.setArg(9, *d_backward->data), "Error while setting forward_backward backward kernel arg.") ||
			!oclPrintError(forward_backward_kernel.setArg(10, *d_backward_start->data), "Error while setting forward_backward backward_start kernel arg.") ||
			!oclPrintError(forward_backward_kernel.setArg(11, *d_backward_end->data), "Error while setting forward_backward backward_end kernel arg.") ||
			!oclPrintError(forward_backward_kernel.setArg(12, time_size), "Error while setting forward_backward time_size kernel arg.") ||
			!oclPrintError(forward_backward_kernel.setArg(13, vector_size), "Error while setting forward_backward vector_size kernel arg.") ||
			!oclPrintError(forward_backward_kernel.setArg(14, norm_step), "Error while setting forward_backward norm_step kernel arg.") ||
			!oclPrintError(forward_backward_kernel.setArg(15, l_vector_arg_size), "Error while setting forward_backward local_memory kernel arg.")) return false;
	} 

	return true;
}

template <typename T>
bool CTCOpenCL<T>::copyToDevice(const T* forward, const T* forward_start, const T* forward_end,
	const T* backward, const T* backward_start, const T* backward_end,
	const T* logits, const int* labels)
{
	cl::Event* ev_copy_forward = NULL;
	cl::Event* ev_copy_forward_start = NULL;
	cl::Event* ev_copy_forward_end = NULL;
	cl::Event* ev_copy_backward = NULL;
	cl::Event* ev_copy_backward_start = NULL;
	cl::Event* ev_copy_backward_end = NULL;
	cl::Event* ev_copy_logits = NULL;
	cl::Event* ev_copy_labels = NULL;
#ifdef USE_PROFILLING
	ev_copy_forward = &this->ev_copy_forward;
	ev_copy_forward_start = &this->ev_copy_forward_start;
	ev_copy_forward_end = &this->ev_copy_forward_end;
	ev_copy_backward = &this->ev_copy_backward;
	ev_copy_backward_start = &this->ev_copy_backward_start;
	ev_copy_backward_end = &this->ev_copy_backward_end;
	ev_copy_logits = &this->ev_copy_logits;
	ev_copy_labels = &this->ev_copy_labels;
#endif

	bool result = d_forward->copyFrom(forward, queue, ev_copy_forward)
		&& d_forward_start->copyFrom(forward_start, queue, ev_copy_forward_start)
		&& d_forward_end->copyFrom(forward_end, queue, ev_copy_forward_end)
		&& d_backward->copyFrom(backward, queue, ev_copy_backward)
		&& d_backward_start->copyFrom(backward_start, queue, ev_copy_backward_start)
		&& d_backward_end->copyFrom(backward_end, queue, ev_copy_backward_end)
		&& d_logits->copyFrom(logits, queue, ev_copy_logits)
		&& d_labels->copyFrom(labels, queue, ev_copy_labels);
	if (!result)
	{
		std::cerr << "Error: CTCOpenCL<T>::copyToDevice: while copy data to gpu." << std::endl;
	}
	return result;
}


template <typename T>
bool CTCOpenCL<T>::copyFromDeviceDebugCore(T* forward, T* forward_start, T* forward_end,
	T* backward, T* backward_start, T* backward_end,
	T* logits, int* labels,
	T* grads, T* loss, T *alphas, T* betas, T* probs, T* full_probs, T* ll_backward)
{
	bool result = d_forward->copyTo(forward, queue)
				&& d_forward_start->copyTo(forward_start, queue)
				&& d_forward_end->copyTo(forward_end, queue)
				&& d_backward->copyTo(backward, queue)
				&& d_backward_start->copyTo(backward_start, queue)
				&& d_backward_end->copyTo(backward_end, queue)
				&& d_logits->copyTo(logits, queue)
				&& d_labels->copyTo(labels, queue)
				&& d_grads->copyTo(grads, queue)
				&& d_loss->copyTo(loss, queue)
				&& d_alphas->copyTo(alphas, queue)
				&& d_betas->copyTo(betas, queue)
				&& d_probs->copyTo(probs, queue)
				&& d_full_probs->copyTo(full_probs, queue)
				&& d_ll_backward->copyTo(ll_backward, queue);
	result = result && oclPrintError(queue.finish(), "Error: CTCOpenCL<T>::copyFromDeviceDebug sync.");
	return result;
}

template <typename T>
bool CTCOpenCL<T>::copyFromDevice(T* grads, T* loss)
{
	cl::Event* ev_copy_grads = NULL;
	cl::Event* ev_copy_loss = NULL;
#ifdef USE_PROFILLING
	ev_copy_grads = &this->ev_copy_grads;
	ev_copy_loss = &this->ev_copy_loss;
#endif
	bool result = d_grads->copyTo(grads, queue, ev_copy_grads)
		&& d_loss->copyTo(loss, queue, ev_copy_loss);
	if (!result)
	{
		std::cerr << "Error: CTCOpenCL<T>::copyFromDevice: while copy data from gpu." << std::endl;
	}
	return result;
}

template <typename T>
bool CTCOpenCL<T>::calcCTCCore(T* grads, T* loss,
	const T* forward, const T* forward_start, const T* forward_end,
	const T* backward, const T* backward_start, const T* backward_end,
	const T* logits, const int* labels, unsigned int time_size, unsigned int vector_size, unsigned int logit_size, unsigned int batch_size, unsigned int norm_step, bool zero_infinity, bool native)
{ 
	if (!this->isValid())
	{
		std::cerr << "Error: CTC is not in valid state. Cannot process projToSphere." << std::endl;
		return false;
	}
	cl::Event* ev_init = NULL;
	cl::Event* ev_forward_backward = NULL;
	cl::Event* ev_finalize = NULL;
	cl::Event* ev_set_zero_to_invalid = NULL;
	cl::Event* ev_forward = NULL;
	cl::Event* ev_backward = NULL;
#ifdef USE_PROFILLING
	ev_init = &this->ev_init;
	ev_forward_backward = &this->ev_forward_backward;
	ev_finalize = &this->ev_finalize;
	ev_set_zero_to_invalid = &this->ev_set_zero_to_invalid;
	ev_forward = &this->ev_forward;
	ev_backward = &this->ev_backward;
#endif
	bool result = createBuffers(time_size, vector_size, logit_size, batch_size, norm_step) &&
		copyToDevice(forward, forward_start, forward_end, backward, backward_start, backward_end, logits, labels);

	cl::Memory ptr_validity_of_batch_ids = (zero_infinity) ? (*d_validity_of_batch_ids->data) : cl::Memory();

	result = result && oclPrintError(queue.enqueueNDRangeKernel(init_kernel, cl::NullRange, cl::NDRange(std::max(vector_size, logit_size), batch_size, time_size), cl::NDRange(std::max(vector_size, logit_size), 1, 1), NULL, ev_init), "Error while running init_kernel kernel.");
	if (split_forward_backward)
	{
		result = result && oclPrintError(forward_kernel.setArg(1, ptr_validity_of_batch_ids), "Error while setting forward validity kernel arg.") && oclPrintError(queue.enqueueNDRangeKernel(forward_kernel, cl::NullRange, cl::NDRange(vector_size, batch_size, 1), cl::NDRange(vector_size, 1, 1), NULL, ev_forward), "Error while running forward_kernel kernel.") &&
			oclPrintError(queue.enqueueNDRangeKernel(backward_kernel, cl::NullRange, cl::NDRange(vector_size, batch_size, 1), cl::NDRange(vector_size, 1, 1), NULL, ev_backward), "Error while running backward_kernel kernel.");
	 }
	else
	{ 
		result = result && oclPrintError(forward_backward_kernel.setArg(2, ptr_validity_of_batch_ids), "Error while setting forward_backward validity kernel arg.") && oclPrintError(queue.enqueueNDRangeKernel(forward_backward_kernel, cl::NullRange, cl::NDRange(vector_size, batch_size, 2), cl::NDRange(vector_size, 1, 1), NULL, ev_forward_backward), "Error while running forward_backward_kernel kernel.");
	}
	result = result && oclPrintError(finalize_kernel.setArg(1, ptr_validity_of_batch_ids), "Error while setting finalize validity kernel arg.") && oclPrintError(queue.enqueueNDRangeKernel(finalize_kernel, cl::NullRange, cl::NDRange(std::max(vector_size, logit_size), batch_size, time_size), cl::NDRange(std::max(vector_size, logit_size), 1, 1), NULL, ev_finalize), "Error while running finalize kernel.");
	if(zero_infinity) oclPrintError(queue.enqueueNDRangeKernel(set_zero_to_invalid_kernel, cl::NullRange, cl::NDRange(256, batch_size, ceilDivBy(logit_size * time_size, 256u)), cl::NDRange(256, 1, 1), NULL, ev_set_zero_to_invalid), "Error while running set_zero_to_invalid kernel.");

	result = result && copyFromDevice(grads, loss);
	result = result && oclPrintError(queue.finish(), "Error: sync.");

#ifdef USE_PROFILLING
	std::cerr << "Compute time: init:" << getEventTime(this->ev_init) << " core:" << ((split_forward_backward) ? (getEventTime(this->ev_forward) + getEventTime(this->ev_backward)) : getEventTime(this->ev_forward_backward)) << " finalize: " << getEventTime(this->ev_finalize) << " zero invalid:" << ((zero_infinity) ? getEventTime(this->ev_set_zero_to_invalid) : 0.0) << std::endl;
	std::cerr << "Copy time: host->device:" << (getEventTime(ev_copy_forward) + getEventTime(ev_copy_forward_start) + getEventTime(ev_copy_forward_end) + getEventTime(ev_copy_backward) + getEventTime(ev_copy_backward_start) + getEventTime(ev_copy_backward_end) + getEventTime(ev_copy_logits) + getEventTime(ev_copy_labels)) << " device->host:" << (getEventTime(ev_copy_grads) + getEventTime(ev_copy_loss)) << std::endl;
#endif

	return result;
}

template <typename T>
bool CTCOpenCL<T>::initCL(bool device_from_stdin)
{
	std::vector<cl::Device> devices;
	cl_int err_id;
	if (device_from_stdin)
	{
		devices = oclGetDevices();
		if (devices.size() == 0)
		{
			std::cerr << "Error: Unable to find any OpenCL device." << std::endl;
			return false;
		}
		device = oclSelectDeviceByStdin(devices);
	}
	else
	{
		devices = oclGetDevices(CL_DEVICE_TYPE_GPU);
		if (devices.size() == 0)
		{
			devices = oclGetDevices(CL_DEVICE_TYPE_CPU);
			if (devices.size() == 0)
			{
				std::cerr << "Error: Unable to find any OpenCL device." << std::endl;
				return false;
			}
			std::cerr << "Info: Unable to find any GPU OpenCL device. Use CPU device instead." << std::endl;
		}
	}
	this->device = devices[0];
	oclPrintDeviceInfo(device, std::cerr, true);
	context = cl::Context(device, NULL, NULL, NULL, &err_id);
	if (!oclPrintError(err_id, "Error while creating cl::Context.")) return false;
	//queue = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err_id);
 
#if !defined(CL_VERSION_2_0) || defined(CL_USE_DEPRECATED_OPENCL_1_2_APIS)
	cl_command_queue_properties properties = 0;
#ifdef USE_PROFILLING
	properties = CL_QUEUE_PROFILING_ENABLE;
#endif
	queue = cl::CommandQueue(context, device, properties, &err_id);
#else
#ifdef USE_PROFILLING
	cl_queue_properties properties[3] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
#else
	cl_queue_properties* properties = NULL;
#endif
	queue = cl::CommandQueue(context, device, properties, &err_id);
#endif

	if (!oclPrintError(err_id, "Error while creating cl::CommandQueue.")) return false;
	//std::string defines(" -D BLOCK_SIZE=" + std::to_string(this->range.getFlatSize()));
	std::string defines(" -Ikernels/");
	if (std::is_same<T, double>()) defines += " -DUSE_DOUBLE";
	else if (!std::is_same<T, float>())
	{
		std::cerr << "Error: CTCOpenCL<T>::initCL: Unsupported datatype. Supported types are float and double" << std::endl;
		return false;
	}
	std::string program_main_filepath("kernels/ctc_main.cl");
	if (compileOpenclSource(ctc_main_program, context, device, defines, program_main_filepath) == OCL_COMPILE_FAILED)
	{
		std::cerr << "Error while loading OpenCL main program." << std::endl;
		return false;
	}
	std::string program_helper_filepath("kernels/ctc_helper.cl");
	if (compileOpenclSource(ctc_helper_program, context, device, defines, program_helper_filepath) == OCL_COMPILE_FAILED)
	{
		std::cerr << "Error while loading OpenCL helper program." << std::endl;
		return false;
	}

	init_kernel = cl::Kernel(ctc_helper_program, "ctc_init", &err_id);
	if (!oclPrintError(err_id, "Error while creating ctc_init kernel.")) return false;
	finalize_kernel = cl::Kernel(ctc_helper_program, "ctc_finalize", &err_id);
	if (!oclPrintError(err_id, "Error while creating ctc_finalize kernel.")) return false;
	set_zero_to_invalid_kernel = cl::Kernel(ctc_helper_program, "ctc_set_zero_to_invalid", &err_id);
	if (!oclPrintError(err_id, "Error while creating set_zero_to_invalid kernel.")) return false;
	if (split_forward_backward)
	{
		forward_kernel = cl::Kernel(ctc_main_program, "ctc_forward", &err_id);
		if (!oclPrintError(err_id, "Error while creating ctc_forward kernel.")) return false;
		backward_kernel = cl::Kernel(ctc_main_program, "ctc_backward", &err_id);
		if (!oclPrintError(err_id, "Error while creating ctc_backward kernel.")) return false;
	}
	else
	{
		forward_backward_kernel = cl::Kernel(ctc_main_program, "ctc_forward_backward", &err_id);
		if (!oclPrintError(err_id, "Error while creating ctc_forward_backward kernel.")) return false;
	}

	return true;
}

#endif
