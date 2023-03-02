#define _USE_MATH_DEFINES
#include <math.h>
#include <iostream>
#include <array>

#if defined(USE_TORCH)
#include <ATen/cuda/CUDAContext.h>
#endif

#include "CudaHelper.h"

#include "CTCCuda.h"
#include "ctc_helper.h"
#include "ctc_main.h"

template <typename T>
bool CTCCuda<T>::calcCTCCore(T* grads, T* loss,
	const T* forward, const T* forward_start, const T* forward_end,
	const T* backward, const T* backward_start, const T* backward_end,
	const T* logits, const int* labels, unsigned int time_size, unsigned int vector_size, unsigned int logit_size, unsigned int batch_size, unsigned int norm_step, bool zero_infinity, bool act_use_native)
{
	cudaStream_t* act_stream = &stream;
#if defined(USE_TORCH)
	cudaStream_t native_stream;
	if (act_use_native)
	{
		native_stream = at::cuda::getCurrentCUDAStream();
		act_stream = &native_stream;
	}
#endif
	dim3 local_helper(std::max(vector_size, logit_size), 1, 1);
	dim3 local_main(vector_size, 1, 1);
	dim3 group_main(1, batch_size, (split_forward_backward) ? 1 : 2);
	dim3 group_helper(1, batch_size, time_size);
	size_t shared_main = sizeof(T) * vector_size * 2 + ((compile_static_matrix && (vector_size == 256)) ? ((256 - REG_LIMIT) * vector_size * sizeof(T)) : 0);
	size_t shared_helper = sizeof(T) * std::max(vector_size, logit_size);

	if (!this->isValid())
	{
		std::cerr << "Error: CTC is not in valid state. Cannot process calcCTCCore." << std::endl;
		return false;
	}
	bool result = createBuffers(time_size, vector_size, logit_size, batch_size, norm_step, act_use_native);
	setBuffers(grads, loss, forward, forward_start, forward_end, backward, backward_start, backward_end, logits, labels);
#ifdef USE_PROFILLING
	cudaEventRecord(ev_copy_to_start, *act_stream);
#endif
	if(!act_use_native) result = result && copyToDevice(forward, forward_start, forward_end, backward, backward_start, backward_end, logits, labels, *act_stream);
#ifdef USE_PROFILLING
	cudaEventRecord(ev_copy_to_stop, *act_stream);
	cudaEventRecord(ev_init_start, *act_stream);
#endif
	ctc_init<T><<<group_helper, local_helper, shared_helper, *act_stream>>>(d_full_probs, d_probs, d_act_labels, d_act_logits, vector_size, logit_size);
#ifdef USE_PROFILLING
	cudaEventRecord(ev_init_stop, *act_stream);
#endif
	unsigned int *act_d_validity_of_batch_ids = (zero_infinity) ? d_validity_of_batch_ids : nullptr;
	if (split_forward_backward)
	{
#ifdef USE_PROFILLING
		cudaEventRecord(ev_forward_start, *act_stream);
#endif
		if (compile_static_matrix)
		{
			switch (vector_size)
			{
			case 256:
				ctc_forward<T, 256><<<group_main, local_main, shared_main, *act_stream>>>(d_act_loss, act_d_validity_of_batch_ids, d_alphas, d_probs, d_act_forward, d_act_forward_start, d_act_forward_end, time_size, vector_size, norm_step);
				break;
			case 128:
				ctc_forward<T, 128><<<group_main, local_main, shared_main, *act_stream>>>(d_act_loss, act_d_validity_of_batch_ids, d_alphas, d_probs, d_act_forward, d_act_forward_start, d_act_forward_end, time_size, vector_size, norm_step);
				break;
			case 64:
				ctc_forward<T, 64><<<group_main, local_main, shared_main, *act_stream>>>(d_act_loss, act_d_validity_of_batch_ids, d_alphas, d_probs, d_act_forward, d_act_forward_start, d_act_forward_end, time_size, vector_size, norm_step);
				break;
			default:
				ctc_forward<T, 1><<<group_main, local_main, shared_main, *act_stream>>>(d_act_loss, act_d_validity_of_batch_ids, d_alphas, d_probs, d_act_forward, d_act_forward_start, d_act_forward_end, time_size, vector_size, norm_step);
			}
		}
		else
		{
			ctc_forward<T, 1><<<group_main, local_main, shared_main, *act_stream>>>(d_act_loss, act_d_validity_of_batch_ids, d_alphas, d_probs, d_act_forward, d_act_forward_start, d_act_forward_end, time_size, vector_size, norm_step);
		}
#ifdef USE_PROFILLING
		cudaEventRecord(ev_forward_stop, *act_stream);
		cudaEventRecord(ev_backward_start, *act_stream);
#endif
		if (compile_static_matrix)
		{
			switch (vector_size)
			{
			case 256:
				ctc_backward<T, 256><<<group_main, local_main, shared_main, *act_stream>>>(d_ll_backward, d_betas, d_probs, d_act_backward, d_act_backward_start, d_act_backward_end, time_size, vector_size, norm_step);
				break;
			case 128:
				ctc_backward<T, 128><<<group_main, local_main, shared_main, *act_stream>>>(d_ll_backward, d_betas, d_probs, d_act_backward, d_act_backward_start, d_act_backward_end, time_size, vector_size, norm_step);
				break;
			case 64:
				ctc_backward<T, 64><<<group_main, local_main, shared_main, *act_stream>>>(d_ll_backward, d_betas, d_probs, d_act_backward, d_act_backward_start, d_act_backward_end, time_size, vector_size, norm_step);
				break;
			default:
				ctc_backward<T, 1><<<group_main, local_main, shared_main, *act_stream>>>(d_ll_backward, d_betas, d_probs, d_act_backward, d_act_backward_start, d_act_backward_end, time_size, vector_size, norm_step);
			}
		}
		else
		{
			ctc_backward<T, 1><<<group_main, local_main, shared_main, *act_stream>>>(d_ll_backward, d_betas, d_probs, d_act_backward, d_act_backward_start, d_act_backward_end, time_size, vector_size, norm_step);
		}
#ifdef USE_PROFILLING
		cudaEventRecord(ev_backward_stop, *act_stream);
#endif
	}
	else
	{
#ifdef USE_PROFILLING
		cudaEventRecord(ev_forward_backward_start, *act_stream);
#endif
		if (compile_static_matrix)
		{
			switch (vector_size)
			{
			case 256:
				ctc_forward_backward<T, 256><<<group_main, local_main, shared_main, *act_stream>>>(d_act_loss, d_ll_backward, act_d_validity_of_batch_ids, d_alphas, d_betas, d_probs, d_act_forward, d_act_forward_start, d_act_forward_end, d_act_backward, d_act_backward_start, d_act_backward_end, time_size, vector_size, norm_step);
				break;
			case 128:
				ctc_forward_backward<T, 128><<<group_main, local_main, shared_main, *act_stream>>>(d_act_loss, d_ll_backward, act_d_validity_of_batch_ids, d_alphas, d_betas, d_probs, d_act_forward, d_act_forward_start, d_act_forward_end, d_act_backward, d_act_backward_start, d_act_backward_end, time_size, vector_size, norm_step);
				break;
			case 64:
				ctc_forward_backward<T, 64><<<group_main, local_main, shared_main, *act_stream>>>(d_act_loss, d_ll_backward, act_d_validity_of_batch_ids, d_alphas, d_betas, d_probs, d_act_forward, d_act_forward_start, d_act_forward_end, d_act_backward, d_act_backward_start, d_act_backward_end, time_size, vector_size, norm_step);
				break;
			default:
				ctc_forward_backward<T, 1><<<group_main, local_main, shared_main, *act_stream>>>(d_act_loss, d_ll_backward, act_d_validity_of_batch_ids, d_alphas, d_betas, d_probs, d_act_forward, d_act_forward_start, d_act_forward_end, d_act_backward, d_act_backward_start, d_act_backward_end, time_size, vector_size, norm_step);
			}
		}
		else
		{
			ctc_forward_backward<T, 1><<<group_main, local_main, shared_main, *act_stream>>>(d_act_loss, d_ll_backward, act_d_validity_of_batch_ids, d_alphas, d_betas, d_probs, d_act_forward, d_act_forward_start, d_act_forward_end, d_act_backward, d_act_backward_start, d_act_backward_end, time_size, vector_size, norm_step);
		}
#ifdef USE_PROFILLING
		cudaEventRecord(ev_forward_backward_stop, *act_stream);
#endif
	}

#ifdef USE_PROFILLING
	cudaEventRecord(ev_finalize_start, *act_stream);
#endif
	ctc_finalize<T><<<group_helper, local_helper, shared_helper, *act_stream>>>(d_act_grads, act_d_validity_of_batch_ids, d_alphas, d_betas, d_probs, d_full_probs, d_act_labels, vector_size, logit_size);
#ifdef USE_PROFILLING
	cudaEventRecord(ev_finalize_stop, *act_stream);
	cudaEventRecord(ev_set_zero_to_invalid_start, *act_stream);
#endif
	if(zero_infinity) ctc_set_zero_to_invalid<T><<<group_helper, local_helper, 0, *act_stream>>>(d_act_grads, act_d_validity_of_batch_ids, logit_size, time_size);

#ifdef USE_PROFILLING
	cudaEventRecord(ev_set_zero_to_invalid_stop, *act_stream);
	cudaEventRecord(ev_copy_from_start, *act_stream);
#endif
	if(!act_use_native) result = result && copyFromDevice(grads, loss, *act_stream);
#ifdef USE_PROFILLING
	cudaEventRecord(ev_copy_from_stop, *act_stream);
#endif
	if((!act_use_native) || (this->sync_native)) result = result && cudaPrintError(cudaStreamSynchronize(*act_stream), "Error: sync.");
#ifdef USE_PROFILLING
	if ((!act_use_native) || (this->sync_native))
	{
		std::cerr << "Compute time: init:" << getCudaEventTime(ev_init_start, ev_init_stop) << " core:" << ((split_forward_backward) ? (getCudaEventTime(ev_forward_start, ev_forward_stop) + getCudaEventTime(ev_backward_start, ev_backward_stop)) : getCudaEventTime(ev_forward_backward_start, ev_forward_backward_stop)) << " finalize:" << getCudaEventTime(ev_finalize_start, ev_finalize_stop) << " zero invalid:" << getCudaEventTime(ev_set_zero_to_invalid_start, ev_set_zero_to_invalid_stop) << std::endl;
		std::cerr << "Copy time: host->device:" << getCudaEventTime(ev_copy_to_start, ev_copy_to_stop) << " device->host:" << getCudaEventTime(ev_copy_from_start, ev_copy_from_stop) << std::endl;
	}
#endif

	return result;
}

template class CTCCuda<float>;
template class CTCCuda<double>;