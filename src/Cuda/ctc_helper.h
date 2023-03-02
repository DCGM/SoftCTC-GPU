#ifndef CTC_HELPER
#define CTC_HELPER

#include "ctc_common.h"

#define MIN_FLOAT_TYPE 1e-37
// kernel for generation of full_probs and probs
// kernel dimension (X,Y,Z)=(max(vetror_size, logit_size),batch_size,time_size)
template <typename FLOAT_TYPE>
__global__ void ctc_init( FLOAT_TYPE* full_probs,  FLOAT_TYPE* probs,
     const int* labels,  const FLOAT_TYPE* logits,
    unsigned int vector_size, unsigned int logit_size)
{
    extern __shared__ __align__(sizeof(FLOAT_TYPE)) unsigned char s_temp[];
    FLOAT_TYPE* l_temp = reinterpret_cast<FLOAT_TYPE*>(s_temp);
    
    unsigned int local_id = threadIdx.x;
    unsigned int id_in_batch = blockIdx.y;
    unsigned int batch_size = gridDim.y;
    unsigned int id_in_time = blockIdx.z;
    unsigned int time_size = gridDim.z;

    unsigned int full_id = local_id + id_in_batch * logit_size + id_in_time * logit_size * batch_size;
    unsigned int part_id = local_id + id_in_batch * vector_size + id_in_time * vector_size * batch_size;
    FLOAT_TYPE full_prob_val = softmax(l_temp, local_id, logit_size, (local_id < logit_size) ? logits[full_id] : 0.0f);
    block_barrier();
    if (full_prob_val == 0.0) full_prob_val = MIN_FLOAT_TYPE;
    if (local_id < logit_size)
    {
        full_probs[full_id] = full_prob_val;
        l_temp[local_id] = full_prob_val;
    }

    block_barrier();
    if (local_id < vector_size)
    {
        FLOAT_TYPE prob = l_temp[labels[local_id + id_in_batch * vector_size]];
        probs[part_id] = prob;
    }
}

// kernel for generation of gradients
// kernel dimension (X,Y,Z)=(max(vetror_size, logit_size),batch_size,time_size)
template <typename FLOAT_TYPE>
__global__ void ctc_finalize( FLOAT_TYPE* grads, unsigned int* validity_of_batch_ids,
     const FLOAT_TYPE* alphas,  const FLOAT_TYPE* betas, 
     const FLOAT_TYPE* probs,  const FLOAT_TYPE* full_probs,
     const int* labels,
     unsigned int vector_size, unsigned int logit_size)
{
    extern __shared__ __align__(sizeof(FLOAT_TYPE)) unsigned char s_temp[];
    FLOAT_TYPE* l_temp = reinterpret_cast<FLOAT_TYPE*>(s_temp);
    unsigned int local_id = threadIdx.x;
    unsigned int id_in_batch = blockIdx.y;
    unsigned int batch_size = gridDim.y;
    unsigned int id_in_time = blockIdx.z;
    unsigned int time_size = gridDim.z;

    unsigned int full_id = local_id + id_in_batch * logit_size + id_in_time * logit_size * batch_size;
    unsigned int part_id = local_id + id_in_batch * vector_size + id_in_time * vector_size * batch_size;

    FLOAT_TYPE ab = 0.0;
    FLOAT_TYPE grad = 0.0;
    FLOAT_TYPE prob = 1.0;

    if (local_id < logit_size)
    {
        l_temp[local_id] = 0.0;
    }
    block_barrier();
    if (local_id < vector_size)
    {
        ab = alphas[part_id] * betas[part_id];
        atomicAdd(l_temp + labels[local_id + id_in_batch * vector_size], ab);
        prob = probs[part_id];
    }
    block_barrier();
    if (local_id < logit_size)
    {
        grad = l_temp[local_id];
    }
    block_barrier();
    FLOAT_TYPE ab_sum = reduce_add(l_temp, local_id, vector_size, ab / prob);
    if (local_id < logit_size)
    {
        FLOAT_TYPE full_prob = full_probs[full_id];
        FLOAT_TYPE denominator = full_prob * ab_sum;
        if (denominator == 0.0) denominator = MIN_FLOAT_TYPE;
        grad = full_prob - grad / denominator;
        grad = grad / batch_size;
        if ((validity_of_batch_ids != NULL) && (isinf(grad) || isnan(grad))) validity_of_batch_ids[id_in_batch] = 0;
        grads[full_id] = grad;
    }
}

template <typename FLOAT_TYPE>
__global__ void ctc_set_zero_to_invalid(FLOAT_TYPE* grads, const unsigned int* validity_of_batch_ids,
    unsigned int logit_size, unsigned int time_size)
{
    unsigned int id_in_batch = blockIdx.y;
    if(validity_of_batch_ids[id_in_batch]) return;
    unsigned int local_id = threadIdx.x + blockDim.x * blockIdx.z;
    unsigned int id_in_time = local_id / logit_size;
    unsigned int logit_id = local_id % logit_size;
    unsigned int batch_size = gridDim.y;
    if(id_in_time >= time_size) return;

    grads[logit_id + id_in_batch * logit_size + id_in_time * logit_size * batch_size] = 0.0;
}

#endif