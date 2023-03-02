#include "ctc_common.cl"

// kernel for generation of full_probs and probs
// kernel dimension (X,Y,Z)=(max(vetror_size, logit_size),batch_size,time_size)
__kernel void ctc_init(__global FLOAT_TYPE* full_probs, __global FLOAT_TYPE* probs,
    __global const int* labels, __global const FLOAT_TYPE* logits,
    unsigned int vector_size, unsigned int logit_size, _volatile __local FLOAT_TYPE *l_temp)
{
    unsigned int local_id = get_local_id(0);
    unsigned int id_in_batch = get_group_id(1);
    unsigned int batch_size = get_num_groups(1);
    unsigned int id_in_time = get_group_id(2);
    unsigned int time_size = get_num_groups(2);

    unsigned int full_id = local_id + id_in_batch * logit_size + id_in_time * logit_size * batch_size;
    unsigned int part_id = local_id + id_in_batch * vector_size + id_in_time * vector_size * batch_size;
    FLOAT_TYPE full_prob_val = softmax(l_temp, local_id, logit_size, (local_id < logit_size) ? logits[full_id] : 0.0);
    block_barrier(CLK_LOCAL_MEM_FENCE);
    if (full_prob_val == 0.0) full_prob_val = MIN_FLOAT_TYPE;
    if (local_id < logit_size)
    {
        full_probs[full_id] = full_prob_val;
        l_temp[local_id] = full_prob_val;
    }

    block_barrier(CLK_LOCAL_MEM_FENCE);
    if (local_id < vector_size)
    {
        FLOAT_TYPE prob = l_temp[labels[local_id + id_in_batch * vector_size]];
        probs[part_id] = prob;
    }
}

#ifdef cl_nv_pragma_unroll
inline void atomic_red_shared_f32(_volatile __local float *ptr, float val)
{
    asm("red.shared.add.f32 [%0], %1;" :: "l"(ptr), "f"(val));
}

inline void atomic_red_shared_f64(_volatile __local double* ptr, double val)
{
    asm("red.shared.add.f64 [%0], %1;" :: "l"(ptr), "d"(val));
}
#else
inline void atomic_red_shared_f32(_volatile __local float *ptr, float val)
{
    atomic_fetch_add(ptr, val);
}

inline void atomic_red_shared_f64(_volatile __local double* ptr, double val)
{
    atomic_fetch_add(ptr, val);
}
#endif

inline void atomic_red_shared_float(_volatile __local FLOAT_TYPE* ptr, FLOAT_TYPE val)
{
#ifdef USE_DOUBLE
    atomic_red_shared_f64(ptr, val);
#else
    atomic_red_shared_f32(ptr, val);
#endif
}


// kernel for generation of gradients
// kernel dimension (X,Y,Z)=(max(vetror_size, logit_size),batch_size,time_size)
__kernel void ctc_finalize(__global FLOAT_TYPE* grads, __global unsigned int* validity_of_batch_ids,
    __global const FLOAT_TYPE* alphas, __global const FLOAT_TYPE* betas, 
    __global const FLOAT_TYPE* probs, __global const FLOAT_TYPE* full_probs,
    __global const int* labels,
    unsigned int vector_size, unsigned int logit_size, _volatile __local FLOAT_TYPE* l_temp)
{
    unsigned int local_id = get_local_id(0);
    unsigned int id_in_batch = get_group_id(1);
    unsigned int batch_size = get_num_groups(1);
    unsigned int id_in_time = get_group_id(2);
    unsigned int time_size = get_num_groups(2);

    unsigned int full_id = local_id + id_in_batch * logit_size + id_in_time * logit_size * batch_size;
    unsigned int part_id = local_id + id_in_batch * vector_size + id_in_time * vector_size * batch_size;

    FLOAT_TYPE ab = 0.0;
    FLOAT_TYPE grad = 0.0;
    FLOAT_TYPE prob = 1.0;

    if (local_id < logit_size)
    {
        l_temp[local_id] = 0.0;
    }
    block_barrier(CLK_LOCAL_MEM_FENCE);
    if (local_id < vector_size)
    {
        ab = alphas[part_id] * betas[part_id];
        atomic_red_shared_float(l_temp + labels[local_id + id_in_batch * vector_size], ab); // fix atomic add to FLOAT_TYPE
        prob = probs[part_id];
    }
    block_barrier(CLK_LOCAL_MEM_FENCE);
    if (local_id < logit_size)
    {
        grad = l_temp[local_id];
    }
    block_barrier(CLK_LOCAL_MEM_FENCE);/**/
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

__kernel void ctc_set_zero_to_invalid(__global FLOAT_TYPE* grads, __constant const unsigned int* validity_of_batch_ids,
    unsigned int logit_size, unsigned int time_size)
{
    unsigned int id_in_batch = get_group_id(1);
    if (validity_of_batch_ids[id_in_batch]) return;
    unsigned int local_id = get_local_id(0) + get_local_size(0) * get_group_id(2);
    unsigned int id_in_time = local_id / logit_size;
    unsigned int logit_id = local_id % logit_size;
    unsigned int batch_size = get_num_groups(1);
    if (id_in_time >= time_size) return;

    grads[logit_id + id_in_batch * logit_size + id_in_time * logit_size * batch_size] = 0.0;
}
