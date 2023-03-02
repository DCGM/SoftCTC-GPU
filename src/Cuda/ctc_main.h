#ifndef CTC_MAIN
#define CTC_MAIN

#include "ctc_common.h"

template <typename FLOAT_TYPE, unsigned int VECTOR_SIZE>
__device__ inline void load_connections(FLOAT_TYPE* p_connections, FLOAT_TYPE* l_connections, const FLOAT_TYPE* g_connections)
{
    #pragma unroll
    for (int i = 0; i < ((VECTOR_SIZE > REG_LIMIT) ? REG_LIMIT : VECTOR_SIZE); i++)
    {
        p_connections[i] = g_connections[VECTOR_SIZE * i];
    }
    #pragma unroll
    for (int i = REG_LIMIT; i < VECTOR_SIZE; i++)
    {
        l_connections[VECTOR_SIZE * i] = g_connections[VECTOR_SIZE * i];
    }
}

template <typename FLOAT_TYPE, unsigned int VECTOR_SIZE>
__device__ inline void ctc_step(FLOAT_TYPE* loss, unsigned int* validity_of_batch_ids,
    FLOAT_TYPE* coef, const FLOAT_TYPE* probs,
    const FLOAT_TYPE* connections, const FLOAT_TYPE* connections_start, const FLOAT_TYPE* connections_end,
    unsigned int vector_size, int start_time, int end_time, int time_change, unsigned int norm_step, FLOAT_TYPE* l_temp)
{
    FLOAT_TYPE* l_val = l_temp + vector_size;
    unsigned int id_in_batch = blockIdx.y;
    unsigned int batch_size = gridDim.y;
    unsigned int id_in_vector = threadIdx.x;

    unsigned int NL_id = id_in_batch * vector_size + id_in_vector;

    FLOAT_TYPE gr_ll = 0.0f;
    unsigned int TNL_id = NL_id + start_time * batch_size * vector_size;
    FLOAT_TYPE val = connections_start[NL_id] * probs[TNL_id];
    
    // normalize
    val = normalize_l(l_temp, id_in_vector, vector_size, val);
    if (id_in_vector == 0) gr_ll += log(l_temp[0]); //accumulate normalization coefs
    coef[TNL_id] = val;

    for (int act_time = start_time + time_change; act_time != end_time; act_time += time_change)
    {
        TNL_id = NL_id + act_time * batch_size * vector_size;
        l_val[id_in_vector] = val;
        block_barrier();
        val = vector_mat_mul<FLOAT_TYPE, VECTOR_SIZE>(l_val, connections, l_temp + 2 * vector_size + id_in_vector - REG_LIMIT * vector_size, vector_size) * probs[TNL_id];
        block_barrier();
        if(act_time % norm_step == 0) 
        {
            val = normalize_l(l_temp, id_in_vector, vector_size, val);
            if (id_in_vector == 0) gr_ll += log(l_temp[0]);
        }
        coef[TNL_id] = val;
    }
    block_barrier();
    val = normalize_mul(l_temp, id_in_vector, vector_size, val, connections_end[NL_id]);
    if (id_in_vector == 0) gr_ll += log(l_temp[0]);
    coef[TNL_id] = val;
    if (id_in_vector == 0)
    {
        FLOAT_TYPE act_loss = -gr_ll;
        if (validity_of_batch_ids != NULL)
        {
            bool is_act_batch_id_valid = (!isnan(act_loss) && !isinf(act_loss));
            validity_of_batch_ids[id_in_batch] = is_act_batch_id_valid;
            if (!is_act_batch_id_valid) act_loss = 0.0;
        }
        loss[id_in_batch] = act_loss;
    }
}

// kernel dimension (X,Y,Z)=(vetror_size,batch_size,1)
template <typename FLOAT_TYPE, unsigned int VECTOR_SIZE>
__global__ void ctc_forward( FLOAT_TYPE* loss, unsigned int* validity_of_batch_ids,
     FLOAT_TYPE* alphas,  const FLOAT_TYPE* probs,
     const FLOAT_TYPE* forward,  const FLOAT_TYPE* forward_start,  const FLOAT_TYPE* forward_end,
    unsigned int time_size, unsigned int vector_size, unsigned int norm_step)
{
    if (VECTOR_SIZE != 1) vector_size = VECTOR_SIZE;
    extern __shared__ __align__(sizeof(FLOAT_TYPE)) unsigned char s_temp[];
    FLOAT_TYPE* l_temp = reinterpret_cast<FLOAT_TYPE*>(s_temp);

    unsigned int id_in_batch = blockIdx.y;
    unsigned int id_in_vector = threadIdx.x;

    const FLOAT_TYPE* forward_backward_g = forward + id_in_vector + id_in_batch * vector_size * vector_size;
    FLOAT_TYPE forward_backward_p[VECTOR_SIZE];
    if (VECTOR_SIZE != 1) load_connections<FLOAT_TYPE, VECTOR_SIZE>(forward_backward_p, l_temp + vector_size * 2 + id_in_vector - REG_LIMIT * vector_size, forward_backward_g);
    const FLOAT_TYPE* forward_backward = (VECTOR_SIZE == 1) ? forward_backward_g : forward_backward_p;

    ctc_step<FLOAT_TYPE, VECTOR_SIZE>(loss, validity_of_batch_ids, alphas, probs, forward_backward, forward_start, forward_end, vector_size, 0, time_size, 1, norm_step, l_temp);
}

// kernel dimension (X,Y,Z)=(vetror_size,batch_size,1)
template <typename FLOAT_TYPE, unsigned int VECTOR_SIZE>
__global__ void ctc_backward( FLOAT_TYPE* ll_backward,
     FLOAT_TYPE* betas,  const FLOAT_TYPE* probs,
     const FLOAT_TYPE* backward,  const FLOAT_TYPE* backward_start,  const FLOAT_TYPE* backward_end,
    unsigned int time_size, unsigned int vector_size, unsigned int norm_step)
{
    if (VECTOR_SIZE != 1) vector_size = VECTOR_SIZE;
    extern __shared__ __align__(sizeof(FLOAT_TYPE)) unsigned char s_temp[];
    FLOAT_TYPE* l_temp = reinterpret_cast<FLOAT_TYPE*>(s_temp);

    unsigned int id_in_batch = blockIdx.y;
    unsigned int id_in_vector = threadIdx.x;
    const FLOAT_TYPE* forward_backward_g = backward + id_in_vector + id_in_batch * vector_size * vector_size;
    FLOAT_TYPE forward_backward_p[VECTOR_SIZE];
    if (VECTOR_SIZE != 1) load_connections<FLOAT_TYPE, VECTOR_SIZE>(forward_backward_p, l_temp + vector_size * 2 + id_in_vector - REG_LIMIT * vector_size, forward_backward_g);
    const FLOAT_TYPE* forward_backward = (VECTOR_SIZE == 1) ? forward_backward_g : forward_backward_p;

    ctc_step<FLOAT_TYPE, VECTOR_SIZE>(ll_backward, NULL, betas, probs, forward_backward, backward_start, backward_end, vector_size, time_size - 1, -1, -1, norm_step, l_temp);
}

// kernel dimension (X,Y,Z)=(vetror_size,batch_size,2)
template <typename FLOAT_TYPE, unsigned int VECTOR_SIZE>
__global__ void ctc_forward_backward( FLOAT_TYPE* loss,  FLOAT_TYPE* ll_backward, unsigned int* validity_of_batch_ids,
     FLOAT_TYPE* alphas,  FLOAT_TYPE* betas,  const FLOAT_TYPE* probs,
     const FLOAT_TYPE* forward,  const FLOAT_TYPE* forward_start,  const FLOAT_TYPE* forward_end,
     const FLOAT_TYPE* backward,  const FLOAT_TYPE* backward_start,  const FLOAT_TYPE* backward_end,
    unsigned int time_size, unsigned int vector_size, unsigned int norm_step)
{
    if (VECTOR_SIZE != 1) vector_size = VECTOR_SIZE;
    extern __shared__ __align__(sizeof(FLOAT_TYPE)) unsigned char s_temp[];
    FLOAT_TYPE* l_temp = reinterpret_cast<FLOAT_TYPE*>(s_temp);

    unsigned int id_in_batch = blockIdx.y;
    unsigned int id_in_vector = threadIdx.x;
    const FLOAT_TYPE* forward_backward_g = ((blockIdx.z == 0) ? forward : backward) + id_in_vector + id_in_batch * vector_size * vector_size;
    FLOAT_TYPE forward_backward_p[VECTOR_SIZE];
    if(VECTOR_SIZE != 1) load_connections<FLOAT_TYPE, VECTOR_SIZE>(forward_backward_p, l_temp + vector_size * 2 + id_in_vector - REG_LIMIT * vector_size, forward_backward_g);
    const FLOAT_TYPE* forward_backward = (VECTOR_SIZE == 1) ? forward_backward_g : forward_backward_p;
    if (blockIdx.z == 0) ctc_step<FLOAT_TYPE, VECTOR_SIZE>(loss, validity_of_batch_ids, alphas, probs, forward_backward, forward_start, forward_end, vector_size, 0, time_size, 1, norm_step, l_temp);
    else ctc_step<FLOAT_TYPE, VECTOR_SIZE>(ll_backward, NULL, betas, probs, forward_backward, backward_start, backward_end, vector_size, time_size - 1, -1, -1, norm_step, l_temp);
}

#endif