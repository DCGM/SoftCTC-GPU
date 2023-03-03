const char* ctc_main_file_content = R""""(

#ifdef CONNECTIONS_PRIVATE_MEM
inline void load_connections(FLOAT_TYPE* p_connections, __global const FLOAT_TYPE* g_connections)
{
    #pragma unroll
    for (int i = 0; i < CONNECTIONS_COUNT; i++)
    {
        p_connections[i] = g_connections[CONNECTIONS_COUNT * i];
    }
}
#endif

void ctc_step(__global FLOAT_TYPE* loss, __global unsigned int* validity_of_batch_ids,
    __global FLOAT_TYPE* coef, __global const FLOAT_TYPE* probs,
    CONNECTIONS_SPACE const FLOAT_TYPE* connections, __global const FLOAT_TYPE* connections_start, __global const FLOAT_TYPE* connections_end,
    unsigned int vector_size, int start_time, int end_time, int time_change, unsigned int norm_step,
    _volatile __local FLOAT_TYPE *l_temp)
{
    _volatile __local FLOAT_TYPE* l_val = l_temp + vector_size;
    unsigned int id_in_batch = get_group_id(1);
    unsigned int batch_size = get_num_groups(1);
    unsigned int id_in_vector = get_local_id(0);

    unsigned int NL_id = id_in_batch * vector_size + id_in_vector;
    unsigned int mat_first_id = id_in_vector + id_in_batch * vector_size * vector_size;

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
        block_barrier(CLK_LOCAL_MEM_FENCE);
        val = vector_mat_mul(l_val, connections, vector_size) * probs[TNL_id];
        block_barrier(CLK_LOCAL_MEM_FENCE);
        if(act_time % norm_step == 0) 
        {
            val = normalize_l(l_temp, id_in_vector, vector_size, val);
            if (id_in_vector == 0) gr_ll += log(l_temp[0]);
        }
        coef[TNL_id] = val;
    }
    block_barrier(CLK_LOCAL_MEM_FENCE);
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
            if (!is_act_batch_id_valid) act_loss = 0.0f;
        }
        loss[id_in_batch] = act_loss;
    }
    block_barrier(CLK_GLOBAL_MEM_FENCE);
}

// kernel dimension (X,Y,Z)=(vetror_size,batch_size,1)
__kernel void ctc_forward(__global FLOAT_TYPE* loss, __global unsigned int* validity_of_batch_ids,
    __global FLOAT_TYPE* alphas, __global const FLOAT_TYPE* probs,
    __global const FLOAT_TYPE* forward, __global const FLOAT_TYPE* forward_start, __global const FLOAT_TYPE* forward_end,
    unsigned int time_size, unsigned int vector_size, unsigned int norm_step, _volatile __local FLOAT_TYPE *l_temp)
{
    unsigned int id_in_batch = get_group_id(1);
    unsigned int id_in_vector = get_local_id(0);
    unsigned int mat_first_id = id_in_vector + id_in_batch * vector_size * vector_size;
#ifdef CONNECTIONS_PRIVATE_MEM
    FLOAT_TYPE connections_temp[CONNECTIONS_COUNT];
    load_connections(connections_temp, forward + mat_first_id);
#else
    __global const FLOAT_TYPE* connections_temp = forward + mat_first_id;
#endif
    ctc_step(loss, validity_of_batch_ids, alphas, probs, connections_temp, forward_start, forward_end, vector_size, 0, time_size, 1, norm_step, l_temp);
}

// kernel dimension (X,Y,Z)=(vetror_size,batch_size,1)
__kernel void ctc_backward(__global FLOAT_TYPE* ll_backward,
    __global FLOAT_TYPE* betas, __global const FLOAT_TYPE* probs,
    __global const FLOAT_TYPE* backward, __global const FLOAT_TYPE* backward_start, __global const FLOAT_TYPE* backward_end,
    unsigned int time_size, unsigned int vector_size, unsigned int norm_step, _volatile __local FLOAT_TYPE* l_temp)
{
    unsigned int id_in_batch = get_group_id(1);
    unsigned int id_in_vector = get_local_id(0);
    unsigned int mat_first_id = id_in_vector + id_in_batch * vector_size * vector_size;
#ifdef CONNECTIONS_PRIVATE_MEM
    FLOAT_TYPE connections_temp[CONNECTIONS_COUNT];
    load_connections(connections_temp, backward + mat_first_id);
#else
    __global const FLOAT_TYPE* connections_temp = backward + mat_first_id;
#endif
    ctc_step(ll_backward, NULL, betas, probs, connections_temp, backward_start, backward_end, vector_size, time_size - 1, -1, -1, norm_step, l_temp);
}

// kernel dimension (X,Y,Z)=(vetror_size,batch_size,2)
__kernel void ctc_forward_backward(__global FLOAT_TYPE* loss, __global FLOAT_TYPE* ll_backward, __global unsigned int* validity_of_batch_ids,
    __global FLOAT_TYPE* alphas, __global FLOAT_TYPE* betas, __global const FLOAT_TYPE* probs,
    __global const FLOAT_TYPE* forward, __global const FLOAT_TYPE* forward_start, __global const FLOAT_TYPE* forward_end,
    __global const FLOAT_TYPE* backward, __global const FLOAT_TYPE* backward_start, __global const FLOAT_TYPE* backward_end,
    unsigned int time_size, unsigned int vector_size, unsigned int norm_step, _volatile __local FLOAT_TYPE* l_temp)
{
    unsigned int id_in_batch = get_group_id(1);
    unsigned int id_in_vector = get_local_id(0);
    unsigned int mat_first_id = id_in_vector + id_in_batch * vector_size * vector_size;
#ifdef CONNECTIONS_PRIVATE_MEM
    FLOAT_TYPE connections_temp[CONNECTIONS_COUNT];
    load_connections(connections_temp, ((get_global_id(2) == 0) ? forward : backward) + mat_first_id);
#else
    __global const FLOAT_TYPE* connections_temp = ((get_global_id(2) == 0) ? forward : backward) + mat_first_id;
#endif
    if (get_global_id(2) == 0) ctc_step(loss, validity_of_batch_ids, alphas, probs, connections_temp, forward_start, forward_end, vector_size, 0, time_size, 1, norm_step, l_temp);
    else ctc_step(ll_backward, NULL, betas, probs, connections_temp, backward_start, backward_end, vector_size, time_size - 1, -1, -1, norm_step, l_temp);
}
)"""";
