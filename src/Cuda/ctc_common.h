#ifndef CTC_COMMON
#define CTC_COMMON

#define REG_LIMIT 212

#ifdef BLOCK_SYNCED
__device__ inline void block_barrier()
{
    __syncwarp();
}
#else
__device__ inline void block_barrier()
{
    __syncthreads();
}
#endif

// synced before and after reduce
template <typename FLOAT_TYPE>
__device__ void swipe_up(FLOAT_TYPE* l_reduce_cache, unsigned int id_in_block, unsigned int reduce_size, FLOAT_TYPE value)
{
    
    FLOAT_TYPE* lt_reduce_cache = l_reduce_cache + id_in_block;
    block_barrier();
    *lt_reduce_cache = value;
    // reduce part
    block_barrier();
    unsigned int aligned_reduce_size = 1 << (32-__clz(reduce_size)-1);
    if(id_in_block < reduce_size - aligned_reduce_size)
    {
        value += lt_reduce_cache[aligned_reduce_size];
        lt_reduce_cache[0] = value;
    }
    block_barrier();
    for (int i = aligned_reduce_size >> 1; i > 0; i >>= 1)
    {
        if (id_in_block < i)
        {
            value += lt_reduce_cache[i];
            lt_reduce_cache[0] = value;
        }
        block_barrier();
    }
}

//atomicAdd instead of parallel reduction is slower
/*template <typename FLOAT_TYPE>
__device__ void swipe_up(FLOAT_TYPE* l_reduce_cache, unsigned int id_in_block, unsigned int reduce_size, FLOAT_TYPE value)
{
    if (id_in_block == 0) l_reduce_cache[0] = 0.0;
    //FLOAT_TYPE* lt_reduce_cache = l_reduce_cache + id_in_block;
    block_barrier();
    if(id_in_block < reduce_size) atomicAdd(l_reduce_cache, value);
    block_barrier();
}*/

// synced only before reduce
template <typename FLOAT_TYPE>
__device__ FLOAT_TYPE reduce_add(FLOAT_TYPE* l_reduce_cache, unsigned int id_in_block, unsigned int reduce_size, FLOAT_TYPE value)
{
    swipe_up(l_reduce_cache, id_in_block, reduce_size, value);
    return l_reduce_cache[0];
}

// synced only before reduce
template <typename FLOAT_TYPE>
__device__ FLOAT_TYPE normalize_mul(FLOAT_TYPE* l_reduce_cache, unsigned int id_in_block, unsigned int reduce_size, FLOAT_TYPE value, FLOAT_TYPE coef)
{
    return value / reduce_add(l_reduce_cache, id_in_block, reduce_size, value * coef);
}

// synced only before reduce
template <typename FLOAT_TYPE>
__device__ FLOAT_TYPE normalize_l(FLOAT_TYPE* l_reduce_cache, unsigned int id_in_block, unsigned int reduce_size, FLOAT_TYPE value)
{
    return value / reduce_add(l_reduce_cache, id_in_block, reduce_size, value);
}

template <typename FLOAT_TYPE>
__device__ FLOAT_TYPE softmax(FLOAT_TYPE* l_reduce_cache, unsigned int id_in_block, unsigned int reduce_size, FLOAT_TYPE value)
{
    FLOAT_TYPE exp_value = exp(value);
    return exp_value / reduce_add(l_reduce_cache, id_in_block, reduce_size, exp_value);
}


template <typename FLOAT_TYPE, unsigned int VECTOR_SIZE>
__device__ FLOAT_TYPE vector_mat_mul(const FLOAT_TYPE* vector, const FLOAT_TYPE* mat, const FLOAT_TYPE* l_mat, unsigned int vector_size)
{
    FLOAT_TYPE val = 0.0f;
    if (VECTOR_SIZE == 1)
    {
        for (int i = 0; i < vector_size; i++)
        {
            val += vector[i] * mat[i * vector_size];
        }
    }
    else
    {
        #pragma unroll
        for (int i = 0; i < ((VECTOR_SIZE > REG_LIMIT) ? REG_LIMIT : VECTOR_SIZE); i++)
        {
            val += vector[i] * mat[i];
        }

        #pragma unroll
        for (int i = REG_LIMIT; i < VECTOR_SIZE; i++)
        {
            val += vector[i] * l_mat[i * VECTOR_SIZE];
        }
    }
    return val;
}

#endif