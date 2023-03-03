const char * ctc_common_file_content = R""""(
#ifndef CTC_COMMON
#define CTC_COMMON
#if defined(WARP_SIZE) && defined(BLOCK_SIZE) && (WARP_SIZE % BLOCK_SIZE == 0) && (BLOCK_SIZE <= WARP_SIZE)
#define WARP_SYNCED
#endif

#ifdef WARP_SYNCED
#define _volatile volatile
inline void block_barrier(cl_mem_fence_flags flags)
{
}
#else
#define _volatile
inline void block_barrier(cl_mem_fence_flags flags)
{
    barrier(flags);
}
#endif

#ifdef USE_DOUBLE
#define MIN_FLOAT_TYPE FLT_MIN
#define ZERO_FLOAT_TYPE 0.0
#define FLOAT_TYPE double
#else
#define MIN_FLOAT_TYPE DBL_MIN
#define ZERO_FLOAT_TYPE 0.0f
#define FLOAT_TYPE float
#endif

//#define CONNECTIONS_PRIVATE_MEM
//#define CONNECTIONS_COUNT 128

#if (!defined(CONNECTIONS_COUNT)) || (CONNECTIONS_COUNT < 1) || (CONNECTIONS_COUNT > 1024)
#undef CONNECTIONS_PRIVATE_MEM
#endif

#ifdef CONNECTIONS_PRIVATE_MEM
    #define CONNECTIONS_SPACE
#else
    #define CONNECTIONS_SPACE __global
#endif

// synced before and after reduce
/*void swipe_up(_volatile __local FLOAT_TYPE* l_reduce_cache, unsigned int id_in_block, unsigned int reduce_size, FLOAT_TYPE value)
{
    _volatile __local FLOAT_TYPE* lt_reduce_cache = l_reduce_cache + id_in_block;
    block_barrier(CLK_LOCAL_MEM_FENCE);
    *lt_reduce_cache = value;
    // reduce part
    block_barrier(CLK_LOCAL_MEM_FENCE);
    for (int i = reduce_size >> 1; i > 0; i >>= 1)
    {
        if (id_in_block < i)
        {
            value += lt_reduce_cache[i];
            lt_reduce_cache[0] = value;
        }
        block_barrier(CLK_LOCAL_MEM_FENCE);
    }
}*/

void swipe_up(_volatile __local FLOAT_TYPE* l_reduce_cache, unsigned int id_in_block, unsigned int reduce_size, FLOAT_TYPE value)
{
    _volatile __local FLOAT_TYPE* lt_reduce_cache = l_reduce_cache + id_in_block;
    block_barrier(CLK_LOCAL_MEM_FENCE);
    *lt_reduce_cache = value;
    // reduce part
    block_barrier(CLK_LOCAL_MEM_FENCE);
    unsigned int aligned_reduce_size = 1 << (32-clz(reduce_size)-1);
    if(id_in_block < reduce_size - aligned_reduce_size)
    {
        value += lt_reduce_cache[aligned_reduce_size];
        lt_reduce_cache[0] = value;
    }
    block_barrier(CLK_LOCAL_MEM_FENCE);
    for (int i = aligned_reduce_size >> 1; i > 0; i >>= 1)
    {
        if (id_in_block < i)
        {
            value += lt_reduce_cache[i];
            lt_reduce_cache[0] = value;
        }
        block_barrier(CLK_LOCAL_MEM_FENCE);
    }
}

// synced only before reduce
FLOAT_TYPE reduce_add(_volatile __local FLOAT_TYPE* l_reduce_cache, unsigned int id_in_block, unsigned int reduce_size, FLOAT_TYPE value)
{
    swipe_up(l_reduce_cache, id_in_block, reduce_size, value);
    return l_reduce_cache[0];
}

// synced only before reduce
FLOAT_TYPE normalize_mul(_volatile __local FLOAT_TYPE* l_reduce_cache, unsigned int id_in_block, unsigned int reduce_size, FLOAT_TYPE value, FLOAT_TYPE coef)
{
    return value / reduce_add(l_reduce_cache, id_in_block, reduce_size, value * coef);
}

// synced only before reduce
FLOAT_TYPE normalize_l(_volatile __local FLOAT_TYPE* l_reduce_cache, unsigned int id_in_block, unsigned int reduce_size, FLOAT_TYPE value)
{
    return value / reduce_add(l_reduce_cache, id_in_block, reduce_size, value);
}

FLOAT_TYPE softmax(_volatile __local FLOAT_TYPE* l_reduce_cache, unsigned int id_in_block, unsigned int reduce_size, FLOAT_TYPE value)
{
    FLOAT_TYPE exp_value = exp(value);
    return exp_value / reduce_add(l_reduce_cache, id_in_block, reduce_size, exp_value);
}

#ifdef CONNECTIONS_PRIVATE_MEM
FLOAT_TYPE vector_mat_mul(_volatile __local const FLOAT_TYPE* vector, const FLOAT_TYPE* mat, unsigned int vector_size)
{
    FLOAT_TYPE val = 0.0f;
    #pragma unroll
    for (int i = 0; i < CONNECTIONS_COUNT; i++)
    {
        val += vector[i] * mat[i];
    }
    return val;
}
#else
FLOAT_TYPE vector_mat_mul(_volatile __local const FLOAT_TYPE* vector, __global const FLOAT_TYPE* mat, unsigned int vector_size)
{
    FLOAT_TYPE val = 0.0f;
    for (int i = 0; i < vector_size; i++)
    {
        val += vector[i] * mat[i * vector_size];
    }
    return val;
}
#endif


#endif
)"""";