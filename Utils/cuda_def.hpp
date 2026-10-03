#pragma once

#if defined(__CUDACC__)
    #define HD_DEF __host__ __device__
#else
    #define HD_DEF
#endif
