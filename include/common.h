#ifndef __COMMON_H
#define __COMMON_H
#include <stdlib.h>
#include <sys/time.h>

#define N_ITERS 5 // simulation iterations
#define DATA_TYPE float

long get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000 + tv.tv_usec / 1000;
}

#endif
