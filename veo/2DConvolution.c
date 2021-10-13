#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <ve_offload.h>

#include "common.h"

void init(DATA_TYPE* A, size_t size) {
    const size_t NI = size;
    const size_t NJ = size;

    for (size_t i = 0; i < NI; ++i) {
        for (size_t j = 0; j < NJ; ++j) {
            A[i * NJ + j] = (DATA_TYPE)rand() / (DATA_TYPE)RAND_MAX;
        }
    }
}
int main(int argc, char** argv) {
    long begin = get_time();
    size_t size = 5;
    if (argc > 1)
        size = atoi(argv[1]);

    long totalTime = 0;
    struct veo_proc_handle* proc = veo_proc_create(0);
    uint64_t handle = veo_load_library(proc, "./kernel.so");
    if (handle == 0) {
        perror("[VEO]:veo failed to load the shared library\n");
        return 0;
    }
    struct veo_thr_ctxt* ctx = veo_context_open(proc);
    int problem_bytes = size * size * sizeof(DATA_TYPE);
    DATA_TYPE* A = (DATA_TYPE*)malloc(problem_bytes);
    init(A, size);

    uint64_t A_ptr;
    veo_alloc_mem(proc, &A_ptr, problem_bytes);
    veo_write_mem(proc, A_ptr, (void*)A, problem_bytes);
    uint64_t B_ptr;
    veo_alloc_mem(proc, &B_ptr, problem_bytes);

    struct veo_args* argp = veo_args_alloc();
    veo_args_set_i64(argp, 0, A_ptr);
    veo_args_set_i64(argp, 1, B_ptr);
    veo_args_set_i64(argp, 2, size);
    uint64_t conv2D = veo_get_sym(proc, handle, "conv2D");

    uint64_t id = veo_call_async(ctx, conv2D, argp);
    uint64_t retval;
    veo_call_wait_result(ctx, id, &retval);
    veo_free_mem(proc, A_ptr);
    veo_free_mem(proc, B_ptr);
    veo_args_free(argp);
    free(A);

    veo_context_close(ctx);
    veo_proc_destroy(proc);
    long finish = get_time();
    printf("runtime %ld ms\n", finish - begin);
    return 0;
}
