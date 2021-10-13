#include "common.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <ve_offload.h>

void init_array(DATA_TYPE* A, size_t size) {
    const size_t M = 0;
    const size_t N = 0;

    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            A[i * N + j] = ((DATA_TYPE)(i + 1) * (j + 1)) / (M + 1);
        }
    }
}

int main(int argc, char** argv) {
    long begin = get_time();
    size_t size = 5;
    if (argc > 1)
        size = atoi(argv[1]);

    struct veo_proc_handle* proc = veo_proc_create(0);
    uint64_t handle = veo_load_library(proc, "./kernel.so");
    if (handle == 0) {
        perror("[VEO]:veo failed to load the shared library\n");
        return 0;
    }
    struct veo_thr_ctxt* ctx = veo_context_open(proc);
    int problem_bytes = size * size * sizeof(DATA_TYPE);
    DATA_TYPE* A = (DATA_TYPE*)malloc(problem_bytes);
    init_array(A, size);

    uint64_t A_ptr;
    veo_alloc_mem(proc, &A_ptr, problem_bytes);
    veo_write_mem(proc, A_ptr, (void*)A, problem_bytes);
    uint64_t R_ptr;
    veo_alloc_mem(proc, &R_ptr, problem_bytes);
    uint64_t Q_ptr;
    veo_alloc_mem(proc, &Q_ptr, problem_bytes);

    struct veo_args* argp = veo_args_alloc();
    veo_args_set_i64(argp, 0, A_ptr);
    veo_args_set_i64(argp, 1, R_ptr);
    veo_args_set_i64(argp, 2, Q_ptr);
    veo_args_set_i64(argp, 3, size);

    uint64_t gramschmidt = veo_get_sym(proc, handle, "gramschmidt");
    uint64_t id = veo_call_async(ctx, gramschmidt, argp);
    uint64_t retval;
    veo_call_wait_result(ctx, id, &retval);
    veo_free_mem(proc, A_ptr);
    veo_free_mem(proc, R_ptr);
    veo_free_mem(proc, Q_ptr);
    veo_args_free(argp);
    free(A);
    veo_context_close(ctx);
    veo_proc_destroy(proc);
    long finish = get_time();
    printf("runtime %ld ms\n", finish - begin);
    return 0;
}