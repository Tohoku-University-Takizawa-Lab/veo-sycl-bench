#include "common.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <ve_offload.h>

void init_arrays(DATA_TYPE* A, DATA_TYPE* C, size_t size) {
    const size_t N = size;
    const size_t M = size;

    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < M; j++) {
            A[i * M + j] = ((DATA_TYPE)i * j) / N;
        }

        for (size_t j = 0; j < N; j++) {
            C[i * M + j] = ((DATA_TYPE)i * j + 2) / N;
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
    long problem_bytes = size * size * sizeof(DATA_TYPE);
    DATA_TYPE* A = (DATA_TYPE*)malloc(problem_bytes);
    DATA_TYPE* C = (DATA_TYPE*)malloc(problem_bytes);
    init_arrays(A, C, size);

    uint64_t A_ptr;
    veo_alloc_mem(proc, &A_ptr, problem_bytes);
    veo_write_mem(proc, A_ptr, (void*)A, problem_bytes);
    uint64_t C_ptr;
    veo_alloc_mem(proc, &C_ptr, problem_bytes);
    veo_write_mem(proc, C_ptr, (void*)C, problem_bytes);

    struct veo_args* argp = veo_args_alloc();
    veo_args_set_i64(argp, 0, A_ptr);
    veo_args_set_i64(argp, 1, C_ptr);
    veo_args_set_i64(argp, 2, size);

    uint64_t kernel_syrk = veo_get_sym(proc, handle, "kernel_syrk");
    uint64_t id = veo_call_async(ctx, kernel_syrk, argp);
    uint64_t retval;
    veo_call_wait_result(ctx, id, &retval);

    veo_free_mem(proc, A_ptr);
    veo_free_mem(proc, C_ptr);
    veo_args_free(argp);
    free(A);
    free(C);
    veo_context_close(ctx);
    veo_proc_destroy(proc);
    long finish = get_time();
    printf("runtime %ld ms\n", finish - begin);
    return 0;
}