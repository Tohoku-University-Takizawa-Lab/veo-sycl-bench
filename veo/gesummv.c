#include "common.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <ve_offload.h>

void init(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* x, size_t size) {
    const size_t N = size;
    for (size_t i = 0; i < N; i++) {
        x[i] = 1;
        for (size_t j = 0; j < N; j++) {
            A[i * N + j] = 2;
            B[i * N + j] = 3;
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
    DATA_TYPE* B = (DATA_TYPE*)malloc(problem_bytes);
    DATA_TYPE* x = (DATA_TYPE*)malloc(size * sizeof(DATA_TYPE));
    init(A, B, x, size);

    uint64_t A_ptr;
    veo_alloc_mem(proc, &A_ptr, problem_bytes);
    veo_write_mem(proc, A_ptr, (void*)A, problem_bytes);
    uint64_t B_ptr;
    veo_alloc_mem(proc, &B_ptr, problem_bytes);
    veo_write_mem(proc, B_ptr, (void*)B, problem_bytes);
    uint64_t x_ptr;
    veo_alloc_mem(proc, &x_ptr, size * sizeof(DATA_TYPE));
    veo_write_mem(proc, x_ptr, (void*)x, size * sizeof(DATA_TYPE));
    uint64_t y_ptr;
    veo_alloc_mem(proc, &y_ptr, size * sizeof(DATA_TYPE));
    uint64_t tmp_ptr;
    veo_alloc_mem(proc, &tmp_ptr, size * sizeof(DATA_TYPE));

    struct veo_args* argp = veo_args_alloc();
    veo_args_set_i64(argp, 0, A_ptr);
    veo_args_set_i64(argp, 1, B_ptr);
    veo_args_set_i64(argp, 2, x_ptr);
    veo_args_set_i64(argp, 3, y_ptr);
    veo_args_set_i64(argp, 4, tmp_ptr);
    veo_args_set_i64(argp, 5, size);

    uint64_t kernel_gesummv = veo_get_sym(proc, handle, "kernel_gesummv");
    uint64_t id = veo_call_async(ctx, kernel_gesummv, argp);
    uint64_t retval;
    veo_call_wait_result(ctx, id, &retval);

    veo_free_mem(proc, A_ptr);
    veo_free_mem(proc, B_ptr);
    veo_free_mem(proc, x_ptr);
    veo_free_mem(proc, y_ptr);
    veo_free_mem(proc, tmp_ptr);
    veo_args_free(argp);
    free(A);
    free(B);
    free(x);
    veo_context_close(ctx);
    veo_proc_destroy(proc);
    long finish = get_time();
    printf("runtime %ld ms\n", finish - begin);
    return 0;
}