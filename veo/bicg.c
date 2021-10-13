#include "common.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <ve_offload.h>

#ifndef M_PI
#define M_PI 3.14159
#endif

void init_array(DATA_TYPE* A, DATA_TYPE* p, DATA_TYPE* r, size_t size) {
    const size_t NX = size;
    const size_t NY = size;
    for (size_t i = 0; i < NX; i++) {
        r[i] = i * M_PI;

        for (size_t j = 0; j < NY; j++) {
            A[i * NY + j] = ((DATA_TYPE)i * j) / NX;
        }
    }
    for (size_t i = 0; i < NY; i++) {
        p[i] = i * M_PI;
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
    DATA_TYPE* p = (DATA_TYPE*)malloc(size * sizeof(DATA_TYPE));
    DATA_TYPE* r = (DATA_TYPE*)malloc(size * sizeof(DATA_TYPE));
    init_array(A, p, r, size);

    uint64_t A_ptr;
    veo_alloc_mem(proc, &A_ptr, problem_bytes);
    veo_write_mem(proc, A_ptr, (void*)A, problem_bytes);
    uint64_t p_ptr;
    veo_alloc_mem(proc, &p_ptr, size * sizeof(DATA_TYPE));
    veo_write_mem(proc, p_ptr, (void*)p, size * sizeof(DATA_TYPE));
    uint64_t r_ptr;
    veo_alloc_mem(proc, &r_ptr, size * sizeof(DATA_TYPE));
    veo_write_mem(proc, r_ptr, (void*)r, size * sizeof(DATA_TYPE));
    uint64_t s_ptr;
    veo_alloc_mem(proc, &s_ptr, size * sizeof(DATA_TYPE));
    uint64_t q_ptr;
    veo_alloc_mem(proc, &q_ptr, size * sizeof(DATA_TYPE));

    struct veo_args* argp = veo_args_alloc();
    veo_args_set_i64(argp, 0, A_ptr);
    veo_args_set_i64(argp, 1, r_ptr);
    veo_args_set_i64(argp, 2, s_ptr);
    veo_args_set_i64(argp, 3, p_ptr);
    veo_args_set_i64(argp, 4, q_ptr);
    veo_args_set_i64(argp, 5, size);

    uint64_t kernel_bicg = veo_get_sym(proc, handle, "kernel_bicg");
    uint64_t id = veo_call_async(ctx, kernel_bicg, argp);
    uint64_t retval;
    veo_call_wait_result(ctx, id, &retval);

    veo_free_mem(proc, A_ptr);
    veo_free_mem(proc, s_ptr);
    veo_free_mem(proc, p_ptr);
    veo_free_mem(proc, r_ptr);
    veo_free_mem(proc, q_ptr);
    veo_args_free(argp);
    free(A);
    free(p);
    free(r);
    veo_context_close(ctx);
    veo_proc_destroy(proc);
    long finish = get_time();
    printf("runtime %ld ms\n", finish - begin);
    return 0;
}