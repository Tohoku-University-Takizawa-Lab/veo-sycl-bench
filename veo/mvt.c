#include "common.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <ve_offload.h>

void init_arrays(DATA_TYPE* a, DATA_TYPE* x1, DATA_TYPE* x2, DATA_TYPE* y_1, DATA_TYPE* y_2, size_t size) {
    const size_t N = size;

    for (size_t i = 0; i < N; i++) {
        x1[i] = 0.0;
        x2[i] = 0.0;
        y_1[i] = 0.0;
        y_2[i] = 0.0;

        for (size_t j = 0; j < N; j++) {
            a[i * N + j] = (DATA_TYPE)(i + j + 1.0) / N;
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
    DATA_TYPE* a = (DATA_TYPE*)malloc(size * size * sizeof(DATA_TYPE));
    DATA_TYPE* x1 = (DATA_TYPE*)malloc(size * sizeof(DATA_TYPE));
    DATA_TYPE* x2 = (DATA_TYPE*)malloc(size * sizeof(DATA_TYPE));
    DATA_TYPE* y1 = (DATA_TYPE*)malloc(size * sizeof(DATA_TYPE));
    DATA_TYPE* y2 = (DATA_TYPE*)malloc(size * sizeof(DATA_TYPE));
    init_arrays(a, x1, x2, y1, y2, size);

    uint64_t a_ptr;
    veo_alloc_mem(proc, &a_ptr, size * size * sizeof(DATA_TYPE));
    veo_write_mem(proc, a_ptr, (void*)a, size * size * sizeof(DATA_TYPE));
    uint64_t x1_ptr;
    veo_alloc_mem(proc, &x1_ptr, size * sizeof(DATA_TYPE));
    veo_write_mem(proc, x1_ptr, (void*)x1, size * sizeof(DATA_TYPE));
    uint64_t x2_ptr;
    veo_alloc_mem(proc, &x2_ptr, size * sizeof(DATA_TYPE));
    veo_write_mem(proc, x2_ptr, (void*)x2, size * sizeof(DATA_TYPE));
    uint64_t y1_ptr;
    veo_alloc_mem(proc, &y1_ptr, size * sizeof(DATA_TYPE));
    veo_write_mem(proc, y1_ptr, (void*)y1, size * sizeof(DATA_TYPE));
    uint64_t y2_ptr;
    veo_alloc_mem(proc, &y2_ptr, size * sizeof(DATA_TYPE));
    veo_write_mem(proc, y2_ptr, (void*)y2, size * sizeof(DATA_TYPE));

    struct veo_args* argp = veo_args_alloc();
    veo_args_set_i64(argp, 0, a_ptr);
    veo_args_set_i64(argp, 1, x1_ptr);
    veo_args_set_i64(argp, 2, x2_ptr);
    veo_args_set_i64(argp, 3, y1_ptr);
    veo_args_set_i64(argp, 4, y2_ptr);
    veo_args_set_i64(argp, 5, size);

    uint64_t kernel_mvt = veo_get_sym(proc, handle, "kernel_mvt");
    uint64_t id = veo_call_async(ctx, kernel_mvt, argp);
    uint64_t retval;
    veo_call_wait_result(ctx, id, &retval);

    veo_free_mem(proc, a_ptr);
    veo_free_mem(proc, x1_ptr);
    veo_free_mem(proc, x2_ptr);
    veo_free_mem(proc, y1_ptr);
    veo_free_mem(proc, y2_ptr);
    veo_args_free(argp);
    free(a);
    free(x1);
    free(x2);
    free(y1);
    free(y2);
    veo_context_close(ctx);
    veo_proc_destroy(proc);
    long finish = get_time();
    printf("runtime %ld ms\n", finish - begin);
    return 0;
}