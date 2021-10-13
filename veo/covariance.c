#include "common.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <ve_offload.h>
void init_arrays(DATA_TYPE* data, size_t size) {
    const size_t M = size;
    const size_t N = size;

    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            data[i * (N + 1) + j] = ((DATA_TYPE)i * j) / M;
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
    DATA_TYPE* data = (DATA_TYPE*)malloc((size + 1) * (size + 1) * sizeof(DATA_TYPE));
    init_arrays(data, size);
    uint64_t data_ptr;
    veo_alloc_mem(proc, &data_ptr, (size + 1) * (size + 1) * sizeof(DATA_TYPE));
    veo_write_mem(proc, data_ptr, (void*)data, (size + 1) * (size + 1) * sizeof(DATA_TYPE));
    uint64_t mean_ptr;
    veo_alloc_mem(proc, &mean_ptr, (size + 1) * sizeof(DATA_TYPE));
    uint64_t symmat_ptr;
    veo_alloc_mem(proc, &symmat_ptr, (size + 1) * (size + 1) * sizeof(DATA_TYPE));
    struct veo_args* argp = veo_args_alloc();
    veo_args_set_i64(argp, 0, data_ptr);
    veo_args_set_i64(argp, 1, symmat_ptr);
    veo_args_set_i64(argp, 2, mean_ptr);
    veo_args_set_i64(argp, 3, size);

    uint64_t covariance = veo_get_sym(proc, handle, "covariance");
    uint64_t id = veo_call_async(ctx, covariance, argp);
    uint64_t retval;
    veo_call_wait_result(ctx, id, &retval);

    veo_free_mem(proc, data_ptr);
    veo_free_mem(proc, mean_ptr);
    veo_free_mem(proc, symmat_ptr);
    veo_args_free(argp);
    free(data);
    veo_context_close(ctx);
    veo_proc_destroy(proc);
    long finish = get_time();
    printf("runtime %ld ms\n", finish - begin);
    return 0;
}