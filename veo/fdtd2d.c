#include "common.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <ve_offload.h>

int TMAX = 500;
void init_arrays(DATA_TYPE* fict, DATA_TYPE* ex, DATA_TYPE* ey, DATA_TYPE* hz, size_t size) {
    const size_t NX = size;
    const size_t NY = size;

    for (size_t i = 0; i < TMAX; i++) {
        fict[i] = (DATA_TYPE)i;
    }

    for (size_t i = 0; i < NX; i++) {
        for (size_t j = 0; j < NY; j++) {
            ex[i * NY + j] = ((DATA_TYPE)i * (j + 1) + 1) / NX;
            ey[i * NY + j] = ((DATA_TYPE)(i - 1) * (j + 2) + 2) / NX;
            hz[i * NY + j] = ((DATA_TYPE)(i - 9) * (j + 4) + 3) / NX;
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
    DATA_TYPE* fict = (DATA_TYPE*)malloc(TMAX * sizeof(DATA_TYPE));
    DATA_TYPE* ex = (DATA_TYPE*)malloc((size + 1) * size * sizeof(DATA_TYPE));
    DATA_TYPE* ey = (DATA_TYPE*)malloc((size + 1) * size * sizeof(DATA_TYPE));
    DATA_TYPE* hz = (DATA_TYPE*)malloc(size * size * sizeof(DATA_TYPE));
    init_arrays(fict, ex, ey, hz, size);

    uint64_t fict_ptr;
    veo_alloc_mem(proc, &fict_ptr, TMAX * sizeof(DATA_TYPE));
    veo_write_mem(proc, fict_ptr, (void*)fict, TMAX * sizeof(DATA_TYPE));
    uint64_t ex_ptr;
    veo_alloc_mem(proc, &ex_ptr, (size + 1) * size * sizeof(DATA_TYPE));
    veo_write_mem(proc, ex_ptr, (void*)ex, (size + 1) * size * sizeof(DATA_TYPE));
    uint64_t ey_ptr;
    veo_alloc_mem(proc, &ey_ptr, (size + 1) * size * sizeof(DATA_TYPE));
    veo_write_mem(proc, ey_ptr, (void*)ey, (size + 1) * size * sizeof(DATA_TYPE));
    uint64_t hz_ptr;
    veo_alloc_mem(proc, &hz_ptr, size * size * sizeof(DATA_TYPE));
    veo_write_mem(proc, hz_ptr, (void*)hz, size * size * sizeof(DATA_TYPE));

    struct veo_args* argp = veo_args_alloc();
    veo_args_set_i64(argp, 0, fict_ptr);
    veo_args_set_i64(argp, 1, ex_ptr);
    veo_args_set_i64(argp, 2, ey_ptr);
    veo_args_set_i64(argp, 3, hz_ptr);
    veo_args_set_i64(argp, 4, size);

    uint64_t kernel_Fdtd = veo_get_sym(proc, handle, "kernel_Fdtd");
    uint64_t id = veo_call_async(ctx, kernel_Fdtd, argp);
    uint64_t retval;
    veo_call_wait_result(ctx, id, &retval);

    veo_free_mem(proc, fict_ptr);
    veo_free_mem(proc, ex_ptr);
    veo_free_mem(proc, ey_ptr);
    veo_free_mem(proc, hz_ptr);
    veo_args_free(argp);
    free(fict);
    free(ex);
    free(ey);
    free(hz);
    veo_context_close(ctx);
    veo_proc_destroy(proc);
    long finish = get_time();
    printf("runtime %ld ms\n", finish - begin);
    return 0;
}