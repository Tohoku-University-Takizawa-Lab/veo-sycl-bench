#include "common.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <ve_offload.h>

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

    long problem_bytes = size * sizeof(float);
    float* input1 = (float*)malloc(problem_bytes);
    float* input2 = (float*)malloc(problem_bytes);
    for (size_t i = 0; i < size; i++) {
        input1[i] = 1.0;
        input2[i] = 2.0;
    }
    float* sumv1 = (float*)malloc(sizeof(float));
    float* sumv2 = (float*)malloc(sizeof(float));
    float* xy = (float*)malloc(sizeof(float));
    float* xx = (float*)malloc(sizeof(float));
    *sumv1 = 0;
    *sumv2 = 0;
    *xy = 0;
    *xx = 0;
    uint64_t input1_ptr;
    veo_alloc_mem(proc, &input1_ptr, problem_bytes);
    veo_write_mem(proc, input1_ptr, (void*)input1, problem_bytes);
    uint64_t input2_ptr;
    veo_alloc_mem(proc, &input2_ptr, problem_bytes);
    veo_write_mem(proc, input2_ptr, (void*)input2, problem_bytes);
    uint64_t sumv1_ptr;
    veo_alloc_mem(proc, &sumv1_ptr, sizeof(float));
    veo_write_mem(proc, sumv1_ptr, (void*)sumv1, sizeof(float));
    uint64_t sumv2_ptr;
    veo_alloc_mem(proc, &sumv2_ptr, sizeof(float));
    veo_write_mem(proc, sumv2_ptr, (void*)sumv2, sizeof(float));
    uint64_t xy_ptr;
    veo_alloc_mem(proc, &xy_ptr, sizeof(float));
    veo_write_mem(proc, xy_ptr, (void*)xy, sizeof(float));
    uint64_t xx_ptr;
    veo_alloc_mem(proc, &xx_ptr, sizeof(float));
    veo_write_mem(proc, xx_ptr, (void*)xx, sizeof(float));

    struct veo_args* argp = veo_args_alloc();
    veo_args_set_i64(argp, 0, input1_ptr);
    veo_args_set_i64(argp, 1, input2_ptr);
    veo_args_set_i64(argp, 2, sumv1_ptr);
    veo_args_set_i64(argp, 3, sumv2_ptr);
    veo_args_set_i64(argp, 4, xy_ptr);
    veo_args_set_i64(argp, 5, xx_ptr);
    veo_args_set_i64(argp, 6, size);

    uint64_t kernel_coeff = veo_get_sym(proc, handle, "kernel_coeff");
    uint64_t id = veo_call_async(ctx, kernel_coeff, argp);
    uint64_t retval;
    veo_call_wait_result(ctx, id, &retval);
    veo_read_mem(proc, (void*)sumv1, sumv1_ptr, sizeof(float));
    veo_read_mem(proc, (void*)sumv2, sumv2_ptr, sizeof(float));
    veo_read_mem(proc, (void*)xy, xy_ptr, sizeof(float));
    veo_read_mem(proc, (void*)xx, xx_ptr, sizeof(float));
    // printf("vh result:sumv1=%f sumv2=%f xy=%f xx=%f\n", *sumv1, *sumv2, *xy, *xx);

    veo_free_mem(proc, input1_ptr);
    veo_free_mem(proc, input2_ptr);
    veo_free_mem(proc, sumv1_ptr);
    veo_free_mem(proc, sumv2_ptr);
    veo_free_mem(proc, xy_ptr);
    veo_free_mem(proc, xx_ptr);
    veo_args_free(argp);
    free(input1);
    free(input2);
    free(sumv1);
    free(sumv2);
    free(xy);
    free(xx);

    veo_context_close(ctx);
    veo_proc_destroy(proc);
    long finish = get_time();
    printf("runtime %ld ms\n", finish - begin);
    return 0;
}