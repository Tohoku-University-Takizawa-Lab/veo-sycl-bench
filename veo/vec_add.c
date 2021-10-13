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
    long problem_bytes = size * sizeof(DATA_TYPE);
    DATA_TYPE* input1 = (DATA_TYPE*)malloc(problem_bytes);
    DATA_TYPE* input2 = (DATA_TYPE*)malloc(problem_bytes);
    DATA_TYPE* output = (DATA_TYPE*)malloc(problem_bytes);
    for (size_t i = 0; i < size; i++) {
        input1[i] = (DATA_TYPE)i;
        input2[i] = (DATA_TYPE)i;
        output[i] = 0;
    }

    uint64_t input1_ptr;
    veo_alloc_mem(proc, &input1_ptr, problem_bytes);
    veo_write_mem(proc, input1_ptr, (void*)input1, problem_bytes);
    uint64_t input2_ptr;
    veo_alloc_mem(proc, &input2_ptr, problem_bytes);
    veo_write_mem(proc, input2_ptr, (void*)input2, problem_bytes);
    uint64_t output_ptr;
    veo_alloc_mem(proc, &output_ptr, problem_bytes);
    veo_write_mem(proc, output_ptr, (void*)output, problem_bytes);

    struct veo_args* argp = veo_args_alloc();
    veo_args_set_i64(argp, 0, input1_ptr);
    veo_args_set_i64(argp, 1, input2_ptr);
    veo_args_set_i64(argp, 2, output_ptr);
    veo_args_set_i64(argp, 3, size);

    uint64_t kernel_add = veo_get_sym(proc, handle, "kernel_add");
    uint64_t id = veo_call_async(ctx, kernel_add, argp);
    uint64_t retval;
    veo_call_wait_result(ctx, id, &retval);

    veo_free_mem(proc, input1_ptr);
    veo_free_mem(proc, input2_ptr);
    veo_free_mem(proc, output_ptr);
    veo_args_free(argp);
    free(input1);
    free(input2);
    free(output);
    veo_context_close(ctx);
    veo_proc_destroy(proc);
    long finish = get_time();
    printf("runtime %ld ms\n", finish - begin);
    return 0;
}