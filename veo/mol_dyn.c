#include "common.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <ve_offload.h>

typedef struct {
    float x, y, z;
} Atom;

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
    long problem_bytes = size * sizeof(Atom);
    float* buf = (float*)malloc(problem_bytes);
    Atom* input = (Atom*)buf;
    int* neighbour = (int*)malloc(size * sizeof(int));
    for (size_t i = 0; i < size; i++) {
        Atom temp = {(float)i, (float)i, (float)i};
        input[i] = temp;
    }
    for (size_t i = 0; i < size; i++) {
        neighbour[i] = i + 1;
    }
    uint64_t input_ptr;
    veo_alloc_mem(proc, &input_ptr, problem_bytes);
    veo_write_mem(proc, input_ptr, (void*)buf, problem_bytes);
    uint64_t output_ptr;
    veo_alloc_mem(proc, &output_ptr, problem_bytes);
    uint64_t neighbour_ptr;
    veo_alloc_mem(proc, &neighbour_ptr, size * sizeof(int));
    veo_write_mem(proc, neighbour_ptr, (void*)neighbour, size * sizeof(int));

    struct veo_args* argp = veo_args_alloc();
    veo_args_set_i64(argp, 0, input_ptr);
    veo_args_set_i64(argp, 1, output_ptr);
    veo_args_set_i64(argp, 2, neighbour_ptr);
    veo_args_set_i64(argp, 3, size);

    uint64_t kernel_dyn = veo_get_sym(proc, handle, "kernel_dyn");
    uint64_t id = veo_call_async(ctx, kernel_dyn, argp);
    uint64_t retval;
    veo_call_wait_result(ctx, id, &retval);

    veo_free_mem(proc, input_ptr);
    veo_free_mem(proc, output_ptr);
    veo_free_mem(proc, neighbour_ptr);
    veo_args_free(argp);
    free(input);
    free(neighbour);
    veo_context_close(ctx);
    veo_proc_destroy(proc);
    long finish = get_time();
    printf("runtime %ld ms\n", finish - begin);
    return 0;
}