#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <ve_offload.h>

#include "common.h"

/* Array initialization. */
void init_array(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* C, DATA_TYPE* D, size_t size) {
    const size_t NI = size;
    const size_t NJ = size;
    const size_t NK = size;
    const size_t NL = size;
    for (size_t i = 0; i < NI; i++) {
        for (size_t j = 0; j < NK; j++) {
            A[i * NI + j] = ((DATA_TYPE)i * j) / NI;
        }
    }
    for (size_t i = 0; i < NK; i++) {
        for (size_t j = 0; j < NJ; j++) {
            B[i * NK + j] = ((DATA_TYPE)i * (j + 1)) / NJ;
        }
    }
    for (size_t i = 0; i < NL; i++) {
        for (size_t j = 0; j < NJ; j++) {
            C[i * NL + j] = ((DATA_TYPE)i * (j + 3)) / NL;
        }
    }
    for (size_t i = 0; i < NI; i++) {
        for (size_t j = 0; j < NL; j++) {
            D[i * NL + j] = ((DATA_TYPE)i * (j + 2)) / NK;
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
    DATA_TYPE* C = (DATA_TYPE*)malloc(problem_bytes);
    DATA_TYPE* D = (DATA_TYPE*)malloc(problem_bytes);
    init_array(A, B, C, D, size);

    uint64_t A_ptr;
    uint64_t B_ptr;
    uint64_t C_ptr;
    uint64_t D_ptr;
    veo_alloc_mem(proc, &A_ptr, problem_bytes);
    veo_write_mem(proc, A_ptr, (void*)A, problem_bytes);
    veo_alloc_mem(proc, &B_ptr, problem_bytes);
    veo_write_mem(proc, B_ptr, (void*)B, problem_bytes);
    veo_alloc_mem(proc, &C_ptr, problem_bytes);
    veo_write_mem(proc, C_ptr, (void*)C, problem_bytes);
    veo_alloc_mem(proc, &D_ptr, problem_bytes);
    veo_write_mem(proc, D_ptr, (void*)D, problem_bytes);

    uint64_t E_ptr;
    veo_alloc_mem(proc, &E_ptr, problem_bytes);

    struct veo_args* argp = veo_args_alloc();
    veo_args_set_i64(argp, 0, A_ptr);
    veo_args_set_i64(argp, 1, B_ptr);
    veo_args_set_i64(argp, 2, C_ptr);
    veo_args_set_i64(argp, 3, D_ptr);
    veo_args_set_i64(argp, 4, E_ptr);
    veo_args_set_i64(argp, 5, size);
    uint64_t kernel_mm2 = veo_get_sym(proc, handle, "kernel_mm2");
    uint64_t id = veo_call_async(ctx, kernel_mm2, argp);
    uint64_t retval;
    veo_call_wait_result(ctx, id, &retval);

    /*verify results*/
    DATA_TYPE* E = (DATA_TYPE*)malloc(problem_bytes);
    // const size_t NI = size;
    // const size_t NJ = size;
    // const size_t NK = size;
    // const size_t NL = size;
    veo_read_mem(proc, (void*)E, E_ptr, problem_bytes);
    // printf("VH results: \n");
    // for(size_t i = 0; i < NI; i++) {
    //     for(size_t j = 0; j < NL; j++) {
    //         for(size_t k = 0; k < NJ; ++k) {
    //             printf("%f ", E[i * NL + j]);
    //         }
    //     }
    // }
    // printf("\n");

    veo_free_mem(proc, A_ptr);
    veo_free_mem(proc, B_ptr);
    veo_free_mem(proc, C_ptr);
    veo_free_mem(proc, D_ptr);
    veo_free_mem(proc, E_ptr);
    veo_args_free(argp);
    free(A);
    free(B);
    free(C);
    free(D);
    free(E);
    veo_context_close(ctx);
    veo_proc_destroy(proc);
    long finish = get_time();
    printf("runtime %ld ms\n", finish - begin);
    return 0;
}