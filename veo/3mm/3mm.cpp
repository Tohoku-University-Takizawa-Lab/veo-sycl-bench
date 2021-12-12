#include <chrono>
#include <iostream>
#include <stdio.h>
#include <ve_offload.h>
#include <vector>

using namespace std;

void init_array(vector<float> A, vector<float> B, vector<float> C, vector<float> D, size_t size) {
    const size_t NI = size;
    const size_t NJ = size;
    const size_t NK = size;
    const size_t NL = size;
    const size_t NM = size;
    for (size_t i = 0; i < NI; i++) {
        for (size_t j = 0; j < NK; j++) {
            A[i * NK + j] = ((float)i * j) / NI;
        }
    }
    for (size_t i = 0; i < NK; i++) {
        for (size_t j = 0; j < NJ; j++) {
            B[i * NJ + j] = ((float)i * (j + 1)) / NJ;
        }
    }
    for (size_t i = 0; i < NJ; i++) {
        for (size_t j = 0; j < NM; j++) {
            C[i * NM + j] = ((float)i * (j + 3)) / NL;
        }
    }
    for (size_t i = 0; i < NM; i++) {
        for (size_t j = 0; j < NL; j++) {
            D[i * NL + j] = ((float)i * (j + 2)) / NK;
        }
    }
}

int main(int argc, char** argv) {
    auto start = chrono::steady_clock::now();
    size_t size = 5;
    if (argc > 1)
        size = stoi(argv[1]);

    struct veo_proc_handle* proc = veo_proc_create(0);
    uint64_t handle = veo_load_library(proc, "./kernel.so");
    if (handle == 0) {
        perror("[VEO]:veo failed to load the shared library\n");
        return 0;
    }
    struct veo_thr_ctxt* ctx = veo_context_open(proc);

    vector<float> A(size * size);
    vector<float> B(size * size);
    vector<float> C(size * size);
    vector<float> D(size * size);
    init_array(A, B, C, D, size);

    auto data_start = chrono::steady_clock::now();
    int problem_bytes = size * size * sizeof(float);
    uint64_t A_ptr;
    uint64_t B_ptr;
    uint64_t C_ptr;
    uint64_t D_ptr;
    veo_alloc_mem(proc, &A_ptr, problem_bytes);
    veo_write_mem(proc, A_ptr, (void*)A.data(), problem_bytes);
    veo_alloc_mem(proc, &B_ptr, problem_bytes);
    veo_write_mem(proc, B_ptr, (void*)B.data(), problem_bytes);
    veo_alloc_mem(proc, &C_ptr, problem_bytes);
    veo_write_mem(proc, C_ptr, (void*)C.data(), problem_bytes);
    veo_alloc_mem(proc, &D_ptr, problem_bytes);
    veo_write_mem(proc, D_ptr, (void*)D.data(), problem_bytes);
    uint64_t E_ptr;
    uint64_t F_ptr;
    uint64_t G_ptr;
    veo_alloc_mem(proc, &E_ptr, problem_bytes);
    veo_alloc_mem(proc, &F_ptr, problem_bytes);
    veo_alloc_mem(proc, &G_ptr, problem_bytes);
    auto data_end = chrono::steady_clock::now();

    struct veo_args* argp = veo_args_alloc();
    veo_args_set_i64(argp, 0, A_ptr);
    veo_args_set_i64(argp, 1, B_ptr);
    veo_args_set_i64(argp, 2, C_ptr);
    veo_args_set_i64(argp, 3, D_ptr);
    veo_args_set_i64(argp, 4, E_ptr);
    veo_args_set_i64(argp, 5, F_ptr);
    veo_args_set_i64(argp, 6, G_ptr);
    veo_args_set_i64(argp, 7, size);

    auto kernel_start = chrono::steady_clock::now();
    uint64_t kernel_mm3 = veo_get_sym(proc, handle, "kernel_mm3");
    uint64_t id = veo_call_async(ctx, kernel_mm3, argp);
    uint64_t retval;
    veo_call_wait_result(ctx, id, &retval);
    auto kernel_end = chrono::steady_clock::now();

    veo_free_mem(proc, A_ptr);
    veo_free_mem(proc, B_ptr);
    veo_free_mem(proc, C_ptr);
    veo_free_mem(proc, D_ptr);
    veo_free_mem(proc, E_ptr);
    veo_args_free(argp);
    veo_context_close(ctx);
    veo_proc_destroy(proc);

    auto end = chrono::steady_clock::now();
    cout << "3mm size=" << size << " runtime(" << chrono::duration_cast<chrono::milliseconds>(end - start).count() << ") data(" << chrono::duration_cast<chrono::milliseconds>(data_end - data_start).count() << ") kernel(" << chrono::duration_cast<chrono::milliseconds>(kernel_end - kernel_start).count() << ")" << endl;
    return 0;
}