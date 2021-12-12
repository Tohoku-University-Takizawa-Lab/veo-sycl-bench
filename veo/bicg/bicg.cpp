#include <chrono>
#include <iostream>
#include <stdio.h>
#include <ve_offload.h>
#include <vector>

using namespace std;

#ifndef M_PI
#define M_PI 3.14159
#endif

void init_array(vector<float> A, vector<float> p, vector<float> r, size_t size) {
    const size_t NX = size;
    const size_t NY = size;
    for (size_t i = 0; i < NX; i++) {
        r[i] = i * M_PI;
        for (size_t j = 0; j < NY; j++) {
            A[i * NY + j] = ((float)i * j) / NX;
        }
    }
    for (size_t i = 0; i < NY; i++) {
        p[i] = i * M_PI;
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

    int problem_bytes = size * size * sizeof(float);
    vector<float> A(size * size);
    vector<float> p(size);
    vector<float> r(size);
    init_array(A, p, r, size);

    auto data_start = chrono::steady_clock::now();
    uint64_t A_ptr;
    veo_alloc_mem(proc, &A_ptr, problem_bytes);
    veo_write_mem(proc, A_ptr, (void*)A.data(), problem_bytes);
    uint64_t p_ptr;
    veo_alloc_mem(proc, &p_ptr, size * sizeof(float));
    veo_write_mem(proc, p_ptr, (void*)p.data(), size * sizeof(float));
    uint64_t r_ptr;
    veo_alloc_mem(proc, &r_ptr, size * sizeof(float));
    veo_write_mem(proc, r_ptr, (void*)r.data(), size * sizeof(float));
    uint64_t s_ptr;
    veo_alloc_mem(proc, &s_ptr, size * sizeof(float));
    uint64_t q_ptr;
    veo_alloc_mem(proc, &q_ptr, size * sizeof(float));
    auto data_end = chrono::steady_clock::now();

    struct veo_args* argp = veo_args_alloc();
    veo_args_set_i64(argp, 0, A_ptr);
    veo_args_set_i64(argp, 1, r_ptr);
    veo_args_set_i64(argp, 2, s_ptr);
    veo_args_set_i64(argp, 3, p_ptr);
    veo_args_set_i64(argp, 4, q_ptr);
    veo_args_set_i64(argp, 5, size);

    auto kernel_start = chrono::steady_clock::now();
    uint64_t kernel_bicg = veo_get_sym(proc, handle, "kernel_bicg");
    uint64_t id = veo_call_async(ctx, kernel_bicg, argp);
    uint64_t retval;
    veo_call_wait_result(ctx, id, &retval);
    auto kernel_end = chrono::steady_clock::now();

    veo_free_mem(proc, A_ptr);
    veo_free_mem(proc, s_ptr);
    veo_free_mem(proc, p_ptr);
    veo_free_mem(proc, r_ptr);
    veo_free_mem(proc, q_ptr);
    veo_args_free(argp);
    veo_context_close(ctx);
    veo_proc_destroy(proc);

    auto end = chrono::steady_clock::now();
    cout << "bicg size=" << size << " runtime(" << chrono::duration_cast<chrono::milliseconds>(end - start).count() << ") data(" << chrono::duration_cast<chrono::milliseconds>(data_end - data_start).count() << ") kernel(" << chrono::duration_cast<chrono::milliseconds>(kernel_end - kernel_start).count() << ")" << endl;
    return 0;
}