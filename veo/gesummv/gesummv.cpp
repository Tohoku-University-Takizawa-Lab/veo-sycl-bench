#include <chrono>
#include <iostream>
#include <stdio.h>
#include <ve_offload.h>
#include <vector>

using namespace std;

void init(vector<float>& A, vector<float>& B, vector<float>& x, size_t size) {
    const size_t N = size;
    for (size_t i = 0; i < N; i++) {
        x[i] = 1;
        for (size_t j = 0; j < N; j++) {
            A[i * N + j] = 2;
            B[i * N + j] = 3;
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

    int problem_bytes = size * size * sizeof(float);
    vector<float> A(size * size);
    vector<float> B(size * size);
    vector<float> x(size);
    init(A, B, x, size);

    auto data_start = chrono::steady_clock::now();
    uint64_t A_ptr;
    veo_alloc_mem(proc, &A_ptr, problem_bytes);
    veo_write_mem(proc, A_ptr, (void*)A.data(), problem_bytes);
    uint64_t B_ptr;
    veo_alloc_mem(proc, &B_ptr, problem_bytes);
    veo_write_mem(proc, B_ptr, (void*)B.data(), problem_bytes);
    uint64_t x_ptr;
    veo_alloc_mem(proc, &x_ptr, size * sizeof(float));
    veo_write_mem(proc, x_ptr, (void*)x.data(), size * sizeof(float));
    uint64_t y_ptr;
    veo_alloc_mem(proc, &y_ptr, size * sizeof(float));
    uint64_t tmp_ptr;
    veo_alloc_mem(proc, &tmp_ptr, size * sizeof(float));
    auto data_end = chrono::steady_clock::now();

    struct veo_args* argp = veo_args_alloc();
    veo_args_set_i64(argp, 0, A_ptr);
    veo_args_set_i64(argp, 1, B_ptr);
    veo_args_set_i64(argp, 2, x_ptr);
    veo_args_set_i64(argp, 3, y_ptr);
    veo_args_set_i64(argp, 4, tmp_ptr);
    veo_args_set_i64(argp, 5, size);

    auto kernel_start = chrono::steady_clock::now();
    uint64_t kernel_gesummv = veo_get_sym(proc, handle, "kernel_gesummv");
    uint64_t id = veo_call_async(ctx, kernel_gesummv, argp);
    uint64_t retval;
    veo_call_wait_result(ctx, id, &retval);
    auto kernel_end = chrono::steady_clock::now();

    veo_free_mem(proc, A_ptr);
    veo_free_mem(proc, B_ptr);
    veo_free_mem(proc, x_ptr);
    veo_free_mem(proc, y_ptr);
    veo_free_mem(proc, tmp_ptr);
    veo_args_free(argp);
    veo_context_close(ctx);
    veo_proc_destroy(proc);
    auto end = chrono::steady_clock::now();
    cout << "gesummv size=" << size << " runtime(" << chrono::duration_cast<chrono::milliseconds>(end - start).count() << ") data(" << chrono::duration_cast<chrono::milliseconds>(data_end - data_start).count() << ") kernel(" << chrono::duration_cast<chrono::milliseconds>(kernel_end - kernel_start).count() << ")" << endl;
    return 0;
}