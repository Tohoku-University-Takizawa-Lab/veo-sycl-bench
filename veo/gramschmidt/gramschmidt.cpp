#include <chrono>
#include <iostream>
#include <stdio.h>
#include <ve_offload.h>
#include <vector>

using namespace std;

void init_array(vector<float>& A, size_t size) {
    const size_t M = 0;
    const size_t N = 0;
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            A[i * N + j] = ((float)(i + 1) * (j + 1)) / (M + 1);
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
    init_array(A, size);

    auto data_start = chrono::steady_clock::now();
    int problem_bytes = size * size * sizeof(float);
    uint64_t A_ptr;
    veo_alloc_mem(proc, &A_ptr, problem_bytes);
    veo_write_mem(proc, A_ptr, (void*)A.data(), problem_bytes);
    uint64_t R_ptr;
    veo_alloc_mem(proc, &R_ptr, problem_bytes);
    uint64_t Q_ptr;
    veo_alloc_mem(proc, &Q_ptr, problem_bytes);
    auto data_end = chrono::steady_clock::now();

    struct veo_args* argp = veo_args_alloc();
    veo_args_set_i64(argp, 0, A_ptr);
    veo_args_set_i64(argp, 1, R_ptr);
    veo_args_set_i64(argp, 2, Q_ptr);
    veo_args_set_i64(argp, 3, size);

    auto kernel_start = chrono::steady_clock::now();
    uint64_t gramschmidt = veo_get_sym(proc, handle, "gramschmidt");
    uint64_t id = veo_call_async(ctx, gramschmidt, argp);
    uint64_t retval;
    veo_call_wait_result(ctx, id, &retval);
    auto kernel_end = chrono::steady_clock::now();

    veo_free_mem(proc, A_ptr);
    veo_free_mem(proc, R_ptr);
    veo_free_mem(proc, Q_ptr);
    veo_args_free(argp);
    veo_context_close(ctx);
    veo_proc_destroy(proc);

    auto end = chrono::steady_clock::now();
    cout << "gramschmidt size=" << size << " runtime(" << chrono::duration_cast<chrono::milliseconds>(end - start).count() << ") data(" << chrono::duration_cast<chrono::milliseconds>(data_end - data_start).count() << ") kernel(" << chrono::duration_cast<chrono::milliseconds>(kernel_end - kernel_start).count() << ")" << endl;
    return 0;
}