#include <chrono>
#include <iostream>
#include <stdio.h>
#include <ve_offload.h>
#include <vector>

using namespace std;

void init(vector<float> A, size_t size) {
    const size_t NI = size;
    const size_t NJ = size;
    for (size_t i = 0; i < NI; ++i) {
        for (size_t j = 0; j < NJ; ++j) {
            A[i * NJ + j] = (float)rand() / (float)RAND_MAX;
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

    auto init_start = chrono::steady_clock::now();
    vector<float> A(size * size);
    init(A, size);
    vector<float> B(size * size);
    auto init_end = chrono::steady_clock::now();

    long problem_bytes = size * size * sizeof(float);
    auto data1_start = chrono::steady_clock::now();
    uint64_t A_ptr;
    uint64_t ret = veo_alloc_mem(proc, &A_ptr, problem_bytes);
    ret = veo_write_mem(proc, A_ptr, (void*)A.data(), problem_bytes);
    uint64_t B_ptr;
    veo_alloc_mem(proc, &B_ptr, problem_bytes);
    auto data1_end = chrono::steady_clock::now();

    struct veo_args* argp = veo_args_alloc();
    veo_args_set_i64(argp, 0, A_ptr);
    veo_args_set_i64(argp, 1, B_ptr);
    veo_args_set_i64(argp, 2, size);
    uint64_t conv2D = veo_get_sym(proc, handle, "conv2D");

    auto kernel_start = chrono::steady_clock::now();
    uint64_t id = veo_call_async(ctx, conv2D, argp);
    uint64_t retval;
    veo_call_wait_result(ctx, id, &retval);
    auto kernel_end = chrono::steady_clock::now();

    auto data2_start = chrono::steady_clock::now();
    veo_read_mem(proc, (void*)B.data(), B_ptr, problem_bytes);
    auto data2_end = chrono::steady_clock::now();

    veo_free_mem(proc, A_ptr);
    veo_free_mem(proc, B_ptr);
    veo_args_free(argp);

    veo_context_close(ctx);
    veo_proc_destroy(proc);

    auto end = chrono::steady_clock::now();
    // cout << "2DConvolution size=" << size << " runtime(" << chrono::duration_cast<chrono::milliseconds>(end - start).count() << ") data(" << chrono::duration_cast<chrono::milliseconds>(data_end - data_start).count() << ") kernel(" << chrono::duration_cast<chrono::milliseconds>(kernel_end - kernel_start).count() << ")" << endl;
    cout << "2DConvolution size=" << size << " runtime("
         << chrono::duration_cast<chrono::microseconds>(end - start).count() << ") data("
         << chrono::duration_cast<chrono::microseconds>(data1_end - data1_start).count() +
                chrono::duration_cast<chrono::microseconds>(data2_end - data2_start).count()
         << ") kernel(" << chrono::duration_cast<chrono::microseconds>(kernel_end - kernel_start).count()
         << ") init(" << chrono::duration_cast<chrono::microseconds>(init_end - init_start).count() << ")"
         << endl;
    return 0;
}
