#include <chrono>
#include <iostream>
#include <stdio.h>
#include <ve_offload.h>
#include <vector>

using namespace std;

#define FLOAT_N 3214212.01
#define EPS 0.005

void init_arrays(vector<float> data, size_t size) {
    const size_t M = size;
    const size_t N = size;
    for (size_t i = 0; i <= M; i++) {
        for (size_t j = 0; j <= N; j++) {
            data[i * N + j] = ((float)i * j) / (M + 1);
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

    vector<float> data((size + 1) * (size + 1));
    init_arrays(data, size);

    auto data_start = chrono::steady_clock::now();
    uint64_t data_ptr;
    veo_alloc_mem(proc, &data_ptr, (size + 1) * (size + 1) * sizeof(float));
    veo_write_mem(proc, data_ptr, (void*)data.data(), (size + 1) * (size + 1) * sizeof(float));
    uint64_t mean_ptr;
    veo_alloc_mem(proc, &mean_ptr, (size + 1) * sizeof(float));
    uint64_t stddev_ptr;
    veo_alloc_mem(proc, &stddev_ptr, (size + 1) * sizeof(float));
    uint64_t symmat_ptr;
    veo_alloc_mem(proc, &symmat_ptr, (size + 1) * (size + 1) * sizeof(float));
    auto data_end = chrono::steady_clock::now();

    struct veo_args* argp = veo_args_alloc();
    veo_args_set_i64(argp, 0, data_ptr);
    veo_args_set_i64(argp, 1, mean_ptr);
    veo_args_set_i64(argp, 2, stddev_ptr);
    veo_args_set_i64(argp, 3, symmat_ptr);
    veo_args_set_i64(argp, 4, size);

    auto kernel_start = chrono::steady_clock::now();
    uint64_t correlation = veo_get_sym(proc, handle, "correlation");
    uint64_t id = veo_call_async(ctx, correlation, argp);
    uint64_t retval;
    veo_call_wait_result(ctx, id, &retval);
    auto kernel_end = chrono::steady_clock::now();

    veo_free_mem(proc, data_ptr);
    veo_free_mem(proc, mean_ptr);
    veo_free_mem(proc, stddev_ptr);
    veo_free_mem(proc, symmat_ptr);
    veo_args_free(argp);
    veo_context_close(ctx);
    veo_proc_destroy(proc);
    auto end = chrono::steady_clock::now();
    cout << "correlation size=" << size << " runtime(" << chrono::duration_cast<chrono::milliseconds>(end - start).count() << ") data(" << chrono::duration_cast<chrono::milliseconds>(data_end - data_start).count() << ") kernel(" << chrono::duration_cast<chrono::milliseconds>(kernel_end - kernel_start).count() << ")" << endl;
    return 0;
}