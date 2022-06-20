#include <chrono>
#include <iostream>
#include <stdio.h>
#include <ve_offload.h>
#include <vector>

using namespace std;

void init_arrays(vector<float> a, vector<float> x1, vector<float> x2, vector<float> y_1, vector<float> y_2, size_t size) {
    const size_t N = size;
    for (size_t i = 0; i < N; i++) {
        x1[i] = 0.0;
        x2[i] = 0.0;
        y_1[i] = 0.0;
        y_2[i] = 0.0;
        for (size_t j = 0; j < N; j++) {
            a[i * N + j] = (float)(i + j + 1.0) / N;
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

    vector<float> a(size * size);
    vector<float> x1(size);
    vector<float> x2(size);
    vector<float> y1(size);
    vector<float> y2(size);
    init_arrays(a, x1, x2, y1, y2, size);

    auto data_start = chrono::steady_clock::now();
    uint64_t a_ptr;
    veo_alloc_mem(proc, &a_ptr, size * size * sizeof(float));
    veo_write_mem(proc, a_ptr, (void*)a.data(), size * size * sizeof(float));
    uint64_t x1_ptr;
    veo_alloc_mem(proc, &x1_ptr, size * sizeof(float));
    veo_write_mem(proc, x1_ptr, (void*)x1.data(), size * sizeof(float));
    uint64_t x2_ptr;
    veo_alloc_mem(proc, &x2_ptr, size * sizeof(float));
    veo_write_mem(proc, x2_ptr, (void*)x2.data(), size * sizeof(float));
    uint64_t y1_ptr;
    veo_alloc_mem(proc, &y1_ptr, size * sizeof(float));
    veo_write_mem(proc, y1_ptr, (void*)y1.data(), size * sizeof(float));
    uint64_t y2_ptr;
    veo_alloc_mem(proc, &y2_ptr, size * sizeof(float));
    veo_write_mem(proc, y2_ptr, (void*)y2.data(), size * sizeof(float));
    auto data_end = chrono::steady_clock::now();

    struct veo_args* argp = veo_args_alloc();
    veo_args_set_i64(argp, 0, a_ptr);
    veo_args_set_i64(argp, 1, x1_ptr);
    veo_args_set_i64(argp, 2, x2_ptr);
    veo_args_set_i64(argp, 3, y1_ptr);
    veo_args_set_i64(argp, 4, y2_ptr);
    veo_args_set_i64(argp, 5, size);

    auto kernel_start = chrono::steady_clock::now();
    uint64_t kernel_mvt = veo_get_sym(proc, handle, "kernel_mvt");
    uint64_t id = veo_call_async(ctx, kernel_mvt, argp);
    uint64_t retval;
    veo_call_wait_result(ctx, id, &retval);
    auto kernel_end = chrono::steady_clock::now();

    veo_free_mem(proc, a_ptr);
    veo_free_mem(proc, x1_ptr);
    veo_free_mem(proc, x2_ptr);
    veo_free_mem(proc, y1_ptr);
    veo_free_mem(proc, y2_ptr);
    veo_args_free(argp);
    veo_context_close(ctx);
    veo_proc_destroy(proc);

    auto end = chrono::steady_clock::now();
    cout << "mvt size=" << size << " runtime(" << chrono::duration_cast<chrono::milliseconds>(end - start).count() << ") data(" << chrono::duration_cast<chrono::milliseconds>(data_end - data_start).count() << ") kernel(" << chrono::duration_cast<chrono::milliseconds>(kernel_end - kernel_start).count() << ")" << endl;
    return 0;
}