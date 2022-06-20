#include <chrono>
#include <iostream>
#include <stdio.h>
#include <ve_offload.h>
#include <vector>

using namespace std;
int TMAX = 500;

void init_arrays(vector<float>& fict, vector<float>& ex, vector<float>& ey, vector<float>& hz, size_t size) {
    const size_t NX = size;
    const size_t NY = size;
    for (size_t i = 0; i < TMAX; i++) {
        fict[i] = (float)i;
    }
    for (size_t i = 0; i < NX; i++) {
        for (size_t j = 0; j < NY; j++) {
            ex[i * NY + j] = ((float)i * (j + 1) + 1) / NX;
            ey[i * NY + j] = ((float)(i - 1) * (j + 2) + 2) / NX;
            hz[i * NY + j] = ((float)(i - 9) * (j + 4) + 3) / NX;
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

    vector<float> fict(TMAX);
    vector<float> ex(size * (size + 1));
    vector<float> ey(size * (size + 1));
    vector<float> hz(size * size);
    init_arrays(fict, ex, ey, hz, size);

    auto data_start = chrono::steady_clock::now();
    uint64_t fict_ptr;
    veo_alloc_mem(proc, &fict_ptr, TMAX * sizeof(float));
    veo_write_mem(proc, fict_ptr, (void*)fict.data(), TMAX * sizeof(float));
    uint64_t ex_ptr;
    veo_alloc_mem(proc, &ex_ptr, (size + 1) * size * sizeof(float));
    veo_write_mem(proc, ex_ptr, (void*)ex.data(), (size + 1) * size * sizeof(float));
    uint64_t ey_ptr;
    veo_alloc_mem(proc, &ey_ptr, (size + 1) * size * sizeof(float));
    veo_write_mem(proc, ey_ptr, (void*)ey.data(), (size + 1) * size * sizeof(float));
    uint64_t hz_ptr;
    veo_alloc_mem(proc, &hz_ptr, size * size * sizeof(float));
    veo_write_mem(proc, hz_ptr, (void*)hz.data(), size * size * sizeof(float));
    auto data_end = chrono::steady_clock::now();

    struct veo_args* argp = veo_args_alloc();
    veo_args_set_i64(argp, 0, fict_ptr);
    veo_args_set_i64(argp, 1, ex_ptr);
    veo_args_set_i64(argp, 2, ey_ptr);
    veo_args_set_i64(argp, 3, hz_ptr);
    veo_args_set_i64(argp, 4, size);

    auto kernel_start = chrono::steady_clock::now();
    uint64_t kernel_Fdtd = veo_get_sym(proc, handle, "kernel_Fdtd");
    uint64_t id = veo_call_async(ctx, kernel_Fdtd, argp);
    uint64_t retval;
    veo_call_wait_result(ctx, id, &retval);
    auto kernel_end = chrono::steady_clock::now();

    veo_free_mem(proc, fict_ptr);
    veo_free_mem(proc, ex_ptr);
    veo_free_mem(proc, ey_ptr);
    veo_free_mem(proc, hz_ptr);
    veo_args_free(argp);
    veo_context_close(ctx);
    veo_proc_destroy(proc);

    auto end = chrono::steady_clock::now();
    cout << "fdtd2d size=" << size << " runtime(" << chrono::duration_cast<chrono::milliseconds>(end - start).count() << ") data(" << chrono::duration_cast<chrono::milliseconds>(data_end - data_start).count() << ") kernel(" << chrono::duration_cast<chrono::milliseconds>(kernel_end - kernel_start).count() << ")" << endl;
    return 0;
}