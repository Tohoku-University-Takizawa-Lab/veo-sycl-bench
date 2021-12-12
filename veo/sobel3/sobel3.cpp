#include "bitmap.h"
#include <chrono>
#include <iostream>
#include <stdio.h>
#include <ve_offload.h>
#include <vector>

using namespace std;

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

    vector<Dot> input(size * size);
    load_bitmap_mirrored("../../Brommy.bmp", size, input);

    auto data_start = chrono::steady_clock::now();
    uint64_t input_ptr;
    veo_alloc_mem(proc, &input_ptr, size * size * sizeof(Dot));
    veo_write_mem(proc, input_ptr, (void*)input.data(), size * size * sizeof(Dot));
    uint64_t output_ptr;
    veo_alloc_mem(proc, &output_ptr, size * size * sizeof(Dot));
    auto data_end = chrono::steady_clock::now();

    struct veo_args* argp = veo_args_alloc();
    veo_args_set_i64(argp, 0, input_ptr);
    veo_args_set_i64(argp, 1, output_ptr);
    veo_args_set_i64(argp, 2, size);

    auto kernel_start = chrono::steady_clock::now();
    uint64_t kernel_sobel = veo_get_sym(proc, handle, "kernel_sobel3");
    uint64_t id = veo_call_async(ctx, kernel_sobel, argp);
    uint64_t retval;
    veo_call_wait_result(ctx, id, &retval);
    auto kernel_end = chrono::steady_clock::now();

    veo_free_mem(proc, input_ptr);
    veo_free_mem(proc, output_ptr);
    veo_args_free(argp);
    veo_context_close(ctx);
    veo_proc_destroy(proc);
    auto end = chrono::steady_clock::now();
    cout << "sobel3 size=" << size << " runtime(" << chrono::duration_cast<chrono::milliseconds>(end - start).count() << ") data(" << chrono::duration_cast<chrono::milliseconds>(data_end - data_start).count() << ") kernel(" << chrono::duration_cast<chrono::milliseconds>(kernel_end - kernel_start).count() << ")" << endl;
    return 0;
}
