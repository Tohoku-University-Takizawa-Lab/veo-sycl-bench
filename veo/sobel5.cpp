#include "bitmap.h"
#include "common.h"
#include <cmath>
#include <string>
#include <ve_offload.h>
#include <vector>

using namespace std;
int main(int argc, char** argv) {
    long begin = get_time();
    size_t size = 5;
    if (argc > 1)
        size = atoi(argv[1]);

    long totalTime = 0;
    struct veo_proc_handle* proc = veo_proc_create(0);
    uint64_t handle = veo_load_library(proc, "./kernel.so");
    if (handle == 0) {
        perror("[VEO]:veo failed to load the shared library\n");
        return 0;
    }
    struct veo_thr_ctxt* ctx = veo_context_open(proc);
    vector<Dot> input;
    input.resize(size * size);
    load_bitmap_mirrored("../Brommy.bmp", size, input);
    uint64_t input_ptr;
    veo_alloc_mem(proc, &input_ptr, size * size * sizeof(Dot));
    veo_write_mem(proc, input_ptr, (void*)input.data(), size * size * sizeof(Dot));
    uint64_t output_ptr;
    veo_alloc_mem(proc, &output_ptr, size * size * sizeof(Dot));

    struct veo_args* argp = veo_args_alloc();
    veo_args_set_i64(argp, 0, input_ptr);
    veo_args_set_i64(argp, 1, output_ptr);
    veo_args_set_i64(argp, 2, size);
    uint64_t kernel_sobel5 = veo_get_sym(proc, handle, "kernel_sobel5");
    uint64_t id = veo_call_async(ctx, kernel_sobel5, argp);
    uint64_t retval;
    veo_call_wait_result(ctx, id, &retval);

    veo_free_mem(proc, input_ptr);
    veo_free_mem(proc, output_ptr);
    veo_args_free(argp);
    veo_context_close(ctx);
    veo_proc_destroy(proc);
    long finish = get_time();
    printf("runtime %ld ms\n", finish - begin);
    return 0;
}
