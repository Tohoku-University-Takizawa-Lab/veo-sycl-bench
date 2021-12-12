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

    vector<float> input1(size, 1.0);
    vector<float> input2(size, 2.0);
    float* sumv1 = (float*)malloc(sizeof(float));
    float* sumv2 = (float*)malloc(sizeof(float));
    float* xy = (float*)malloc(sizeof(float));
    float* xx = (float*)malloc(sizeof(float));
    *sumv1 = 0;
    *sumv2 = 0;
    *xy = 0;
    *xx = 0;

    auto data_start = chrono::steady_clock::now();
    long problem_bytes = size * sizeof(float);
    uint64_t input1_ptr;
    veo_alloc_mem(proc, &input1_ptr, problem_bytes);
    veo_write_mem(proc, input1_ptr, (void*)input1.data(), problem_bytes);
    uint64_t input2_ptr;
    veo_alloc_mem(proc, &input2_ptr, problem_bytes);
    veo_write_mem(proc, input2_ptr, (void*)input2.data(), problem_bytes);
    uint64_t sumv1_ptr;
    veo_alloc_mem(proc, &sumv1_ptr, sizeof(float));
    veo_write_mem(proc, sumv1_ptr, (void*)sumv1, sizeof(float));
    uint64_t sumv2_ptr;
    veo_alloc_mem(proc, &sumv2_ptr, sizeof(float));
    veo_write_mem(proc, sumv2_ptr, (void*)sumv2, sizeof(float));
    uint64_t xy_ptr;
    veo_alloc_mem(proc, &xy_ptr, sizeof(float));
    veo_write_mem(proc, xy_ptr, (void*)xy, sizeof(float));
    uint64_t xx_ptr;
    veo_alloc_mem(proc, &xx_ptr, sizeof(float));
    veo_write_mem(proc, xx_ptr, (void*)xx, sizeof(float));
    auto data_end = chrono::steady_clock::now();

    struct veo_args* argp = veo_args_alloc();
    veo_args_set_i64(argp, 0, input1_ptr);
    veo_args_set_i64(argp, 1, input2_ptr);
    veo_args_set_i64(argp, 2, sumv1_ptr);
    veo_args_set_i64(argp, 3, sumv2_ptr);
    veo_args_set_i64(argp, 4, xy_ptr);
    veo_args_set_i64(argp, 5, xx_ptr);
    veo_args_set_i64(argp, 6, size);

    auto kernel_start = chrono::steady_clock::now();
    uint64_t kernel_coeff = veo_get_sym(proc, handle, "kernel_coeff");
    uint64_t id = veo_call_async(ctx, kernel_coeff, argp);
    uint64_t retval;
    veo_call_wait_result(ctx, id, &retval);
    auto kernel_end = chrono::steady_clock::now();

    veo_read_mem(proc, (void*)sumv1, sumv1_ptr, sizeof(float));
    veo_read_mem(proc, (void*)sumv2, sumv2_ptr, sizeof(float));
    veo_read_mem(proc, (void*)xy, xy_ptr, sizeof(float));
    veo_read_mem(proc, (void*)xx, xx_ptr, sizeof(float));
    // printf("vh result:sumv1=%f sumv2=%f xy=%f xx=%f\n", *sumv1, *sumv2, *xy, *xx);

    veo_free_mem(proc, input1_ptr);
    veo_free_mem(proc, input2_ptr);
    veo_free_mem(proc, sumv1_ptr);
    veo_free_mem(proc, sumv2_ptr);
    veo_free_mem(proc, xy_ptr);
    veo_free_mem(proc, xx_ptr);
    veo_args_free(argp);
    free(sumv1);
    free(sumv2);
    free(xy);
    free(xx);
    veo_context_close(ctx);
    veo_proc_destroy(proc);

    auto end = chrono::steady_clock::now();
    cout << "lin_reg_coeff size=" << size << " runtime(" << chrono::duration_cast<chrono::milliseconds>(end - start).count() << ") data(" << chrono::duration_cast<chrono::milliseconds>(data_end - data_start).count() << ") kernel(" << chrono::duration_cast<chrono::milliseconds>(kernel_end - kernel_start).count() << ")" << endl;
    return 0;
}