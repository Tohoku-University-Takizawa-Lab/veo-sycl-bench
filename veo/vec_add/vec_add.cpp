#include <chrono>
#include <iostream>
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
    
    vector<int> input1(size);
    vector<int> input2(size);
    vector<int> output(size, 0);
    for (size_t i = 0; i < size; i++) {
        input1[i] = i;
        input2[i] = i;
    }

    auto data_start = chrono::steady_clock::now();
    long problem_bytes = size * sizeof(int);
    uint64_t input1_ptr;
    veo_alloc_mem(proc, &input1_ptr, problem_bytes);
    veo_write_mem(proc, input1_ptr, (void*)input1.data(), problem_bytes);
    uint64_t input2_ptr;
    veo_alloc_mem(proc, &input2_ptr, problem_bytes);
    veo_write_mem(proc, input2_ptr, (void*)input2.data(), problem_bytes);
    uint64_t output_ptr;
    veo_alloc_mem(proc, &output_ptr, problem_bytes);
    veo_write_mem(proc, output_ptr, (void*)output.data(), problem_bytes);
    auto data_end = chrono::steady_clock::now();

    int num = 10;
    uint64_t num_ptr;
    veo_alloc_mem(proc, &num_ptr, 4);
    veo_write_mem(proc, num_ptr, (void*)(&num), 4);

    struct veo_args* argp = veo_args_alloc();
    veo_args_set_i64(argp, 0, input1_ptr);
    veo_args_set_i64(argp, 1, input2_ptr);
    veo_args_set_i64(argp, 2, output_ptr);
    veo_args_set_i64(argp, 3, size);
    veo_args_set_i64(argp, 4, size);

    auto kernel_start = chrono::steady_clock::now();
    uint64_t kernel_add = veo_get_sym(proc, handle, "kernel_add");
    uint64_t id = veo_call_async(ctx, kernel_add, argp);
    uint64_t retval;
    veo_call_wait_result(ctx, id, &retval);
    auto kernel_end = chrono::steady_clock::now();

    veo_free_mem(proc, input1_ptr);
    veo_free_mem(proc, input2_ptr);
    veo_free_mem(proc, output_ptr);
    veo_args_free(argp);
    veo_context_close(ctx);
    veo_proc_destroy(proc);
    auto end = chrono::steady_clock::now();
    cout << "vec_add size=" << size << " runtime(" << chrono::duration_cast<chrono::milliseconds>(end - start).count() << ") data(" << chrono::duration_cast<chrono::milliseconds>(data_end - data_start).count() << ") kernel(" << chrono::duration_cast<chrono::milliseconds>(kernel_end - kernel_start).count() << ")" << endl;
    return 0;
}