#include <chrono>
#include <iostream>
#include <stdio.h>
#include <ve_offload.h>
#include <vector>

using namespace std;

typedef struct {
    float x, y, z;
} Atom;

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

    vector<Atom> input(size);
    for (size_t i = 0; i < size; i++) {
        Atom temp = {(float)i, (float)i, (float)i};
        input[i] = temp;
    }
    vector<int> neighbour(size);
    for (size_t i = 0; i < size; i++) {
        neighbour[i] = (i + 1) % size;
    }

    auto data_start = chrono::steady_clock::now();
    uint64_t input_ptr;
    long problem_bytes = size * sizeof(Atom);
    veo_alloc_mem(proc, &input_ptr, problem_bytes);
    veo_write_mem(proc, input_ptr, (void*)input.data(), problem_bytes);
    uint64_t output_ptr;
    veo_alloc_mem(proc, &output_ptr, problem_bytes);
    uint64_t neighbour_ptr;
    veo_alloc_mem(proc, &neighbour_ptr, size * sizeof(int));
    veo_write_mem(proc, neighbour_ptr, (void*)neighbour.data(), size * sizeof(int));
    auto data_end = chrono::steady_clock::now();

    struct veo_args* argp = veo_args_alloc();
    veo_args_set_i64(argp, 0, input_ptr);
    veo_args_set_i64(argp, 1, output_ptr);
    veo_args_set_i64(argp, 2, neighbour_ptr);
    veo_args_set_i64(argp, 3, size);

    auto kernel_start = chrono::steady_clock::now();
    uint64_t kernel_dyn = veo_get_sym(proc, handle, "kernel_dyn");
    uint64_t id = veo_call_async(ctx, kernel_dyn, argp);
    uint64_t retval;
    veo_call_wait_result(ctx, id, &retval);
    auto kernel_end = chrono::steady_clock::now();

    veo_free_mem(proc, input_ptr);
    veo_free_mem(proc, output_ptr);
    veo_free_mem(proc, neighbour_ptr);
    veo_args_free(argp);
    veo_context_close(ctx);
    veo_proc_destroy(proc);

    auto end = chrono::steady_clock::now();
    cout << "mod_dyn size=" << size << " runtime(" << chrono::duration_cast<chrono::milliseconds>(end - start).count() << ") data(" << chrono::duration_cast<chrono::milliseconds>(data_end - data_start).count() << ") kernel(" << chrono::duration_cast<chrono::milliseconds>(kernel_end - kernel_start).count() << ")" << endl;
    return 0;
}
