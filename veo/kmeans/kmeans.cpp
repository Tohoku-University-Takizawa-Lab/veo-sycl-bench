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

    int nfeatures = 2;
    int nclusters = 3;
    vector<float> features(nfeatures * size);
    vector<float> clusters(nclusters * size);
    for (int i = 0; i < nfeatures * size; ++i) {
        features[i] = 2.0f;
    }
    for (int i = 0; i < nclusters * size; ++i) {
        clusters[i] = 1.0f;
    }

    auto data_start = chrono::steady_clock::now();
    long feature_size = nfeatures * size * sizeof(float);
    long cluster_size = nclusters * size * sizeof(float);
    uint64_t features_ptr;
    veo_alloc_mem(proc, &features_ptr, feature_size);
    veo_write_mem(proc, features_ptr, (void*)features.data(), feature_size);
    uint64_t clusters_ptr;
    veo_alloc_mem(proc, &clusters_ptr, cluster_size);
    veo_write_mem(proc, clusters_ptr, (void*)clusters.data(), cluster_size);
    uint64_t membership_ptr;
    veo_alloc_mem(proc, &membership_ptr, size * sizeof(int));
    auto data_end = chrono::steady_clock::now();

    struct veo_args* argp = veo_args_alloc();
    veo_args_set_i64(argp, 0, features_ptr);
    veo_args_set_i64(argp, 1, clusters_ptr);
    veo_args_set_i64(argp, 2, membership_ptr);
    veo_args_set_i64(argp, 3, size);

    auto kernel_start = chrono::steady_clock::now();
    uint64_t kernel_kmeans = veo_get_sym(proc, handle, "kernel_kmeans");
    uint64_t id = veo_call_async(ctx, kernel_kmeans, argp);
    uint64_t retval;
    veo_call_wait_result(ctx, id, &retval);
    auto kernel_end = chrono::steady_clock::now();

    veo_free_mem(proc, features_ptr);
    veo_free_mem(proc, clusters_ptr);
    veo_free_mem(proc, membership_ptr);
    veo_args_free(argp);

    veo_context_close(ctx);
    veo_proc_destroy(proc);
    auto end = chrono::steady_clock::now();
    cout << "kmeans size=" << size << " runtime(" << chrono::duration_cast<chrono::milliseconds>(end - start).count() << ") data(" << chrono::duration_cast<chrono::milliseconds>(data_end - data_start).count() << ") kernel(" << chrono::duration_cast<chrono::milliseconds>(kernel_end - kernel_start).count() << ")" << endl;
    return 0;
}
