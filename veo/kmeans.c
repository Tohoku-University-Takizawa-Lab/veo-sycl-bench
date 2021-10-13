#include "common.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <ve_offload.h>

int main(int argc, char** argv) {
    long begin = get_time();
    size_t size = 5;
    if (argc > 1)
        size = atoi(argv[1]);

    struct veo_proc_handle* proc = veo_proc_create(0);
    uint64_t handle = veo_load_library(proc, "./kernel.so");
    if (handle == 0) {
        perror("[VEO]:veo failed to load the shared library\n");
        return 0;
    }
    struct veo_thr_ctxt* ctx = veo_context_open(proc);
    int nfeatures = 2;
    int nclusters = 3;
    long feature_size = nfeatures * size * sizeof(float);
    long cluster_size = nclusters * size * sizeof(float);
    float* features = (float*)malloc(feature_size);
    float* clusters = (float*)malloc(cluster_size);
    for (int i = 0; i < nfeatures * size; ++i) {
        features[i] = 2.0f;
    }
    for (int i = 0; i < nclusters * size; ++i) {
        clusters[i] = 1.0f;
    }
    uint64_t features_ptr;
    veo_alloc_mem(proc, &features_ptr, feature_size);
    veo_write_mem(proc, features_ptr, (void*)features, feature_size);
    uint64_t clusters_ptr;
    veo_alloc_mem(proc, &clusters_ptr, cluster_size);
    veo_write_mem(proc, clusters_ptr, (void*)clusters, cluster_size);
    uint64_t membership_ptr;
    veo_alloc_mem(proc, &membership_ptr, size * sizeof(int));

    struct veo_args* argp = veo_args_alloc();
    veo_args_set_i64(argp, 0, features_ptr);
    veo_args_set_i64(argp, 1, clusters_ptr);
    veo_args_set_i64(argp, 2, membership_ptr);
    veo_args_set_i64(argp, 3, size);

    uint64_t kernel_kmeans = veo_get_sym(proc, handle, "kernel_kmeans");
    uint64_t id = veo_call_async(ctx, kernel_kmeans, argp);
    uint64_t retval;
    veo_call_wait_result(ctx, id, &retval);

    veo_free_mem(proc, features_ptr);
    veo_free_mem(proc, clusters_ptr);
    veo_free_mem(proc, membership_ptr);
    free(features);
    free(clusters);
    veo_context_close(ctx);
    veo_proc_destroy(proc);
    long finish = get_time();
    printf("runtime %ld ms\n", finish - begin);
    return 0;
}