#include "common.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <ve_offload.h>

#define SOFTENING 1e-9f

typedef struct {
    float x, y, z, vx, vy, vz;
} Body;

void randomizeBodies(float* data, int n) {
    for (int i = 0; i < n; i++) {
        data[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
    }
}

int main(const int argc, const char** argv) {
    long begin = get_time();
    struct veo_proc_handle* proc = veo_proc_create(0);
    uint64_t handle = veo_load_library(proc, "./kernel.so");
    if (handle == 0) {
        perror("[VEO]:veo failed to load the shared library\n");
        return 1;
    }
    struct veo_thr_ctxt* ctx = veo_context_open(proc);
    int nBodies = 5;
    if (argc > 1)
        nBodies = atoi(argv[1]);

    const float dt = 0.01f; // time step
    int bytes = nBodies * sizeof(Body);
    float* buf = (float*)malloc(bytes);
    Body* p = (Body*)buf;
    randomizeBodies(buf, 6 * nBodies); // Init pos / vel data

    uint64_t body_ptr;
    veo_alloc_mem(proc, &body_ptr, bytes);
    veo_write_mem(proc, body_ptr, (void*)buf, bytes);

    struct veo_args* args = veo_args_alloc();
    veo_args_set_i64(args, 0, body_ptr);
    veo_args_set_float(args, 1, dt);
    veo_args_set_i64(args, 2, nBodies);

    uint64_t id = veo_call_async_by_name(ctx, handle, "kernel_body", args);
    uint64_t retval;
    veo_call_wait_result(ctx, id, &retval);
    veo_read_mem(proc, (void*)buf, body_ptr, bytes);
    // printf("vh p[2].x=%f\n", p[2].x);

    veo_free_mem(proc, body_ptr);
    veo_args_free(args);
    free(buf);
    veo_context_close(ctx);
    veo_proc_destroy(proc);
    long finish = get_time();
    printf("runtime %ld ms\n", finish - begin);
    return 0;
}