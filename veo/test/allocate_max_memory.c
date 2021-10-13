#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <stdlib.h>
#include <ve_offload.h>

int main(int argc, char **argv){
    int size = 10;
    if (argc > 1)
        size = atoi(argv[1]);
    struct veo_proc_handle *proc = veo_proc_create(0);
    uint64_t handle = veo_load_library(proc, "./kernel.so");

    struct veo_thr_ctxt *ctx = veo_context_open(proc);
    uint64_t A_ptr;
    veo_alloc_mem(proc, &A_ptr, size*sizeof(double));
    veo_free_mem(proc, A_ptr);
    veo_context_close(ctx);
    veo_proc_destroy(proc);
    return 0;
}