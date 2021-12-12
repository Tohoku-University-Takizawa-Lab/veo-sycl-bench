#include <stdio.h>
#include <stdlib.h>

#define FLT_MAX 500000.0
void kernel_kmeans(float* features, float* clusters, int* membership,
                   size_t size) {
    // printf("kmeans size=%d\n", size);
    int nfeatures = 2;
    int nclusters = 3;
    for (size_t gid = 0; gid < size; ++gid) {
        int index = 0;
        double min_dist = FLT_MAX;
        for (size_t i = 0; i < nclusters; i++) {
            double dist = 0;
            for (size_t l = 0; l < nfeatures; l++) {
                dist +=
                    (features[l * size + gid] - clusters[i * nfeatures + l]) *
                    (features[l * size + gid] - clusters[i * nfeatures + l]);
            }
            if (dist < min_dist) {
                min_dist = dist;
                index = gid;
            }
        }
        membership[gid] = index;
    }
}
