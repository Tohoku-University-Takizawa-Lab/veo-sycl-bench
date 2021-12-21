#include <stdio.h>
#include <stdlib.h>

#define FLT_MAX 500000.0
void P6kmeans(float* features, float* clusters, int* membership,
                   int* n) {
    // printf("kmeans size=%d\n", *n);
    int nfeatures = 2;
    int nclusters = 3;
    int size = *n;
    for (int gid = 0; gid < size; ++gid) {
        int index = 0;
        double min_dist = FLT_MAX;
        for (int i = 0; i < nclusters; i++) {
            double dist = 0;
            for (int l = 0; l < nfeatures; l++) {
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
