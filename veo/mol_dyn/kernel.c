#include <stdio.h>
#include <stdlib.h>

typedef struct {
    float x, y, z;
} Atom;

void kernel_dyn(Atom* input, Atom* output, int* neighbour, size_t size) {
    // printf("mol_dyn size=%d\n", size);
    int neighCount = 15;
    int cutsq = 50;
    int lj1 = 20;
    float lj2 = 0.003f;
    int inum = 0;
    for (int i = 0; i < size; ++i) {
        Atom ipos = input[i];
        Atom f = {0.0f, 0.0f, 0.0f};
        int j = 0;
        while (j < neighCount) {
            int jidx = neighbour[j * inum + i];
            Atom jpos = input[jidx];
            // Calculate distance
            float delx = ipos.x - jpos.x;
            float dely = ipos.y - jpos.y;
            float delz = ipos.z - jpos.z;
            float r2inv = delx * delx + dely * dely + delz * delz;
            // If distance is less than cutoff, calculate force
            if (r2inv < cutsq) {
                r2inv = 10.0f / r2inv;
                float r6inv = r2inv * r2inv * r2inv;
                float forceC = r2inv * r6inv * (lj1 * r6inv - lj2);
                f.x += delx * forceC;
                f.y += dely * forceC;
                f.z += delz * forceC;
            }
            j++;
        }
        output[i] = f;
    }
}
