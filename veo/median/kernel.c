#include <stdio.h>
#include <stdlib.h>

typedef struct {
    float x;
    float y;
    float z;
} Dot;
void swap(Dot A[], int i, int j) {
    if (A[i].x > A[j].x) {
        float temp = A[i].x;
        A[i].x = A[j].x;
        A[j].x = temp;
    }
}
void kernel_median(Dot* input, Dot* output, size_t size) {
    // printf("median size=%d\n", size);
    for (size_t i = 0; i < size; i++) {
        int x = i % size;
        int y = i / size;
        Dot window[9];
        int k = 0;
        for (int i = -1; i < 2; i++) {
            for (int j = -1; j < 2; j++) {
                unsigned int xs = fmin(fmax(x + j, 0), size - 1); // borders are handled here with extended values
                unsigned int ys = fmin(fmax(y + i, 0), size - 1);
                window[k] = input[xs + ys * size];
                k++;
            }
            swap(window, 0, 1);
            swap(window, 2, 3);
            swap(window, 0, 2);
            swap(window, 1, 3);
            swap(window, 1, 2);
            swap(window, 4, 5);
            swap(window, 7, 8);
            swap(window, 6, 8);
            swap(window, 6, 7);
            swap(window, 4, 7);
            swap(window, 4, 6);
            swap(window, 5, 8);
            swap(window, 5, 7);
            swap(window, 5, 6);
            swap(window, 0, 5);
            swap(window, 0, 4);
            swap(window, 1, 6);
            swap(window, 1, 5);
            swap(window, 1, 4);
            swap(window, 2, 7);
            swap(window, 3, 8);
            swap(window, 3, 7);
            swap(window, 2, 5);
            swap(window, 2, 4);
            swap(window, 3, 6);
            swap(window, 3, 5);
            swap(window, 3, 4);
        }
        output[i] = window[4];
    }
}
