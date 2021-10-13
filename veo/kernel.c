#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define DATA_TYPE float

void kernel_mm2(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* C, DATA_TYPE* D, DATA_TYPE* E, size_t size) {
    // printf("mm2_cpu size=%d\n", size);
    const size_t NI = size;
    const size_t NJ = size;
    const size_t NK = size;
    const size_t NL = size;

    for (size_t i = 0; i < NI; i++) {
        for (size_t j = 0; j < NJ; j++) {
            for (size_t k = 0; k < NK; ++k) {
                C[i * NJ + j] += A[i * NK + k] * B[k * NJ + j];
            }
        }
    }
    for (size_t i = 0; i < NI; i++) {
        for (size_t j = 0; j < NL; j++) {
            E[i * NL + j] = 0;
            for (size_t k = 0; k < NJ; ++k) {
                E[i * NL + j] += C[i * NJ + k] * D[k * NL + j];
                // printf("%f ", E[i * NL + j]);
            }
        }
    }
    // printf("\n");
}

void kernel_mm3(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* C, DATA_TYPE* D, DATA_TYPE* E, DATA_TYPE* F, DATA_TYPE* G, size_t size) {
    const size_t NI = size;
    const size_t NJ = size;
    const size_t NK = size;
    const size_t NL = size;
    const size_t NM = size;
    /* E := A*B */
    for (size_t i = 0; i < NI; i++) {
        for (size_t j = 0; j < NJ; j++) {
            E[i * NJ + j] = 0;
            for (size_t k = 0; k < NK; ++k) {
                E[i * NJ + j] += A[i * NK + k] * B[k * NJ + j];
            }
        }
    }
    /* F := C*D */
    for (size_t i = 0; i < NI; i++) {
        for (size_t j = 0; j < NL; j++) {
            F[i * NL + j] = 0;
            for (size_t k = 0; k < NM; ++k) {
                F[i * NL + j] += C[i * NM + k] * D[k * NL + j];
            }
        }
    }
    /* G := E*F */
    for (size_t i = 0; i < NI; i++) {
        for (size_t j = 0; j < NL; j++) {
            G[i * NL + j] = 0;
            for (size_t k = 0; k < NJ; ++k) {
                G[i * NL + j] += E[i * NJ + k] * F[k * NL + j];
            }
        }
    }
}

void conv2D(DATA_TYPE* A, DATA_TYPE* B, size_t size) {
    // printf("conv2D size=%d\n", size);
    const size_t NI = size;
    const size_t NJ = size;

    const DATA_TYPE c11 = +0.2, c21 = +0.5, c31 = -0.8;
    const DATA_TYPE c12 = -0.3, c22 = +0.6, c32 = -0.9;
    const DATA_TYPE c13 = +0.4, c23 = +0.7, c33 = +0.10;

    for (size_t i = 1; i < NI - 1; ++i) {
        for (size_t j = 1; j < NJ - 1; ++j) {
            B[i * NJ + j] = c11 * A[(i - 1) * NJ + (j - 1)] + c12 * A[(i + 0) * NJ + (j - 1)] + c13 * A[(i + 1) * NJ + (j - 1)] + c21 * A[(i - 1) * NJ + (j + 0)] + c22 * A[(i + 0) * NJ + (j + 0)] + c23 * A[(i + 1) * NJ + (j + 0)] + c31 * A[(i - 1) * NJ + (j + 1)] + c32 * A[(i + 0) * NJ + (j + 1)] + c33 * A[(i + 1) * NJ + (j + 1)];
        }
    }
}

void conv3D(DATA_TYPE* A, DATA_TYPE* B, size_t size) {
    // printf("conv3D size=%d\n", size);
    const size_t NI = size;
    const size_t NJ = size;
    const size_t NK = size;

    const DATA_TYPE c11 = +2, c21 = +5, c31 = -8;
    const DATA_TYPE c12 = -3, c22 = +6, c32 = -9;
    const DATA_TYPE c13 = +4, c23 = +7, c33 = +10;

    for (size_t i = 1; i < NI - 1; ++i) {
        for (size_t j = 1; j < NJ - 1; ++j) {
            for (size_t k = 1; k < NK - 1; ++k) {
                B[i * (NK * NJ) + j * NK + k] = c11 * A[(i - 1) * (NK * NJ) + (j - 1) * NK + (k - 1)] + c13 * A[(i + 1) * (NK * NJ) + (j - 1) * NK + (k - 1)] + c21 * A[(i - 1) * (NK * NJ) + (j - 1) * NK + (k - 1)] + c23 * A[(i + 1) * (NK * NJ) + (j - 1) * NK + (k - 1)] + c31 * A[(i - 1) * (NK * NJ) + (j - 1) * NK + (k - 1)] + c33 * A[(i + 1) * (NK * NJ) + (j - 1) * NK + (k - 1)] + c12 * A[(i + 0) * (NK * NJ) + (j - 1) * NK + (k + 0)] + c22 * A[(i + 0) * (NK * NJ) + (j + 0) * NK + (k + 0)] + c32 * A[(i + 0) * (NK * NJ) + (j + 1) * NK + (k + 0)] + c11 * A[(i - 1) * (NK * NJ) + (j - 1) * NK + (k + 1)] + c13 * A[(i + 1) * (NK * NJ) + (j - 1) * NK + (k + 1)] + c21 * A[(i - 1) * (NK * NJ) + (j + 0) * NK + (k + 1)] + c23 * A[(i + 1) * (NK * NJ) + (j + 0) * NK + (k + 1)] + c31 * A[(i - 1) * (NK * NJ) + (j + 1) * NK + (k + 1)] + c33 * A[(i + 1) * (NK * NJ) + (j + 1) * NK + (k + 1)];
            }
        }
    }
}

void kernel_atax(DATA_TYPE* A, DATA_TYPE* x, DATA_TYPE* y, DATA_TYPE* tmp, size_t size) {
    // printf("atax size=%d\n", size);
    const size_t NX = size;
    const size_t NY = size;

    for (size_t i = 0; i < NX; i++) {
        for (size_t j = 0; j < NY; j++) {
            tmp[i] += A[i * NY + j] * x[j];
        }

        for (size_t j = 0; j < NY; j++) {
            y[j] += A[i * NY + j] * tmp[i];
        }
    }
}

void kernel_bicg(DATA_TYPE* A, DATA_TYPE* r, DATA_TYPE* s, DATA_TYPE* p, DATA_TYPE* q, size_t size) {
    // printf("bicg size=%d\n", size);
    const size_t NX = size;
    const size_t NY = size;

    for (size_t i = 0; i < NX; i++) {
        for (size_t j = 0; j < NY; j++) {
            s[j] += r[i] * A[i * NY + j];
            q[i] += A[i * NY + j] * p[j];
        }
    }
}

#define FLOAT_N 3214212.01
#define EPS 0.005
#define sqrt_of_array_cell(x, j) sqrt(x[j])

void correlation(DATA_TYPE* data, DATA_TYPE* mean, DATA_TYPE* stddev, DATA_TYPE* symmat, size_t size) {
    // printf("correlation size=%d\n", size);
    const size_t M = size;
    const size_t N = size;

    // Determine mean of column vectors of input data matrix
    for (size_t j = 1; j <= M; j++) {
        mean[j] = 0.0;

        for (size_t i = 1; i <= N; i++) {
            mean[j] += data[i * (M + 1) + j];
        }

        mean[j] /= (DATA_TYPE)FLOAT_N;
    }

    // Determine standard deviations of column vectors of data matrix.
    for (size_t j = 1; j <= M; j++) {
        stddev[j] = 0.0;

        for (size_t i = 1; i <= N; i++) {
            stddev[j] += (data[i * (M + 1) + j] - mean[j]) * (data[i * (M + 1) + j] - mean[j]);
        }

        stddev[j] /= FLOAT_N;
        stddev[j] = sqrt_of_array_cell(stddev, j);
        stddev[j] = stddev[j] <= EPS ? 1.0 : stddev[j];
    }

    // Center and reduce the column vectors.
    for (size_t i = 1; i <= N; i++) {
        for (size_t j = 1; j <= M; j++) {
            data[i * (M + 1) + j] -= mean[j];
            data[i * (M + 1) + j] /= sqrt(FLOAT_N);
            data[i * (M + 1) + j] /= stddev[j];
        }
    }

    // Calculate the m * m correlation matrix.
    for (size_t j1 = 1; j1 <= M - 1; j1++) {
        symmat[j1 * (M + 1) + j1] = 1.0;

        for (size_t j2 = j1 + 1; j2 <= M; j2++) {
            symmat[j1 * (M + 1) + j2] = 0.0;

            for (size_t i = 1; i <= N; i++) {
                symmat[j1 * (M + 1) + j2] += (data[i * (M + 1) + j1] * data[i * (M + 1) + j2]);
            }

            symmat[j2 * (M + 1) + j1] = symmat[j1 * (M + 1) + j2];
        }
    }
    symmat[M * (M + 1) + M] = 1.0;
}
DATA_TYPE float_n = 3214212.01;
void covariance(DATA_TYPE* data, DATA_TYPE* symmat, DATA_TYPE* mean, size_t size) {
    // printf("covariance size=%d\n", size);
    const auto M = size;
    const auto N = size;

    // Determine mean of column vectors of input data matrix
    for (size_t j = 1; j <= M; j++) {
        mean[j] = 0.0;
        for (size_t i = 1; i <= N; i++) {
            mean[j] += data[i * (M + 1) + j];
        }
        mean[j] /= float_n;
    }

    // Center the column vectors.
    for (size_t i = 1; i <= N; i++) {
        for (size_t j = 1; j <= M; j++) {
            data[i * (M + 1) + j] -= mean[j];
        }
    }

    // Calculate the m * m covariance matrix.
    for (size_t j1 = 1; j1 <= M; j1++) {
        for (size_t j2 = j1; j2 <= M; j2++) {
            symmat[j1 * (M + 1) + j2] = 0.0;
            for (size_t i = 1; i <= N; i++) {
                symmat[j1 * (M + 1) + j2] += data[i * (M + 1) + j1] * data[i * (M + 1) + j2];
            }
            symmat[j2 * (M + 1) + j1] = symmat[j1 * (M + 1) + j2];
        }
    }
}

void kernel_Fdtd(DATA_TYPE* fict, DATA_TYPE* ex, DATA_TYPE* ey, DATA_TYPE* hz, size_t size) {
    // printf("fdtd size=%d\n", size);
    const size_t NX = size;
    const size_t NY = size;
    const int TMAX = 500;
    for (size_t t = 0; t < TMAX; t++) {
        for (size_t j = 0; j < NY; j++) {
            ey[0 * NY + j] = fict[t];
        }
        for (size_t i = 1; i < NX; i++) {
            for (size_t j = 0; j < NY; j++) {
                ey[i * NY + j] = ey[i * NY + j] - 0.5 * (hz[i * NY + j] - hz[(i - 1) * NY + j]);
            }
        }
        for (size_t i = 0; i < NX; i++) {
            for (size_t j = 1; j < NY; j++) {
                ex[i * (NY + 1) + j] = ex[i * (NY + 1) + j] - 0.5 * (hz[i * NY + j] - hz[i * NY + (j - 1)]);
            }
        }
        for (size_t i = 0; i < NX; i++) {
            for (size_t j = 0; j < NY; j++) {
                hz[i * NY + j] = hz[i * NY + j] - 0.7 * (ex[i * (NY + 1) + (j + 1)] - ex[i * (NY + 1) + j] + ey[(i + 1) * NY + j] - ey[i * NY + j]);
            }
        }
    }
}

#define ALPHA 32412
#define BETA 2123
void kernel_gemm(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* C, size_t size) {
    // printf("gemm size=%d\n", size);
    const size_t NI = size;
    const size_t NJ = size;
    const size_t NK = size;

    for (size_t i = 0; i < NI; i++) {
        for (size_t j = 0; j < NJ; j++) {
            C[i * NJ + j] *= BETA;
            for (size_t k = 0; k < NK; ++k) {
                C[i * NJ + j] += ALPHA * A[i * NK + k] * B[k * NJ + j];
            }
        }
    }
}

void kernel_gesummv(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* x, DATA_TYPE* y, DATA_TYPE* tmp, size_t size) {
    // printf("gesummv size=%d\n", size);
    const size_t N = size;
    DATA_TYPE alpha = 1;
    DATA_TYPE beta = 1;
    for (size_t i = 0; i < N; i++) {
        tmp[i] = 0;
        y[i] = 0;
        for (size_t j = 0; j < N; j++) {
            tmp[i] = A[i * N + j] * x[j] + tmp[i];
            y[i] = B[i * N + j] * x[j] + y[i];
        }
        y[i] = alpha * tmp[i] + beta * y[i];
    }
}

void gramschmidt(DATA_TYPE* A, DATA_TYPE* R, DATA_TYPE* Q, size_t size) {
    // printf("gramschmidt size=%d\n", size);
    const size_t M = size;
    const size_t N = size;
    for (size_t k = 0; k < N; k++) {
        DATA_TYPE nrm = 0;
        for (size_t i = 0; i < M; i++) {
            nrm += A[i * N + k] * A[i * N + k];
        }
        R[k * N + k] = sqrt(nrm);
        for (size_t i = 0; i < M; i++) {
            Q[i * N + k] = A[i * N + k] / R[k * N + k];
        }
        for (size_t j = k + 1; j < N; j++) {
            R[k * N + j] = 0;
            for (size_t i = 0; i < M; i++) {
                R[k * N + j] += Q[i * N + k] * A[i * N + j];
            }
            for (size_t i = 0; i < M; i++) {
                A[i * N + j] = A[i * N + j] - Q[i * N + k] * R[k * N + j];
            }
        }
    }
}

void kernel_mvt(DATA_TYPE* a, DATA_TYPE* x1, DATA_TYPE* x2, DATA_TYPE* y1, DATA_TYPE* y2, size_t size) {
    // printf("mvt size=%d\n", size);
    const size_t N = size;
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++) {
            x1[i] = x1[i] + a[i * N + j] * y1[j];
        }
    }
    for (size_t k = 0; k < N; k++) {
        for (size_t l = 0; l < N; l++) {
            x2[k] = x2[k] + a[k * N + l] * y2[l];
        }
    }
}

void kernel_syrk(DATA_TYPE* A, DATA_TYPE* C, size_t size) {
    // printf("syrk size=%d\n", size);
    const size_t N = size;
    const size_t M = size;
    DATA_TYPE alpha = 123;
    DATA_TYPE beta = 14512;
    /*  C := alpha*A*A' + beta*C */
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++) {
            C[i * M + j] *= beta;
        }
    }
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++) {
            for (size_t k = 0; k < M; k++) {
                C[i * N + j] += alpha * A[i * M + k] * A[j * M + k];
            }
        }
    }
}

void kernel_syr2k(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* C, size_t size) {
    // printf("syrk2 size=%d\n", size);
    const size_t N = size;
    const size_t M = size;
    DATA_TYPE alpha = 1;
    DATA_TYPE beta = 1;
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++) {
            C[i * N + j] *= beta;
        }
    }
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++) {
            for (size_t k = 0; k < M; k++) {
                C[i * N + j] += alpha * A[i * M + k] * B[j * M + k];
                C[i * N + j] += alpha * B[i * M + k] * A[j * M + k];
            }
        }
    }
}

void kernel_add(DATA_TYPE* input1, DATA_TYPE* input2, DATA_TYPE* output, size_t size) {
    // printf("add size=%d\n", size);
    for (size_t i = 0; i < size; ++i) {
        output[i] = input1[i] + input2[i];
    }
}

void kernel_prod(DATA_TYPE* input1, DATA_TYPE* input2, DATA_TYPE* output, size_t size) {
    // printf("prod size=%d\n", size);
    for (size_t i = 0; i < size; ++i) {
        output[i] = input1[i] * input2[i];
    }
}

void kernel_coeff(float* input1, float* input2, float* sumv1, float* sumv2, float* xy, float* xx, size_t size) {
    // printf("coeff size=%d\n", size);
    for (size_t i = 0; i < size; ++i) {
        *sumv1 += input1[i];
        *sumv2 += input2[i];
        *xy += input1[i] * input2[i];
        *xx += input1[i] * input1[i];
    }
    // printf("ve result:sumv1=%f sumv2=%f xy=%f xx=%f\n", *sumv1, *sumv2, *xy, *xx);
}

#define FLT_MAX 500000.0
void kernel_kmeans(double* features, double* clusters, int* membership,
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
typedef struct {
    float x, y, z, vx, vy, vz;
} Body;
void kernel_body(Body* p, float dt, int n) {
    // printf("nbody dt=%f n=%d\n", dt, n);
    for (int i = 0; i < n; i++) {
        float Fx = 0.0f;
        float Fy = 0.0f;
        float Fz = 0.0f;
        for (int j = 0; j < n; j++) {
            float dx = p[j].x - p[i].x;
            float dy = p[j].y - p[i].y;
            float dz = p[j].z - p[i].z;
            float distSqr = dx * dx + dy * dy + dz * dz + 1e-9f;
            float invDist = 1.0f / sqrtf(distSqr);
            float invDist3 = invDist * invDist * invDist;

            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }
        p[i].vx += dt * Fx;
        p[i].vy += dt * Fy;
        p[i].vz += dt * Fz;
    }
    for (int i = 0; i < n; i += 1) {
        p[i].x += p[i].vx * dt;
        p[i].y += p[i].vy * dt;
        p[i].z += p[i].vz * dt;
    }
    // printf("ve p[2].x=%f\n", p[2].x);
}

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
void kernel_sobel(Dot* input, Dot* output, size_t size) {
    // printf("sobel size=%d\n", size);
    const float kernel[] = {1, 0, -1, 2, 0, -2, 1, 0, -1};
    int radius = 3;
    for (size_t i = 0; i < size; i++) {
        int x = i % size;
        int y = i / size;
        Dot Gx, Gy;
        for (int x_shift = 0; x_shift < 3; x_shift++)
            for (int y_shift = 0; y_shift < 3; y_shift++) {
                int xs = x + x_shift - 1;
                int ys = y + y_shift - 1;
                if (x == xs && y == ys)
                    continue;
                if (xs < 0 || xs >= size || ys < 0 || ys >= size)
                    continue;
                Dot sample = input[xs + ys * size];
                int offset_x = x_shift + y_shift * radius;
                int offset_y = y_shift + x_shift * radius;
                float conv_x = kernel[offset_x];
                Dot conv4_x = {conv_x, conv_x, conv_x};
                Gx.x += conv4_x.x * sample.x;
                Gx.y += conv4_x.y * sample.y;
                Gx.z += conv4_x.z * sample.z;
                float conv_y = kernel[offset_y];
                Dot conv4_y = {conv_y, conv_y, conv_y};
                Gy.x += conv4_y.x * sample.x;
                Gy.y += conv4_y.y * sample.y;
                Gy.z += conv4_y.z * sample.z;
            }
        Dot color = {hypot(Gx.x, Gy.x), hypot(Gx.y, Gy.y), hypot(Gx.y, Gy.y)};
        Dot minval = {0.0, 0.0, 0.0};
        Dot maxval = {1.0, 1.0, 1.0};
        if (color.x < minval.x) {
            output[i] = minval;
        } else if (color.x > maxval.x) {
            output[i] = maxval;
        } else {
            output[i] = color;
        }
    }
}
void kernel_sobel5(Dot* input, Dot* output, size_t size) {
    // printf("sobel5 size=%d\n", size);
    const float kernel[] = {1, 2, 0, -2, -1, 4, 8, 0, -8, -4, 6, 12, 0, -12, -6, 4, 8, 0, -8, -4, 1, 2, 0, -2, -1};
    int radius = 5;
    for (size_t i = 0; i < size; i++) {
        int x = i % size;
        int y = i / size;
        Dot Gx, Gy;
        for (int x_shift = 0; x_shift < 5; x_shift++)
            for (int y_shift = 0; y_shift < 5; y_shift++) {
                int xs = x + x_shift - 2;
                int ys = y + y_shift - 2;
                if (x == xs && y == ys)
                    continue;
                if (xs < 0 || xs >= size || ys < 0 || ys >= size)
                    continue;
                Dot sample = input[xs + ys * size];
                int offset_x = x_shift + y_shift * radius;
                int offset_y = y_shift + x_shift * radius;
                float conv_x = kernel[offset_x];
                Dot conv4_x = {conv_x, conv_x, conv_x};
                Gx.x += conv4_x.x * sample.x;
                Gx.y += conv4_x.y * sample.y;
                Gx.z += conv4_x.z * sample.z;
                float conv_y = kernel[offset_y];
                Dot conv4_y = {conv_y, conv_y, conv_y};
                Gy.x += conv4_y.x * sample.x;
                Gy.y += conv4_y.y * sample.y;
                Gy.z += conv4_y.z * sample.z;
            }
        Dot color = {hypot(Gx.x, Gy.x), hypot(Gx.y, Gy.y), hypot(Gx.y, Gy.y)};
        Dot minval = {0.0, 0.0, 0.0};
        Dot maxval = {1.0, 1.0, 1.0};
        if (color.x < minval.x) {
            output[i] = minval;
        } else if (color.x > maxval.x) {
            output[i] = maxval;
        } else {
            output[i] = color;
        }
    }
}

void kernel_sobel7(Dot* input, Dot* output, size_t size) {
    // printf("sobel7 size=%d\n", size);
    const float kernel[] = {
        130, 120, 78, 0, -78, -120, -130,
        180, 195, 156, 0, -156, -195, -180,
        234, 312, 390, 0, -390, -312, -234,
        260, 390, 780, 0, -780, -390, -260,
        234, 312, 390, 0, -390, -312, -234,
        180, 195, 156, 0, -156, -195, -180,
        130, 120, 78, 0, -78, -120, -130};
    int radius = 7;
    for (size_t i = 0; i < size; i++) {
        int x = i % size;
        int y = i / size;
        Dot Gx, Gy;
        for (int x_shift = 0; x_shift < 7; x_shift++)
            for (int y_shift = 0; y_shift < 7; y_shift++) {
                int xs = x + x_shift - 2;
                int ys = y + y_shift - 2;
                if (x == xs && y == ys)
                    continue;
                if (xs < 0 || xs >= size || ys < 0 || ys >= size)
                    continue;
                Dot sample = input[xs + ys * size];
                int offset_x = x_shift + y_shift * radius;
                int offset_y = y_shift + x_shift * radius;
                float conv_x = kernel[offset_x];
                Dot conv4_x = {conv_x, conv_x, conv_x};
                Gx.x += conv4_x.x * sample.x;
                Gx.y += conv4_x.y * sample.y;
                Gx.z += conv4_x.z * sample.z;
                float conv_y = kernel[offset_y];
                Dot conv4_y = {conv_y, conv_y, conv_y};
                Gy.x += conv4_y.x * sample.x;
                Gy.y += conv4_y.y * sample.y;
                Gy.z += conv4_y.z * sample.z;
            }
        Dot color = {hypot(Gx.x, Gy.x), hypot(Gx.y, Gy.y), hypot(Gx.y, Gy.y)};
        Dot minval = {0.0, 0.0, 0.0};
        Dot maxval = {1.0, 1.0, 1.0};
        if (color.x < minval.x) {
            output[i] = minval;
        } else if (color.x > maxval.x) {
            output[i] = maxval;
        } else {
            output[i] = color;
        }
    }
}

void kernel_error(float* input1, float* input2, float* alpha, float* beta, float* output, size_t size) {
    // printf("lin_reg_error size=%d\n", size);
    for (size_t i = 0; i < size; i++) {
        float error = 0.0;
        for (size_t j = 0; j < size; j++) {
            float e = (alpha[i] * input1[j] + beta[i]) - input2[j];
            error += e * e;
        }
        output[i] = error;
    }
}
