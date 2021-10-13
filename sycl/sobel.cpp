#include "bitmap.h"
#include "common.h"
#include <CL/sycl.hpp>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

REGISTER_KERNEL(sobel);
using namespace cl::sycl;

int main(int argc, char** argv) {
    long begin = get_time();
    size_t size = 5;
    if (argc > 1)
        size = atoi(argv[1]);

    accelerator_selector as;
    queue q(as);

    vector<Dot> input;
    input.resize(size * size);
    load_bitmap_mirrored("../Brommy.bmp", size, input);
    buffer<Dot> in_buff(input.data(), range<1>(size * size));
    Dot* output = (Dot*)malloc(size * size * sizeof(Dot));
    buffer<Dot> out_buff(output, range<1>(size * size));

    q.submit([&](handler& cgh) {
        auto in_access = in_buff.get_access<access::mode::read>(cgh);
        auto out_access = out_buff.get_access<access::mode::write>(cgh);
        cgh.parallel_for<class sobel>(range<1>(size), [=](id<1> i) {
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
                        Dot sample = in_access[xs + ys * size];
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
                    out_access[i] = minval;
                } else if (color.x > maxval.x) {
                    out_access[i] = maxval;
                } else {
                    out_access[i] = color;
                }
            }
        });
    });
    q.wait();

    long finish = get_time();
    printf("runtime %ld ms\n", finish - begin);
    return 0;
}
