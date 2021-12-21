#include "bitmap.h"
#include <CL/sycl.hpp>
#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

REGISTER_KERNEL(sobel7);
using namespace cl::sycl;

int main(int argc, char** argv) {
    auto start = std::chrono::steady_clock::now();
    int size = 5;
    if (argc > 1)
        size = std::stoi(argv[1]);

    accelerator_selector as;
    queue q(as);
    // queue q;

    vector<Dot> input(size * size);
    load_bitmap_mirrored("../../Brommy.bmp", size, input);
    std::vector<Dot> output(size * size);
    buffer<Dot> in_buff(input.data(), range<1>(size * size));
    buffer<Dot> out_buff(output.data(), range<1>(size * size));
    buffer<int> n_buff(&size, range<1>(1));

    q.submit([&](handler& cgh) {
        auto in_access = in_buff.get_access<access::mode::read_write>(cgh);
        auto out_access = out_buff.get_access<access::mode::read_write>(cgh);
        auto no_access = n_buff.get_access<access::mode::read_write>(cgh);

        cgh.single_task<class sobel7>([=]() {
            const float kernel[] = {
                130, 120, 78, 0, -78, -120, -130,
                180, 195, 156, 0, -156, -195, -180,
                234, 312, 390, 0, -390, -312, -234,
                260, 390, 780, 0, -780, -390, -260,
                234, 312, 390, 0, -390, -312, -234,
                180, 195, 156, 0, -156, -195, -180,
                130, 120, 78, 0, -78, -120, -130};
            int radius = 7;
            for (size_t i = 0; i < no_access[0]; i++) {
                int x = i % no_access[0];
                int y = i / no_access[0];
                Dot Gx, Gy;
                for (int x_shift = 0; x_shift < 7; x_shift++)
                    for (int y_shift = 0; y_shift < 7; y_shift++) {
                        int xs = x + x_shift - 1;
                        int ys = y + y_shift - 1;
                        if (x == xs && y == ys)
                            continue;
                        if (xs < 0 || xs >= no_access[0] || ys < 0 || ys >= no_access[0])
                            continue;
                        Dot sample = in_access[xs + ys * no_access[0]];
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
    auto end = std::chrono::steady_clock::now();
    std::cout << "sobel7 size=" << size << " runtime(" << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << ")" << std::endl;
    return 0;
}
