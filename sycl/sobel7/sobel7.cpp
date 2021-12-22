#include "bitmap.h"
#include <CL/sycl.hpp>
#include <chrono>
#include <iostream>
#include <vector>

using namespace cl::sycl;

int main(int argc, char** argv) {
    auto start = std::chrono::steady_clock::now();
    size_t size = 5;
    if (argc > 1)
        size = std::stoi(argv[1]);

    std::vector<float4> input(size * size);
    std::vector<float4> output(size * size);
    load_bitmap_mirrored("../../Brommy.bmp", size, input);
    buffer<float4, 2> input_buf(input.data(), range<2>(size, size));
    buffer<float4, 2> output_buf(output.data(), range<2>(size, size));

    queue q;
    q.submit([&](handler& cgh) {
        auto in = input_buf.get_access<access::mode::read>(cgh);
        auto out = output_buf.get_access<access::mode::write>(cgh);
        range<2> ndrange{size, size};
        // Sobel kernel 7x7
        const float kernel[] = {130, 120, 78, 0, -78, -120, -130, 180, 195, 156, 0, -156, -195, -180, 234, 312, 390, 0,
                                -390, -312, -234, 260, 390, 780, 0, -780, -390, -260, 234, 312, 390, 0, -390, -312, -234, 180, 195, 156, 0,
                                -156, -195, -180, 130, 120, 78, 0, -78, -120, -130};

        cgh.parallel_for<class Sobel7BenchKernel>(ndrange, [in, out, kernel, size_ = size](id<2> gid) {
            int x = gid[0];
            int y = gid[1];
            float4 Gx = float4(0, 0, 0, 0);
            float4 Gy = float4(0, 0, 0, 0);
            const int radius = 7;

            // constant-size loops in [0,1,2,3,4,5,6]
            for (int x_shift = 0; x_shift < 7; x_shift++) {
                for (int y_shift = 0; y_shift < 7; y_shift++) {
                    // sample position
                    uint xs = x + x_shift - 3; // [x-3,x-2,x-1,x,x+1,x+2,x+3]
                    uint ys = y + y_shift - 3; // [y-3,y-2,y-1,y,y+1,y+2,y+2]
                    // for the same pixel, convolution is always 0
                    if (x == xs && y == ys)
                        continue;
                    // boundary check
                    if (xs < 0 || xs >= size_ || ys < 0 || ys >= size_)
                        continue;
                    // sample color
                    float4 sample = in[{xs, ys}];
                    // convolution calculation
                    int offset_x = x_shift + y_shift * radius;
                    int offset_y = y_shift + x_shift * radius;
                    float conv_x = kernel[offset_x];
                    float4 conv4_x = float4(conv_x);
                    Gx += conv4_x * sample;
                    float conv_y = kernel[offset_y];
                    float4 conv4_y = float4(conv_y);
                    Gy += conv4_y * sample;
                }
            }
            // taking root of sums of squares of Gx and Gy
            float4 color = hypot(Gx, Gy);
            float4 minval = float4(0.0, 0.0, 0.0, 0.0);
            float4 maxval = float4(1.0, 1.0, 1.0, 1.0);
            out[gid] = clamp(color, minval, maxval);
        });
    });
    q.wait();
    auto end = std::chrono::steady_clock::now();
    std::cout << "sobel7 size=" << size << " runtime(" << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << ")" << std::endl;
    return 0;
}
