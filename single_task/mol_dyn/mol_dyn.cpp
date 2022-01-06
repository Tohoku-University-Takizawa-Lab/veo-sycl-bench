#include <CL/sycl.hpp>
#include <chrono>
#include <iostream>
#include <vector>
#include <cmath>

REGISTER_KERNEL(dyn);
using namespace cl::sycl;

typedef struct {
    float x, y, z;
} Atom;

int main(int argc, char** argv) {
    auto start = std::chrono::steady_clock::now();
    int size = 5;
    if (argc > 1)
        size = std::stoi(argv[1]);

    accelerator_selector as;
    queue q(as);
    // queue q;

    std::vector<Atom> input(size);
    std::vector<Atom> output(size);
    std::vector<int> neighbour(size);
    for (int i = 0; i < size; i++) {
        Atom temp = {(float)i, (float)i, (float)i};
        input[i] = temp;
    }
    for (int i = 0; i < size; i++) {
        neighbour[i] = i + 1;
    }

    buffer<Atom> in_buff(input.data(), range<1>(size));
    buffer<int> neb_buff(neighbour.data(), range<1>(size));
    buffer<Atom> out_buff(output.data(), range<1>(size));
    buffer<int> n_buff(&size, range<1>(1));
    auto task_start = std::chrono::steady_clock::now();
    q.submit([&](handler& cgh) {
        auto in_access = in_buff.get_access<access::mode::read_write>(cgh);
        auto out_access = out_buff.get_access<access::mode::read_write>(cgh);
        auto neb_access = neb_buff.get_access<access::mode::read_write>(cgh);
        auto no_access = n_buff.get_access<access::mode::read_write>(cgh);

        cgh.single_task<class dyn>([=]() {
            int neighCount = 15;
            int cutsq = 50;
            int lj1 = 20;
            float lj2 = 0.003f;
            int inum = 0;
            for (int i = 0; i < no_access[0]; ++i) {
                Atom ipos = in_access[i];
                Atom f = {0.0f, 0.0f, 0.0f};
                int j = 0;
                while (j < neighCount) {
                    int jidx = neb_access[j * inum + i];
                    Atom jpos = in_access[jidx];
                    float delx = ipos.x - jpos.x;
                    float dely = ipos.y - jpos.y;
                    float delz = ipos.z - jpos.z;
                    float r2inv = delx * delx + dely * dely + delz * delz;
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
                out_access[i] = f;
            }
        });
    });
    q.wait();
    auto task_end = std::chrono::steady_clock::now();
    auto end = std::chrono::steady_clock::now();
    std::cout << "size=" << size 
    << " runtime(" << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << ")" 
    << " task(" << std::chrono::duration_cast<std::chrono::milliseconds>(task_end - task_start).count() << ")" 
    << std::endl;
    return 0;
}