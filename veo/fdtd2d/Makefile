include ../config.mk

TARGET = fdtd2d kernel.so
all: $(TARGET)

fdtd2d: fdtd2d.o
	$(CXX) $< -o $@ $(VEFLAGS)
fdtd2d.o: fdtd2d.cpp
	$(CXX) -c $< -o $@ $(VEFLAGS)
kernel.so: kernel.c
	$(NCC) -fpic -shared -o $@ $<

.PHONY: clean
clean:
	rm -rf *.o $(TARGET)

.PHONY: run
run: $(TARGET)
	./fdtd2d $(PROBLEM_SIZE)
