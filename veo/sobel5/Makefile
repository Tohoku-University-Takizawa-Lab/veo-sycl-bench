include ../config.mk

TARGET = sobel5 kernel.so
all: $(TARGET)

sobel5: sobel5.o
	$(CXX) $< -o $@ $(VEFLAGS)
sobel5.o: sobel5.cpp
	$(CXX) -c $< -o $@ $(VEFLAGS)
kernel.so: kernel.c
	$(NCC) -fpic -shared -o $@ $<

.PHONY: clean
clean:
	rm -rf *.o $(TARGET)

.PHONY: run
run: $(TARGET)
	./sobel5 $(PROBLEM_SIZE)
