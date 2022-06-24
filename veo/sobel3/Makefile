include ../config.mk

TARGET = sobel3 kernel.so
all: $(TARGET)

sobel3: sobel3.o
	$(CXX) $< -o $@ $(VEFLAGS)
sobel3.o: sobel3.cpp
	$(CXX) -c $< -o $@ $(VEFLAGS)
kernel.so: kernel.c
	$(NCC) -fpic -shared -o $@ $<

.PHONY: clean
clean:
	rm -rf *.o $(TARGET)

.PHONY: run
run: $(TARGET)
	./sobel3 $(PROBLEM_SIZE)
