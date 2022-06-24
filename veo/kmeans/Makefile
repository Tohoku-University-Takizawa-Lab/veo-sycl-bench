include ../config.mk

TARGET = kmeans kernel.so
all: $(TARGET)

kmeans: kmeans.o
	$(CXX) $< -o $@ $(VEFLAGS)
kmeans.o: kmeans.cpp
	$(CXX) -c $< -o $@ $(VEFLAGS)
kernel.so: kernel.c
	$(NCC) -fpic -shared -o $@ $<

.PHONY: clean
clean:
	rm -rf *.o $(TARGET)

.PHONY: run
run: $(TARGET)
	./kmeans $(PROBLEM_SIZE)
