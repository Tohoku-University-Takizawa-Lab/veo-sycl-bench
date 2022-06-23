include ../config.mk

TARGET = atax kernel.so
all: $(TARGET)

atax: atax.o
	$(CXX) $< -o $@ $(VEFLAGS)
atax.o: atax.cpp
	$(CXX) -c $< -o $@ $(VEFLAGS)
kernel.so: kernel.c
	$(NCC) -fpic -shared -o $@ $<

.PHONY: clean
clean:
	rm -rf *.o $(TARGET)

.PHONY: run
run: $(TARGET)
	./atax $(PROBLEM_SIZE)
