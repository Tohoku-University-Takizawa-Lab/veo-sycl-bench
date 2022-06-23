include ../config.mk

TARGET = scalar_prod kernel.so
all: $(TARGET)

scalar_prod: scalar_prod.o
	$(CXX) $< -o $@ $(VEFLAGS)
scalar_prod.o: scalar_prod.cpp
	$(CXX) -c $< -o $@ $(VEFLAGS)
kernel.so: kernel.c
	$(NCC) -fpic -shared -o $@ $<

.PHONY: clean
clean:
	rm -rf *.o $(TARGET)

.PHONY: run
run: $(TARGET)
	./scalar_prod $(PROBLEM_SIZE)
