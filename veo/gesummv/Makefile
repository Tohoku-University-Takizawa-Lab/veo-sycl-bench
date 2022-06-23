include ../config.mk

TARGET = gesummv kernel.so
all: $(TARGET)

gesummv: gesummv.o
	$(CXX) $< -o $@ $(VEFLAGS)
gesummv.o: gesummv.cpp
	$(CXX) -c $< -o $@ $(VEFLAGS)
kernel.so: kernel.c
	$(NCC) -fpic -shared -o $@ $<

.PHONY: clean
clean:
	rm -rf *.o $(TARGET)

.PHONY: run
run: $(TARGET)
	./gesummv $(PROBLEM_SIZE)
