include ../config.mk

TARGET = bicg kernel.so
all: $(TARGET)

bicg: bicg.o
	$(CXX) $< -o $@ $(VEFLAGS)
bicg.o: bicg.cpp
	$(CXX) -c $< -o $@ $(VEFLAGS)
kernel.so: kernel.c
	$(NCC) -fpic -shared -o $@ $<

.PHONY: clean
clean:
	rm -rf *.o $(TARGET)

.PHONY: run
run: $(TARGET)
	./bicg $(PROBLEM_SIZE)
