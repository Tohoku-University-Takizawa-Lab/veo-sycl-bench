include ../config.mk

TARGET = lin_reg_error kernel.so
all: $(TARGET)

lin_reg_error: lin_reg_error.o
	$(CXX) $< -o $@ $(VEFLAGS)
lin_reg_error.o: lin_reg_error.cpp
	$(CXX) -c $< -o $@ $(VEFLAGS)
kernel.so: kernel.c
	$(NCC) -fpic -shared -o $@ $<

.PHONY: clean
clean:
	rm -rf *.o $(TARGET)

.PHONY: run
run: $(TARGET)
	./lin_reg_error $(PROBLEM_SIZE)
