include ../config.mk

TARGET = lin_reg_coeff kernel.so
all: $(TARGET)

lin_reg_coeff: lin_reg_coeff.o
	$(CXX) $< -o $@ $(VEFLAGS)
lin_reg_coeff.o: lin_reg_coeff.cpp
	$(CXX) -c $< -o $@ $(VEFLAGS)
kernel.so: kernel.c
	$(NCC) -fpic -shared -o $@ $<

.PHONY: clean
clean:
	rm -rf *.o $(TARGET)

.PHONY: run
run: $(TARGET)
	./lin_reg_coeff $(PROBLEM_SIZE)
