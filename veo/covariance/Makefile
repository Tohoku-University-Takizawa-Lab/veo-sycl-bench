include ../config.mk

TARGET = covariance kernel.so
all: $(TARGET)

covariance: covariance.o
	$(CXX) $< -o $@ $(VEFLAGS)
covariance.o: covariance.cpp
	$(CXX) -c $< -o $@ $(VEFLAGS)
kernel.so: kernel.c
	$(NCC) -fpic -shared -o $@ $<

.PHONY: clean
clean:
	rm -rf *.o $(TARGET)

.PHONY: run
run: $(TARGET)
	./covariance $(PROBLEM_SIZE)
