HOME := $(shell dirname $$(dirname $$PWD))
INCLUDES := $(HOME)/include

CFLAGS := -I$(INCLUDES) -ffast-math -mfma -fopenmp -O3

all: mult_test

mult_test: test_matmul.cpp
	$(CXX) $^ -o test_matmul.out $(CFLAGS)
	./test_matmul.out

fc_test: test_frac2car.cpp
	$(CXX) $^ -o test_frac2car.out $(CFLAGS)