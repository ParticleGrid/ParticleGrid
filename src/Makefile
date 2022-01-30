
#use CDEFS=-DNO_V8F to remove avx support

MAVX=
ifeq (,$(findstring -DNO_V8F,$(CDEFS)))
    # Found
    MAVX=-mavx
endif
FOMP=
ifeq (,$(findstring -DOMP_OFF,$(CDEFS)))
    # Found
    FOMP=-fopenmp
endif

CFLAGS = -g -O -ffunction-sections -ffast-math -mfma -Wall $(MAVX) $(FOMP) $(CDEFS) $(CEXTRAS)

ParticleGrid: cgridgen.cpp
	$(CXX) -shared -std=c++11 -fPIC $$(python3 -m pybind11 --includes) $^ -o $@$$(python3-config --extension-suffix) $(CFLAGS)

bench: generate_bench.cpp
	$(CXX) $^ -o $@ $(CFLAGS)

clean:
	rm *.o *.so
