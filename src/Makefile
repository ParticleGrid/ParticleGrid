
CFLAGS = -g -O -ffunction-sections -ffast-math -mfma -Wall $(CDEFS) $(CEXTRAS)

main: main.cpp
	$(CXX) -shared -std=c++11 -fPIC -fopenmp $$(python3 -m pybind11 --includes) $^ -o $@$$(python3-config --extension-suffix) $(CFLAGS)

clean:
	rm *.o *.so
