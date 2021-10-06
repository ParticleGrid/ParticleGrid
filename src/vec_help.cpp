
#include <string>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

bool support(){
    return __builtin_cpu_supports("avx") > 0;
}

PYBIND11_MODULE(vec_help, m) {
    m.doc() = "Detect AVX";
    m.def("support", &support,
        "Check for extension support");
}
