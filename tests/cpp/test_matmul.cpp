#include "math_utils.hpp"

#include <vector>
#include <iostream>
#include <cmath>

template<typename T>
bool is_close( T a, T b)
{
  return std::fabs(a-b) < 1e-7;
}


int main(int argc, char* argv[]){
  /*
      1, 1, 1
      2, 2, 2,
      3, 3, 3
  */
  std::vector<double> input_mat({1, 1, 1, 2, 2, 2, 3, 3, 3});

  std::array<double, 9> _transform;
  
  /*
      1, 0, 0
      0, 1, 0
      0, 0, 1
  */
  _transform = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  
  constexpr size_t m = 3;
  std::vector<double> output(9, 0); 

  matmul<double>(input_mat.data(), _transform, output.data(), m);

  std::cout << "Expected output: \n";

  for(auto i = 0; i < m; ++i){
    for (auto k = 0; k < 3; ++k){
      std::cout << input_mat[i * 3 + k] << ",\t";
    }
    std::cout << std::endl;
  }

  std::cout << "Calculated output: \n";

  bool all_match = true;
  for(auto i = 0; i < m; ++i){
    for (auto k = 0; k < 3; ++k){
      
      all_match = is_close(output[i * 3 + k], input_mat[i * 3 + k]);
      std::cout << output[i * 3 + k] << ",\t";
    }
    std::cout << std::endl;
  }

  std::cout << "Test Status: " << (all_match ? "Passing" : "Failing") << std::endl;

  return 0;
}
