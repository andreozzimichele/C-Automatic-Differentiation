#include "../include/reverseAutoDiff.hpp"
#include <iostream>
#include <vector>

using varType = double;

// Define your mathematical function
std::vector<varType>
myFunction(std::vector<autodiff::reverseNode<varType>> *graph) {
  auto a = (*graph)[0];
  auto b = (*graph)[1];

  auto f = a * a + a * b * b; // The math is traced automatically

  // Flag the final node as the output
  (*graph)[f.get_key()].set_output(true);

  return {f.get_v()};
}

int main() {
  // 1. Define input values [a, b]
  std::vector<varType> inputs = {2.0, 4.0};

  // 2. Run the engine (Returns a pair: [Primal Output, Gradients])
  auto result = autodiff::autoGrad<varType>(myFunction, inputs);

  std::cout << "f(a,b) = " << result.first[0] << "\n";
  std::cout << "df/da  = " << result.second[0] << "\n";
  std::cout << "df/db  = " << result.second[1] << "\n";

  return 0;
}
