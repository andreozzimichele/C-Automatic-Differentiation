#include "forwardAutoDiff.hpp"
#include "reverseAutoDiff.hpp"
#include <cstddef>
#include <vector>

using varType = double;

// keep initialized as a pointer to vector of nodes output (in the future the
// output may be in R^n, at the moment single output (i.e. single element
// vector))
std::vector<varType>
myComputationF(std::vector<autodiff::reverseNode<varType>> *input_graph) {

  std::vector<varType> result{};

  autodiff::reverseNode<varType> f = (*input_graph)[0] * (*input_graph)[1] +
                                     (*input_graph)[2] * (*input_graph)[0];

  result.push_back(f.get_v());
  f.set_output(1); // we specify that f is the function output, i.e.
                   // the starting point of the reverse graph
  input_graph->push_back(f);

  return result;
}

std::vector<varType>
myComputationG(std::vector<autodiff::reverseNode<varType>> *input_graph) {

  std::vector<varType> result{};

  autodiff::reverseNode<varType> g =
      (*input_graph)[0] / (*input_graph)[1] - (*input_graph)[2];

  result.push_back(g.get_v());
  g.set_output(1); // we specify that f is the function output, i.e.
                   // the starting point of the reverse graph
  input_graph->push_back(g);

  return result;
}
int main() {

  // Forward autodiff calculations
  std::cout << "##########################################################"
               "\nForward AD\n\n";
  // Autodiff wrt x
  autodiff::forwardNums<varType> a{25, 1};
  autodiff::forwardNums<varType> b{4, 0};
  autodiff::forwardNums<varType> c{-5, 0};

  autodiff::forwardNums<varType> d{};
  d = a * b + c * a;
  d.print();

  d = a / b - c;
  d.print();

  // Autodiff wrt y
  a.set_dv(0);
  b.set_dv(1);

  d = a * b + c * a;
  d.print();

  d = a / b - c;
  d.print();

  // Autodiff wrt z
  b.set_dv(0);
  c.set_dv(1);

  d = a * b + c * a;
  d.print();

  d = a / b - c;
  d.print();

  // USEFUL FOR DEBUGGING MULTIPLICATION BY A SCALAR
  /*
  d = 1.1 * d;
  d.print();
  */

  std::cout << "\n";

  // Reverse autodiff calculations
  std::cout << "##########################################################"
               "\nReverse AD\n";
  std::vector<varType> input_values{25, 4, -5};
  std::pair<std::vector<varType>, std::vector<varType>> resultF{
      autodiff::autoGrad<varType>(myComputationF, input_values)};
  std::vector<varType> primalF{resultF.first};
  std::vector<varType> gradientF{resultF.second};

  std::cout << "Primal f: [";
  for (size_t it{0}; it < primalF.size(); ++it) {
    std::cout << primalF[it] << " ";
  }
  std::cout << "]\n";

  std::cout << "The final gradient is: [ ";
  for (size_t it{0}; it < gradientF.size(); ++it) {
    std::cout << gradientF[it] << " ";
  }
  std::cout << "]\n\n";

  // Reverse autodiff calculations
  std::cout << "##########################################################"
               "\nReverse AD\n";
  std::pair<std::vector<varType>, std::vector<varType>> resultG{
      autodiff::autoGrad<varType>(myComputationG, input_values)};
  std::vector<varType> primalG{resultG.first};
  std::vector<varType> gradientG{resultG.second};

  std::cout << "Primal g: [";
  for (size_t it{0}; it < primalG.size(); ++it) {
    std::cout << primalG[it] << " ";
  }
  std::cout << "]\n";

  std::cout << "The final gradient is: [ ";
  for (size_t it{0}; it < gradientG.size(); ++it) {
    std::cout << gradientG[it] << " ";
  }
  std::cout << "]\n\n";
}
