#include "forwardAutoDiff.hpp"
// #include "reverseAutoDiff.hpp"
#include "reverseWengertAutoDiff.hpp"
#include <chrono>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <utility>
#include <vector>

using varType = double;

// ============================================================================
// 1. ANALYTICAL GROUND TRUTH (For Function F)
// F(a,b,c) = ((a*b + b*c) * (a - c)) / (a^2 + b^2 + c^2)
// ============================================================================
std::pair<varType, std::vector<varType>> analytical_F(varType a, varType b,
                                                      varType c) {
  // Numerator U and Denominator V
  varType U = (a * b + b * c) * (a - c);
  varType V = a * a + b * b + c * c;
  varType primal = U / V;

  // Partial derivatives of U
  varType dU_da = 2 * a * b;
  varType dU_db = a * a - c * c;
  varType dU_dc = -2 * b * c;

  // Partial derivatives of V
  varType dV_da = 2 * a;
  varType dV_db = 2 * b;
  varType dV_dc = 2 * c;

  // Quotient Rule: d(U/V) = (dU*V - U*dV) / V^2
  varType dF_da = (dU_da * V - U * dV_da) / (V * V);
  varType dF_db = (dU_db * V - U * dV_db) / (V * V);
  varType dF_dc = (dU_dc * V - U * dV_dc) / (V * V);

  return {primal, {dF_da, dF_db, dF_dc}};
}

// ============================================================================
// 2. REVERSE AD FUNCTIONS (Taking Graph Pointers)
// ============================================================================
std::vector<varType>
myConvolutedF(std::vector<autodiff::reverseNode<varType>> *input_graph) {
  std::vector<varType> result{};
  auto a = (*input_graph)[0];
  auto b = (*input_graph)[1];
  auto c = (*input_graph)[2];

  autodiff::reverseNode<varType> f =
      ((a * b + b * c) * (a - c)) / (a * a + b * b + c * c);

  result.push_back(f.get_v());
  f.set_output(1);
  input_graph->push_back(f);
  return result;
}

// Scalable G: Chained operations over N variables.
std::vector<varType>
myScalableG(std::vector<autodiff::reverseNode<varType>> *input_graph) {
  std::vector<varType> result{};

  // 1. FREEZE THE SIZE! Only iterate over the original inputs.
  size_t original_num_inputs = input_graph->size();

  auto g = (*input_graph)[0] * (*input_graph)[1];

  // 2. Use the frozen size as the loop limit limit
  for (size_t i = 1; i < original_num_inputs - 1; ++i) {
    g = g + ((*input_graph)[i] * (*input_graph)[i + 1]) /
                ((*input_graph)[i] - (*input_graph)[i + 1]);
  }

  // 3. Prevent the phantom node (modify the real node in the graph, do not push
  // a copy)
  auto &real_final_node = (*input_graph)[g.get_key()];
  real_final_node.set_output(1);

  result.push_back(real_final_node.get_v());
  return result;
}

// ============================================================================
// 3. FORWARD AD FUNCTIONS (Taking Forward Objects)
// ============================================================================
autodiff::forwardNums<varType>
forward_math_F(const std::vector<autodiff::forwardNums<varType>> &x) {
  return ((x[0] * x[1] + x[1] * x[2]) * (x[0] - x[2])) /
         (x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
}

autodiff::forwardNums<varType>
forward_math_G(const std::vector<autodiff::forwardNums<varType>> &x) {
  auto g = x[0] * x[1];
  for (size_t i = 1; i < x.size() - 1; ++i) {
    g = g + (x[i] * x[i + 1]) / (x[i] - x[i + 1]);
  }
  return g;
}

// ============================================================================
// MAIN EXECUTION
// ============================================================================
int main() {
  std::cout << std::fixed << std::setprecision(6);
  std::cout << "==========================================================\n";
  std::cout << "  BENCHMARKING & VALIDATION: FORWARD vs REVERSE AD\n";
  std::cout << "==========================================================\n\n";

  // -------------------------------------------------------------------------
  // TEST 1: FUNCTION F (Small Scale - 3 Inputs) - Validation Test
  // -------------------------------------------------------------------------
  const int num_iterations_F = 300000;
  std::vector<varType> inputs_F{25.0, 4.0, -5.0};

  std::cout << "--- [FUNCTION F] 3 Inputs | " << num_iterations_F
            << " Iterations ---\n";

  // Analytical
  auto analytical_res = analytical_F(inputs_F[0], inputs_F[1], inputs_F[2]);

  // Forward AD
  varType fwd_primal_F;
  std::vector<varType> fwd_grad_F(3);
  auto start_fwd_F = std::chrono::high_resolution_clock::now();
  for (int iter = 0; iter < num_iterations_F; ++iter) {
    for (int i = 0; i < 3; ++i) {
      std::vector<autodiff::forwardNums<varType>> fwd_inputs(3);
      for (int j = 0; j < 3; ++j) {
        fwd_inputs[j].set_v(inputs_F[j]);
        fwd_inputs[j].set_dv(i == j ? 1.0 : 0.0);
      }
      auto out = forward_math_F(fwd_inputs);
      if (iter == num_iterations_F - 1) {
        fwd_grad_F[i] = out.get_dv();
        if (i == 0)
          fwd_primal_F = out.get_v();
      }
    }
  }
  auto end_fwd_F = std::chrono::high_resolution_clock::now();

  // Reverse AD
  varType rev_primal_F;
  std::vector<varType> rev_grad_F;
  auto start_rev_F = std::chrono::high_resolution_clock::now();
  for (int iter = 0; iter < num_iterations_F; ++iter) {
    auto result = autodiff::autoGrad<varType>(myConvolutedF, inputs_F);
    if (iter == num_iterations_F - 1) {
      rev_primal_F = result.first[0];
      rev_grad_F = result.second;
    }
  }
  auto end_rev_F = std::chrono::high_resolution_clock::now();

  // Print F Comparisons
  std::cout << "[Primal Outputs]\n";
  std::cout << "  Analytical: " << analytical_res.first << "\n";
  std::cout << "  Forward   : " << fwd_primal_F << "\n";
  std::cout << "  Reverse   : " << rev_primal_F << "\n\n";

  std::cout << "[Gradients]\n";
  std::cout << "  Analytical: [ " << analytical_res.second[0] << " | "
            << analytical_res.second[1] << " | " << analytical_res.second[2]
            << " ]\n";
  std::cout << "  Forward   : [ " << fwd_grad_F[0] << " | " << fwd_grad_F[1]
            << " | " << fwd_grad_F[2] << " ]\n";
  std::cout << "  Reverse   : [ " << rev_grad_F[0] << " | " << rev_grad_F[1]
            << " | " << rev_grad_F[2] << " ]\n\n";

  std::cout << "[Execution Time]\n";
  std::cout << "  Forward AD: "
            << std::chrono::duration<double, std::milli>(end_fwd_F -
                                                         start_fwd_F)
                   .count()
            << " ms\n";
  std::cout << "  Reverse AD: "
            << std::chrono::duration<double, std::milli>(end_rev_F -
                                                         start_rev_F)
                   .count()
            << " ms\n\n";

  // -------------------------------------------------------------------------
  // TEST 2: FUNCTION G (Large Scale - 1000 Inputs) - Scaling Test
  // -------------------------------------------------------------------------
  const int num_inputs_G = 1000;
  const int num_iterations_G =
      100; // Drastically lower iterations due to extreme graph size

  std::vector<varType> inputs_G(num_inputs_G);
  for (size_t i = 0; i < num_inputs_G; ++i) {
    inputs_G[i] = (i + 1.0) * 1.5; // Avoid div by zero
  }

  std::cout << "--- [FUNCTION G] " << num_inputs_G << " Inputs | "
            << num_iterations_G << " Iterations ---\n";

  // Forward AD
  varType fwd_primal_G;
  std::vector<varType> fwd_grad_G(num_inputs_G);
  auto start_fwd_G = std::chrono::high_resolution_clock::now();
  for (int iter = 0; iter < num_iterations_G; ++iter) {
    for (int i = 0; i < num_inputs_G; ++i) { // <--- The O(N) penalty loop!
      std::vector<autodiff::forwardNums<varType>> fwd_inputs(num_inputs_G);
      for (int j = 0; j < num_inputs_G; ++j) {
        fwd_inputs[j].set_v(inputs_G[j]);
        fwd_inputs[j].set_dv(i == j ? 1.0 : 0.0);
      }
      auto out = forward_math_G(fwd_inputs);
      if (iter == num_iterations_G - 1) {
        fwd_grad_G[i] = out.get_dv();
        if (i == 0)
          fwd_primal_G = out.get_v();
      }
    }
  }
  auto end_fwd_G = std::chrono::high_resolution_clock::now();

  // Reverse AD
  varType rev_primal_G;
  std::vector<varType> rev_grad_G;
  auto start_rev_G = std::chrono::high_resolution_clock::now();
  for (int iter = 0; iter < num_iterations_G; ++iter) {
    auto result = autodiff::autoGrad<varType>(
        myScalableG, inputs_G); // <--- No inner loop needed!
    if (iter == num_iterations_G - 1) {
      rev_primal_G = result.first[0];
      rev_grad_G = result.second;
    }
  }
  auto end_rev_G = std::chrono::high_resolution_clock::now();

  // Pick 3 random indices to compare
  size_t idx1 = 0, idx2 = num_inputs_G / 2, idx3 = num_inputs_G - 1;

  std::cout << "[Primal Outputs]\n";
  std::cout << "  Forward   : " << fwd_primal_G << "\n";
  std::cout << "  Reverse   : " << rev_primal_G << "\n\n";

  std::cout << "[Gradients at Indices: 0, " << idx2 << ", " << idx3 << "]\n";
  std::cout << "  Forward   : [ " << fwd_grad_G[idx1] << " | "
            << fwd_grad_G[idx2] << " | " << fwd_grad_G[idx3] << " ]\n";
  std::cout << "  Reverse   : [ " << rev_grad_G[idx1] << " | "
            << rev_grad_G[idx2] << " | " << rev_grad_G[idx3] << " ]\n\n";

  std::cout << "[Execution Time]\n";
  std::cout << "  Forward AD: "
            << std::chrono::duration<double, std::milli>(end_fwd_G -
                                                         start_fwd_G)
                   .count()
            << " ms\n";
  std::cout << "  Reverse AD: "
            << std::chrono::duration<double, std::milli>(end_rev_G -
                                                         start_rev_G)
                   .count()
            << " ms\n";
  std::cout << "==========================================================\n";

  return 0;
}
