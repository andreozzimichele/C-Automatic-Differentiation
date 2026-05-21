#pragma once

#include <array>
#include <cstddef>
#include <iostream>
#include <limits>
#include <utility>
#include <vector>

namespace autodiff {

// Removed "output" from opType, as being an output is a state, not a math
// operation
enum class opType { input, sum, sub, mult, div };

template <typename T> class reverseNode {
private:
  T _value{};
  T _adjointGrad{0.0};
  size_t _key{};
  std::array<size_t, 2> _fathers{std::numeric_limits<size_t>::max(),
                                 std::numeric_limits<size_t>::max()};
  opType _op_type{};
  bool _is_output{
      false}; // Decoupled tag to maintain pure mathematical operations

  // Raw pointer to the graph (Stack address never changes, perfectly safe)
  std::vector<reverseNode<T>> *_computational_graph{nullptr};

public:
  // Constructor (No _children vector!)
  reverseNode(T input_v, size_t input_key, std::array<size_t, 2> input_fathers,
              opType input_generatrixOperation,
              std::vector<reverseNode<T>> *input_graph)
      : _value(input_v), _key(input_key), _fathers(input_fathers),
        _op_type(input_generatrixOperation), _computational_graph(input_graph) {
  }

  // Explicitly relying on the default destructor for lightning-fast memory
  // cleanup

  // Fast, inline, no-exception Getters
  T get_v() const noexcept { return _value; }
  T get_adjoint() const noexcept { return _adjointGrad; }
  size_t get_key() const noexcept { return _key; }
  const std::array<size_t, 2> &get_fathers() const noexcept { return _fathers; }
  opType get_op() const noexcept { return _op_type; }
  bool is_output() const noexcept { return _is_output; }

  // Setters
  void set_adjoint(T adjointGrad) noexcept { _adjointGrad = adjointGrad; }
  void add_adjoint(T adjointGrad) noexcept { _adjointGrad += adjointGrad; }
  void set_output(bool check = true) noexcept { _is_output = check; }

  // -----------------------------------------------------------------------
  // OVERLOADED OPERATORS (Building the Wengert Tape)
  // -----------------------------------------------------------------------
  friend reverseNode<T> operator+(const reverseNode<T> &lhs,
                                  const reverseNode<T> &rhs) {
    std::vector<reverseNode<T>> *graph = lhs._computational_graph;
    size_t newKey = graph->size();
    std::array<size_t, 2> fathers{lhs._key, rhs._key};

    reverseNode<T> newNode(lhs._value + rhs._value, newKey, fathers,
                           opType::sum, graph);
    graph->push_back(newNode);
    return newNode;
  }

  friend reverseNode<T> operator-(const reverseNode<T> &lhs,
                                  const reverseNode<T> &rhs) {
    std::vector<reverseNode<T>> *graph = lhs._computational_graph;
    size_t newKey = graph->size();
    std::array<size_t, 2> fathers{lhs._key, rhs._key};

    reverseNode<T> newNode(lhs._value - rhs._value, newKey, fathers,
                           opType::sub, graph);
    graph->push_back(newNode);
    return newNode;
  }

  friend reverseNode<T> operator*(const reverseNode<T> &lhs,
                                  const reverseNode<T> &rhs) {
    std::vector<reverseNode<T>> *graph = lhs._computational_graph;
    size_t newKey = graph->size();
    std::array<size_t, 2> fathers{lhs._key, rhs._key};

    reverseNode<T> newNode(lhs._value * rhs._value, newKey, fathers,
                           opType::mult, graph);
    graph->push_back(newNode);
    return newNode;
  }

  friend reverseNode<T> operator/(const reverseNode<T> &lhs,
                                  const reverseNode<T> &rhs) {
    std::vector<reverseNode<T>> *graph = lhs._computational_graph;
    size_t newKey = graph->size();
    std::array<size_t, 2> fathers{lhs._key, rhs._key};

    reverseNode<T> newNode(lhs._value / rhs._value, newKey, fathers,
                           opType::div, graph);
    graph->push_back(newNode);
    return newNode;
  }

  void print() const {
    std::cout << "Key: " << _key << " | Value: " << _value
              << " | Adjoint: " << _adjointGrad << " | Fathers: ["
              << _fathers[0] << ", " << _fathers[1] << "]\n";
  }
};

// -----------------------------------------------------------------------
// THE BACKWARD SWEEP (Push Architecture)
// -----------------------------------------------------------------------
template <typename T> void reverseGraphAD(std::vector<reverseNode<T>> *graph) {

  // 1. Seed the outputs
  for (auto &node : *graph) {
    if (node.is_output()) {
      node.set_adjoint(1.0);
    }
  }

  // 2. Walk backwards down the tape pushing adjoints to fathers
  for (auto it = graph->rbegin(); it != graph->rend(); ++it) {

    T current_adj = it->get_adjoint();

    // Optimization: If no gradient flowed into this node, skip the math
    // entirely
    if (current_adj == 0.0)
      continue;

    opType op = it->get_op();

    // Inputs have no fathers to push to
    if (op == opType::input)
      continue;

    size_t left = it->get_fathers()[0];
    size_t right = it->get_fathers()[1];

    if (op == opType::sum) {
      (*graph)[left].add_adjoint(current_adj);
      (*graph)[right].add_adjoint(current_adj);
    } else if (op == opType::sub) {
      (*graph)[left].add_adjoint(current_adj);
      (*graph)[right].add_adjoint(-current_adj);
    } else if (op == opType::mult) {
      (*graph)[left].add_adjoint(current_adj * (*graph)[right].get_v());
      (*graph)[right].add_adjoint(current_adj * (*graph)[left].get_v());
    } else if (op == opType::div) {
      T right_v = (*graph)[right].get_v();
      (*graph)[left].add_adjoint(current_adj / right_v);
      (*graph)[right].add_adjoint(-current_adj * (*graph)[left].get_v() /
                                  (right_v * right_v));
    }
  }
}

// -----------------------------------------------------------------------
// THE DRIVER FUNCTION
// -----------------------------------------------------------------------
template <typename T, typename Func>
std::pair<std::vector<T>, std::vector<T>>
autoGrad(Func function, const std::vector<T> &input_values) {

  // Stack allocated vector. It will grow on the heap dynamically.
  std::vector<reverseNode<T>> computational_graph;

  // Prevent vector reallocations by reserving a massive contiguous chunk of
  // memory
  computational_graph.reserve(1000000);

  // Initialize Input Variables
  for (size_t i = 0; i < input_values.size(); ++i) {
    std::array<size_t, 2> input_fathers{std::numeric_limits<size_t>::max(),
                                        std::numeric_limits<size_t>::max()};
    computational_graph.push_back(
        reverseNode<T>(input_values[i], computational_graph.size(),
                       input_fathers, opType::input, &computational_graph));
  }

  // Forward Pass
  std::vector<T> primal_outputs = function(&computational_graph);

  // Backward Pass
  reverseGraphAD<T>(&computational_graph);

  // Extract Gradients
  std::vector<T> gradient(input_values.size());
  for (size_t i = 0; i < input_values.size(); ++i) {
    gradient[i] = computational_graph[i].get_adjoint();
  }

  return {primal_outputs, gradient};
}

} // namespace autodiff
