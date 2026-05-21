#pragma once

#include <array>
#include <cstddef>
#include <iostream>
#include <iterator>
#include <limits>
#include <vector>

namespace autodiff {

enum class opType { input, output, sum, sub, mult, div };

template <typename T> class reverseNode {
private:
  T _value{};       // e.g. v_0
  T _adjointGrad{}; // e.g. overbar{v_0}
  size_t _key{};
  mutable std::array<size_t, 2> _fathers{std::numeric_limits<size_t>::max(),
                                         std::numeric_limits<size_t>::max()};
  mutable std::vector<size_t> _children{};
  opType _op_type{};
  std::vector<reverseNode<T>> *_computational_graph{nullptr};

public:
  reverseNode<T>(T input_v, size_t input_key,
                 std::array<size_t, 2> input_fathers,
                 opType input_generatrixOperation,
                 std::vector<reverseNode<T>> *input_graph,
                 std::vector<size_t> input_children = {})
      : _value(input_v), _key(input_key), _fathers(input_fathers),
        _children(input_children), _op_type(input_generatrixOperation),
        _computational_graph(input_graph) {}

  T get_v() const noexcept { return _value; }
  T get_adjoint() const noexcept { return _adjointGrad; }
  size_t get_key() const noexcept { return _key; }
  const std::array<size_t, 2> &get_fathers() const noexcept { return _fathers; }
  const std::vector<size_t> &get_children() const noexcept { return _children; }
  opType get_op() const noexcept { return _op_type; }
  void set_child(size_t &&childID) noexcept { _children.push_back(childID); }
  void set_output(size_t check) noexcept {
    if (check == 1) { // must be an intentional operation
      _op_type = opType::output;
      _fathers = std::array<size_t, 2>{(_computational_graph->size()) - 1,
                                       (_computational_graph->size()) - 1};
      (*_computational_graph)[_computational_graph->size() - 1].set_child(
          _computational_graph->size());
      _key = _computational_graph->size();
    }
  }
  void set_adjoint(T adjointGrad) noexcept { _adjointGrad = adjointGrad; }
  void print() const {
    std::cout << "Value: " << _value << ";\nAdjoint: " << _adjointGrad
              << ";\nKey: " << _key << ";\nFather nodes: [";
    for (std::array<size_t, 2>::iterator it(_fathers.begin());
         it != _fathers.end(); ++it) {
      std::cout << " " << *it;
      if (std::next(it) != _fathers.end()) {
        std::cout << ",";
      }
    }
    std::cout << " ];\nChildren nodes: [";
    for (std::vector<size_t>::iterator it(_children.begin());
         it != _children.end(); ++it) {
      std::cout << " " << *it;
      if (std::next(it) != _children.end()) {
        std::cout << ",";
      }
    }
    std::cout << " ];\n\n";
  }

  // Overloaded operators are non-member friend functions, so that 5.0 * a is
  // possible
  friend reverseNode<T> operator+(const reverseNode<T> &lhs,
                                  const reverseNode<T> &rhs) {
    std::vector<reverseNode<T>> *graph = lhs._computational_graph;
    T sum{lhs._value + rhs._value};
    size_t newKey{graph->size()};
    std::array<size_t, 2> fathers{lhs._key, rhs._key};
    (*graph)[lhs.get_key()]._children.push_back(newKey);
    (*graph)[rhs.get_key()]._children.push_back(newKey);
    opType operation{opType::sum};
    reverseNode<T> newNode(sum, newKey, fathers, operation,
                           lhs._computational_graph);
    graph->push_back(newNode);
    return newNode;
  }

  friend reverseNode<T> operator-(const reverseNode<T> &lhs,
                                  const reverseNode<T> &rhs) {
    std::vector<reverseNode<T>> *graph = lhs._computational_graph;
    T sum{lhs._value - rhs._value};
    size_t newKey{graph->size()};
    std::array<size_t, 2> fathers{lhs._key, rhs._key};
    (*graph)[lhs.get_key()]._children.push_back(newKey);
    (*graph)[rhs.get_key()]._children.push_back(newKey);
    opType operation{opType::sub};
    reverseNode<T> newNode(sum, newKey, fathers, operation,
                           lhs._computational_graph);
    graph->push_back(newNode);
    return newNode;
  }

  friend reverseNode<T> operator*(const reverseNode<T> &lhs,
                                  const reverseNode<T> &rhs) {
    std::vector<reverseNode<T>> *graph = lhs._computational_graph;
    T sum{lhs._value * rhs._value};
    size_t newKey{graph->size()};
    std::array<size_t, 2> fathers{lhs._key, rhs._key};
    (*graph)[lhs.get_key()]._children.push_back(newKey);
    (*graph)[rhs.get_key()]._children.push_back(newKey);
    opType operation{opType::mult};
    reverseNode<T> newNode(sum, newKey, fathers, operation,
                           lhs._computational_graph);
    graph->push_back(newNode);
    return newNode;
  }

  friend reverseNode<T> operator/(const reverseNode<T> &lhs,
                                  const reverseNode<T> &rhs) {
    std::vector<reverseNode<T>> *graph = lhs._computational_graph;
    T sum{lhs._value / rhs._value};
    size_t newKey{graph->size()};
    std::array<size_t, 2> fathers{lhs._key, rhs._key};
    (*graph)[lhs.get_key()]._children.push_back(newKey);
    (*graph)[rhs.get_key()]._children.push_back(newKey);
    opType operation{opType::div};
    reverseNode<T> newNode(sum, newKey, fathers, operation,
                           lhs._computational_graph);
    graph->push_back(newNode);
    return newNode;
  }
};

template <typename T> void reverseGraphAD(std::vector<reverseNode<T>> *graph) {
  for (typename std::vector<reverseNode<T>>::reverse_iterator currentNode =
           graph->rbegin();
       currentNode != graph->rend(); ++currentNode) {
    T adj{0.0};
    if (currentNode->get_op() == opType::output) {
      adj = 1.0;
    } else if (!currentNode->get_children().empty() &&
               (*graph)[currentNode->get_children()[0]].get_op() ==
                   opType::output) {
      adj = 1.0;
    } else {
      for (size_t childID : currentNode->get_children()) {

        // Case 0: child sees current through sum
        if ((*graph)[childID].get_op() == opType::sum) {
          adj += (*graph)[childID].get_adjoint();
        } // Case 1: child sees current through multiplication
        else if ((*graph)[childID].get_op() == opType::mult) {
          std::array<size_t, 2> fathers_of_current_child{
              (*graph)[childID].get_fathers()};
          size_t cofather_index = 0;
          if (fathers_of_current_child[cofather_index] ==
              currentNode->get_key()) {
            cofather_index = 1;
          }
          adj += (*graph)[childID].get_adjoint() *
                 (*graph)[fathers_of_current_child[cofather_index]].get_v();
        } // Case 2: child sees current through subtraction
        else if ((*graph)[childID].get_op() == opType::sub) {
          if (currentNode->get_key() == (*graph)[childID].get_fathers()[0]) {
            adj += (*graph)[childID].get_adjoint();
          } else {
            adj -= (*graph)[childID].get_adjoint();
          }
        } // Case 3: child sees current through division
        else if ((*graph)[childID].get_op() == opType::div) {
          std::array<size_t, 2> fathers_of_current_child{
              (*graph)[childID].get_fathers()};
          size_t cofather_index = 0;
          if (fathers_of_current_child[cofather_index] ==
              currentNode->get_key()) {
            cofather_index = 1;
          }
          if (cofather_index == 1) {
            adj +=
                (*graph)[childID].get_adjoint() *
                (1 /
                 ((*graph)[fathers_of_current_child[cofather_index]].get_v()));
          } else {
            adj -= (*graph)[childID].get_adjoint() *
                   ((*graph)[fathers_of_current_child[cofather_index]].get_v() /
                    ((*graph)[fathers_of_current_child[1]].get_v() *
                     (*graph)[fathers_of_current_child[1]].get_v()));
          }
        }
      }
    }
    currentNode->set_adjoint(adj);
  }
}

template <typename T, typename Func>
std::pair<std::vector<T>, std::vector<T>>
autoGrad(Func function, const std::vector<T> &input_values) {

  std::vector<reverseNode<T>> computational_graph{};
  computational_graph.reserve(10000);

  for (typename std::vector<T>::const_iterator it(input_values.begin());
       it != input_values.end(); ++it) {

    opType input_op_type{opType::input};
    std::array<size_t, 2> input_fathers{std::numeric_limits<size_t>::max(),
                                        std::numeric_limits<size_t>::max()};

    computational_graph.push_back(
        reverseNode<T>(*it, computational_graph.size(), input_fathers,
                       input_op_type, &computational_graph));
  }
  std::vector<T> f{function(&computational_graph)};

  size_t nInputs = input_values.size();

  std::vector<T> gradient(nInputs);

  reverseGraphAD<T>(&computational_graph);

  // SUBROUTINE TO PRINT THE GRAPH (useful caveman debug)
  /*
  for (typename std::vector<reverseNode<T>>::iterator it(
           computational_graph->begin());
       it != computational_graph->end(); ++it) {
    it->print();
  }
  */

  for (int it = 0; it < gradient.size(); ++it) {
    gradient[it] = computational_graph[it].get_adjoint();
  }

  return {f, gradient};
}

} // namespace autodiff
