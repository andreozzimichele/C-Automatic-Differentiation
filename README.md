# C++ Reverse-Mode Automatic Differentiation Engine

A simple, C++ header-only library for both Forward and Reverse-Mode Automatic Differentiation. Designed for educational purposes and raw execution speed, this engine implements a Dual Number (operator overloading) for forward mode, and Computational-Graph architecture for reverse mode to dynamically compute gradients for mathematical models.

### 1. `forwardAutoDiff.hpp` (Forward-Mode AD)
Implements a dual-number class (`forwardNums<T>`) to track derivatives along with primal values during the forward execution.
* **Mathematical Core:** Propagates derivatives forward using the chain rule ($\dot{z} = \dot{x}\frac{\partial z}{\partial x} + \dot{y}\frac{\partial z}{\partial y}$).
* **Performance Profile:** Scales linearly ($O(N)$) with the number of design variables. To compute a full gradient vector for $N$ inputs, the entire mathematical function must be evaluated $N$ distinct times (seeding a different variable's seed $\dot{x}_i = 1$ each time), while it can compute the gradients of $M$ outputs with respect to one input in a single forward pass.

### 2. `reverseAutoDiff.hpp` (Eager Dynamic Tracing / Pull Architecture)
Implements an eager-mode Reverse AD engine using operator overloading. 
* **The "Pull" Mechanism:** As math functions execute, overloaded operators dynamically trace the graph by adding nodes and logging execution relationships into a `_children` vector. The backward pass sweeps backward from the output, searching through each node's children to "pull" accumulated gradients down to the independent inputs.
* **Control Flow:** Excellent flexibility; seamlessly handles native C++ branching (`if` statements) and dynamic graphs at runtime.

### 3. `wengertTapeAD.hpp` (Optimized Push / Tape Architecture)
A more high-performance implementation of Reverse AD that eliminates the overhead of the dynamic pull architecture.
* **The "Push" Mechanism (The Wengert List):** Completely removes the `_children` vector and dynamic memory heap tracking per node. Instead, math operations are written directly to a flat, contiguous array. 
* **Zero Hidden Heap Allocations:** Uses a fixed-size `std::array` to store exactly two parent nodes (`_fathers`). During the single backward sweep, each node evaluates its local operation and directly "pushes" its accumulated adjoint down to its parents. 
* **Cache Friendly:** Linear iteration over a contiguous vector leverages CPU cache line pre-fetching, running order-of-magnitude faster than pointer-chasing graph traversals.

---

## Quick Start (e.g. for Reverse-Mode)

```cpp
#include "reverseAutoDiff.hpp"
#include <iostream>
#include <vector>

using varType = double;

// Define your mathematical function
std::vector<varType> myFunction(std::vector<autodiff::reverseNode<varType>>* graph) {
    auto a = (*graph)[0];
    auto b = (*graph)[1];
    
    auto f = a * a + b * 3.0; // The math is traced automatically
    
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
