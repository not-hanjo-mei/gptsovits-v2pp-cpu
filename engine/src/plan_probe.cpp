// plan_probe: compile-only probe for a graph on the free engine.
// Prints the graph's inputs/outputs and the compiled step count,
// or the compile error. Used to survey op coverage for graphs the
// engine has not run yet (runtime reference extraction models).
//
//   plan_probe GRAPH.onnx [GRAPH2.onnx ...]

#include <cstdio>
#include <iostream>

#include "cpu.h"
#include "free_engine.h"
#include "onnx_model.h"

int main(int argc, char** argv) {
    if (argc < 2) { std::cerr << "usage: plan_probe GRAPH...\n"; return 2; }
    cpu::log_features("plan_probe");
    int rc = 0;
    for (int i = 1; i < argc; i++) {
        std::string path = argv[i];
        try {
            onnxm::Graph g = onnxm::load_model(path);
            std::printf("%s: %zu nodes; inputs:", path.c_str(), g.nodes.size());
            for (auto& in : g.inputs) std::printf(" %s", in.name.c_str());
            std::printf("; outputs:");
            for (auto& out : g.outputs) std::printf(" %s", out.name.c_str());
            std::printf("\n");
            auto plan = freeeng::Plan::compile(g, "probe");
            std::printf("  COMPILE OK: %zu steps\n", plan->step_count());
        } catch (const std::exception& e) {
            std::printf("  COMPILE FAIL: %s\n", e.what());
            rc = 1;
        }
    }
    return rc;
}
