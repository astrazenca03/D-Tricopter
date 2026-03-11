// main.cpp — Entry point for the tricopter LQR flight controller simulation
//
// Pipeline:
//   1. Load configuration from YAML
//   2. Build control allocation matrix B
//   3. Analyze controllability
//   4. Design LQR controller (trim → linearize → CARE → gain)
//   5. Run closed-loop simulation
//   6. Write results to CSV

#include <iostream>
#include <string>
#include "config.hpp"
#include "control_allocation.hpp"
#include "lqr.hpp"
#include "simulation.hpp"

int main(int argc, char* argv[]) {
    std::string config_path = "config/tricopter_default.yaml";
    if (argc > 1) {
        config_path = argv[1];
    }

    std::cout << "=========================================\n";
    std::cout << "  Asymmetric Servoless Tricopter LQR\n";
    std::cout << "  Flight Controller Simulation\n";
    std::cout << "=========================================\n\n";

    // Step 1: Load configuration
    std::cout << "[1/5] Loading configuration from: " << config_path << "\n";
    tricopter::Config config;
    try {
        config = tricopter::load_config(config_path);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    // Step 2: Build control allocation matrix
    std::cout << "\n[2/5] Building control allocation matrix...\n";
    Eigen::MatrixXd B_alloc = tricopter::build_control_matrix(config.vehicle);

    // Step 3: Analyze controllability
    std::cout << "\n[3/5] Analyzing controllability...\n";
    int rank = tricopter::analyze_controllability(B_alloc);
    if (rank < 6) {
        std::cerr << "\n[WARNING] System may not be fully controllable (rank "
                  << rank << " < 6).\n"
                  << "Proceeding with LQR design — some axes may have poor performance.\n";
    }

    // Step 4: Design LQR controller
    std::cout << "\n[4/5] Designing LQR controller...\n";
    tricopter::LQRResult lqr = tricopter::design_lqr(config, B_alloc);

    if (!lqr.trim.feasible) {
        std::cerr << "\n[WARNING] Hover trim is infeasible. "
                  << "Simulation may not produce meaningful results.\n";
    }

    // Step 5: Run simulation
    std::cout << "\n[5/5] Running closed-loop simulation...\n";
    auto records = tricopter::run_simulation(config, lqr, B_alloc);

    // Write output
    tricopter::write_csv(config.simulation.output_file, records);

    std::cout << "\n=========================================\n";
    std::cout << "  Simulation Complete\n";
    std::cout << "  Output: " << config.simulation.output_file << "\n";
    std::cout << "=========================================\n";

    return 0;
}
