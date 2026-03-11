#pragma once
// simulation.hpp — Closed-loop simulation of the LQR-controlled tricopter.
//
// Runs the full nonlinear dynamics with the LQR controller in the loop.
// Logs state history to CSV. Applies configurable step disturbances.

#include <Eigen/Dense>
#include <string>
#include <vector>
#include "config.hpp"
#include "lqr.hpp"

namespace tricopter {

// One row of simulation data for logging
struct SimulationRecord {
    double time;
    Eigen::Vector3d position;
    Eigen::Vector3d velocity;
    Eigen::Quaterniond quaternion;
    Eigen::Vector3d euler_deg;       // roll, pitch, yaw in degrees (for display)
    Eigen::Vector3d angular_velocity;
    Eigen::VectorXd motor_commands;  // omega² per rotor
};

// Run the closed-loop simulation.
//
// 1. Initialize state from config (hover + perturbation)
// 2. At each timestep:
//    a. Compute state error relative to hover
//    b. Compute control: u = u_hover - K * δx (clamped to motor limits)
//    c. Apply disturbance if within the disturbance window
//    d. Integrate dynamics via RK4
//    e. Record state
// 3. Write CSV log
//
// Returns the full state history.
std::vector<SimulationRecord> run_simulation(const Config& config,
                                              const LQRResult& lqr,
                                              const Eigen::MatrixXd& B_alloc);

// Write simulation records to a CSV file.
void write_csv(const std::string& filepath,
               const std::vector<SimulationRecord>& records);

}  // namespace tricopter
