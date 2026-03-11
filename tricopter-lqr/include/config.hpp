#pragma once
// config.hpp — Configuration data structures and YAML parser
// Loads vehicle geometry, rotor parameters, controller weights, and simulation settings.

#include <Eigen/Dense>
#include <string>
#include <vector>

namespace tricopter {

// Per-rotor physical parameters
struct RotorConfig {
    std::string name;
    Eigen::Vector3d position;       // [m] position from CG in body frame
    Eigen::Vector3d thrust_axis;    // unit vector, thrust direction in body frame
    Eigen::Vector3d spin_axis;      // unit vector, rotor spin axis in body frame
    int spin_direction;             // +1 = CW, -1 = CCW (determines drag torque sign)
    double k_T;                     // [N/(rad/s)²] thrust coefficient
    double k_Q;                     // [N·m/(rad/s)²] drag torque coefficient
};

// Vehicle physical parameters
struct VehicleConfig {
    double mass;                    // [kg]
    Eigen::Matrix3d inertia;        // [kg·m²] full 3x3 inertia tensor (symmetric)
    std::vector<RotorConfig> rotors;
};

// LQR controller tuning
struct ControllerConfig {
    // Q weight matrix diagonal entries (12 states)
    Eigen::Vector3d q_position;
    Eigen::Vector3d q_velocity;
    Eigen::Vector3d q_attitude;
    Eigen::Vector3d q_angular_velocity;

    // R weight matrix diagonal entries (num_rotors inputs)
    Eigen::VectorXd r_weights;

    double omega_min;               // [rad/s] minimum motor speed
    double omega_max;               // [rad/s] maximum motor speed
};

// Impulse disturbance for testing
struct DisturbanceConfig {
    bool enabled;
    double time;                    // [s] when to apply
    Eigen::Vector3d torque;         // [N·m] body-frame torque impulse
};

// Simulation parameters
struct SimulationConfig {
    double dt;                      // [s] integration timestep
    double duration;                // [s] total simulation time

    // Initial conditions (perturbation from hover)
    double initial_roll_deg;
    double initial_pitch_deg;
    double initial_yaw_deg;
    Eigen::Vector3d initial_position;
    Eigen::Vector3d initial_velocity;
    Eigen::Vector3d initial_angular_velocity;

    DisturbanceConfig disturbance;
    std::string output_file;
};

// Top-level configuration container
struct Config {
    VehicleConfig vehicle;
    ControllerConfig controller;
    SimulationConfig simulation;
};

// Parse a YAML config file into a Config struct.
// Throws std::runtime_error on parse failure.
Config load_config(const std::string& filepath);

}  // namespace tricopter
