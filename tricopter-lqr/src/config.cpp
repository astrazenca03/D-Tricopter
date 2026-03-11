// config.cpp — YAML configuration parser
// Loads vehicle geometry, rotor parameters, controller weights, and simulation settings.

#include "config.hpp"
#include <yaml-cpp/yaml.h>
#include <stdexcept>
#include <iostream>

namespace tricopter {

namespace {

// Helper: read a 3-element YAML sequence into an Eigen::Vector3d
Eigen::Vector3d read_vec3(const YAML::Node& node) {
    if (!node.IsSequence() || node.size() != 3) {
        throw std::runtime_error("Expected a 3-element sequence");
    }
    return Eigen::Vector3d(node[0].as<double>(),
                           node[1].as<double>(),
                           node[2].as<double>());
}

}  // namespace

Config load_config(const std::string& filepath) {
    YAML::Node root;
    try {
        root = YAML::LoadFile(filepath);
    } catch (const YAML::Exception& e) {
        throw std::runtime_error("Failed to load config file '" + filepath +
                                 "': " + e.what());
    }

    Config cfg;

    // --- Vehicle ---
    auto vehicle = root["vehicle"];
    cfg.vehicle.mass = vehicle["mass"].as<double>();

    auto J = vehicle["inertia_tensor"];
    double Jxx = J["Jxx"].as<double>();
    double Jxy = J["Jxy"].as<double>();
    double Jxz = J["Jxz"].as<double>();
    double Jyy = J["Jyy"].as<double>();
    double Jyz = J["Jyz"].as<double>();
    double Jzz = J["Jzz"].as<double>();

    // Symmetric 3x3 inertia tensor [kg·m²]
    cfg.vehicle.inertia << Jxx, Jxy, Jxz,
                           Jxy, Jyy, Jyz,
                           Jxz, Jyz, Jzz;

    // --- Rotors ---
    auto rotors = root["rotors"];
    for (const auto& r : rotors) {
        RotorConfig rc;
        rc.name           = r["name"].as<std::string>();
        rc.position       = read_vec3(r["position"]);
        rc.thrust_axis    = read_vec3(r["thrust_axis"]).normalized();
        rc.spin_axis      = read_vec3(r["spin_axis"]).normalized();
        rc.spin_direction = r["spin_direction"].as<int>();
        rc.k_T            = r["k_T"].as<double>();
        rc.k_Q            = r["k_Q"].as<double>();
        cfg.vehicle.rotors.push_back(rc);
    }

    // --- Controller ---
    auto ctrl = root["controller"];
    cfg.controller.q_position         = read_vec3(ctrl["Q_weights"]["position"]);
    cfg.controller.q_velocity         = read_vec3(ctrl["Q_weights"]["velocity"]);
    cfg.controller.q_attitude         = read_vec3(ctrl["Q_weights"]["attitude"]);
    cfg.controller.q_angular_velocity = read_vec3(ctrl["Q_weights"]["angular_velocity"]);

    auto r_node = ctrl["R_weights"];
    int num_rotors = static_cast<int>(cfg.vehicle.rotors.size());
    cfg.controller.r_weights.resize(num_rotors);
    for (int i = 0; i < num_rotors; ++i) {
        cfg.controller.r_weights(i) = r_node[i].as<double>();
    }

    cfg.controller.omega_min = ctrl["omega_min"].as<double>();
    cfg.controller.omega_max = ctrl["omega_max"].as<double>();

    // --- Simulation ---
    auto sim = root["simulation"];
    cfg.simulation.dt       = sim["dt"].as<double>();
    cfg.simulation.duration = sim["duration"].as<double>();

    cfg.simulation.initial_roll_deg  = sim["initial_roll_deg"].as<double>();
    cfg.simulation.initial_pitch_deg = sim["initial_pitch_deg"].as<double>();
    cfg.simulation.initial_yaw_deg   = sim["initial_yaw_deg"].as<double>();

    cfg.simulation.initial_position         = read_vec3(sim["initial_position"]);
    cfg.simulation.initial_velocity         = read_vec3(sim["initial_velocity"]);
    cfg.simulation.initial_angular_velocity = read_vec3(sim["initial_angular_velocity"]);

    auto dist = sim["disturbance"];
    cfg.simulation.disturbance.enabled = dist["enabled"].as<bool>();
    cfg.simulation.disturbance.time    = dist["time"].as<double>();
    cfg.simulation.disturbance.torque  = read_vec3(dist["torque"]);

    cfg.simulation.output_file = sim["output_file"].as<std::string>();

    std::cout << "[Config] Loaded " << num_rotors << " rotors, mass = "
              << cfg.vehicle.mass << " kg\n";
    return cfg;
}

}  // namespace tricopter
