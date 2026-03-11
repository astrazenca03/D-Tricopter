// simulation.cpp — Closed-loop LQR simulation with nonlinear dynamics
//
// The simulation loop:
//   1. Compute state error δx = x - x_hover
//   2. Apply LQR control: δu = -K * δx
//   3. Total command: u = u_hover + δu (clamped to motor limits)
//   4. Integrate dynamics with RK4
//   5. Log and repeat

#include "simulation.hpp"
#include "dynamics.hpp"
#include "integrator.hpp"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <cmath>

namespace tricopter {

namespace {

// Compute the 12-dimensional state error for the LQR controller.
//
// δx = [δpos(3), δvel(3), δφ(3), δω(3)]
//
// The attitude error δφ is extracted from the quaternion error:
//   q_error = q_desired⁻¹ ⊗ q_actual
// For small angles, q_error ≈ [1, δφ/2], so δφ ≈ 2 * q_error.vec()
Eigen::VectorXd compute_state_error(const Eigen::VectorXd& state,
                                     const Eigen::VectorXd& state_hover) {
    Eigen::VectorXd error(LIN_STATE_DIM);

    // Position error
    error.segment<3>(0) = state.segment<3>(0) - state_hover.segment<3>(0);

    // Velocity error
    error.segment<3>(3) = state.segment<3>(3) - state_hover.segment<3>(3);

    // Attitude error via quaternion difference
    Eigen::Quaterniond q_actual(state(6), state(7), state(8), state(9));
    Eigen::Quaterniond q_desired(state_hover(6), state_hover(7),
                                  state_hover(8), state_hover(9));
    q_actual.normalize();
    q_desired.normalize();

    // q_error = q_desired.inverse() * q_actual
    Eigen::Quaterniond q_error = q_desired.inverse() * q_actual;

    // Ensure shortest path (scalar part positive)
    if (q_error.w() < 0.0) {
        q_error.coeffs() *= -1.0;
    }

    // Small-angle approximation: δφ ≈ 2 * [q_error.x, q_error.y, q_error.z]
    error.segment<3>(6) = 2.0 * q_error.vec();

    // Angular velocity error
    error.segment<3>(9) = state.segment<3>(10) - state_hover.segment<3>(10);

    return error;
}

}  // namespace

std::vector<SimulationRecord> run_simulation(const Config& config,
                                              const LQRResult& lqr,
                                              const Eigen::MatrixXd& B_alloc) {
    const int N = static_cast<int>(config.vehicle.rotors.size());
    const double dt = config.simulation.dt;
    const double duration = config.simulation.duration;
    const int num_steps = static_cast<int>(duration / dt);

    // --- Build initial state ---
    Eigen::VectorXd state(STATE_DIM);

    // Position
    state.segment<3>(0) = config.simulation.initial_position;

    // Velocity
    state.segment<3>(3) = config.simulation.initial_velocity;

    // Attitude: hover = identity, perturbed by initial Euler angles
    double roll  = config.simulation.initial_roll_deg * M_PI / 180.0;
    double pitch = config.simulation.initial_pitch_deg * M_PI / 180.0;
    double yaw   = config.simulation.initial_yaw_deg * M_PI / 180.0;
    Eigen::Quaterniond q_init = euler_to_quaternion(roll, pitch, yaw);
    state(6) = q_init.w();
    state(7) = q_init.x();
    state(8) = q_init.y();
    state(9) = q_init.z();

    // Angular velocity
    state.segment<3>(10) = config.simulation.initial_angular_velocity;

    // --- Build hover state (reference) ---
    Eigen::VectorXd state_hover(STATE_DIM);
    state_hover.setZero();
    state_hover(6) = 1.0;  // identity quaternion [1, 0, 0, 0]

    // Dynamics function bound to vehicle config
    auto dyn_func = [&config](const Eigen::VectorXd& x,
                              const Eigen::VectorXd& u) {
        return dynamics(x, u, config.vehicle);
    };

    // Pre-allocate records
    std::vector<SimulationRecord> records;
    records.reserve(num_steps + 1);

    double omega_min_sq = config.controller.omega_min * config.controller.omega_min;
    double omega_max_sq = config.controller.omega_max * config.controller.omega_max;

    std::cout << "\n=== Starting Simulation ===\n";
    std::cout << "  Duration: " << duration << " s, dt: " << dt
              << " s, steps: " << num_steps << "\n";
    std::cout << "  Initial attitude: roll=" << config.simulation.initial_roll_deg
              << "°, pitch=" << config.simulation.initial_pitch_deg
              << "°, yaw=" << config.simulation.initial_yaw_deg << "°\n";

    for (int step = 0; step <= num_steps; ++step) {
        double t = step * dt;

        // --- Record current state ---
        SimulationRecord rec;
        rec.time = t;
        rec.position = state.segment<3>(0);
        rec.velocity = state.segment<3>(3);
        rec.quaternion = Eigen::Quaterniond(state(6), state(7), state(8), state(9));
        rec.quaternion.normalize();
        Eigen::Vector3d euler = quaternion_to_euler(rec.quaternion);
        rec.euler_deg = euler * (180.0 / M_PI);
        rec.angular_velocity = state.segment<3>(10);

        // --- Compute control ---
        Eigen::VectorXd delta_x = compute_state_error(state, state_hover);

        // LQR control law: δu = -K * δx
        Eigen::VectorXd delta_u = -lqr.K * delta_x;

        // Total command: u = u_hover + δu
        Eigen::VectorXd u = lqr.trim.omega_sq_hover + delta_u;

        // Clamp to motor limits
        for (int i = 0; i < N; ++i) {
            u(i) = std::max(omega_min_sq, std::min(omega_max_sq, u(i)));
        }

        rec.motor_commands = u;
        records.push_back(rec);

        // --- Apply disturbance ---
        // Disturbance is modeled as an impulsive torque applied for one timestep.
        // We add it as an additional torque term in the control input.
        Eigen::VectorXd u_effective = u;
        bool disturbance_active = false;

        if (config.simulation.disturbance.enabled &&
            std::abs(t - config.simulation.disturbance.time) < dt / 2.0) {
            disturbance_active = true;
        }

        // --- Integrate ---
        if (step < num_steps) {
            if (disturbance_active) {
                // Apply disturbance by modifying the dynamics function for this step
                Eigen::Vector3d dist_torque = config.simulation.disturbance.torque;
                auto dyn_with_dist = [&config, &dist_torque](
                    const Eigen::VectorXd& x, const Eigen::VectorXd& ctrl) {
                    Eigen::VectorXd dxdt = dynamics(x, ctrl, config.vehicle);
                    // Add disturbance torque: J * omega_dot += dist_torque
                    // → omega_dot += J⁻¹ * dist_torque
                    Eigen::Vector3d omega_dot_extra =
                        config.vehicle.inertia.inverse() * dist_torque;
                    dxdt.segment<3>(10) += omega_dot_extra;
                    return dxdt;
                };
                state = rk4_step(dyn_with_dist, state, u_effective, dt);
                std::cout << "  [t=" << std::fixed << std::setprecision(3) << t
                          << "s] Disturbance applied: τ = ["
                          << dist_torque.transpose() << "] N·m\n";
            } else {
                state = rk4_step(dyn_func, state, u_effective, dt);
            }
        }

        // Progress output (every second)
        if (step % static_cast<int>(1.0 / dt) == 0) {
            std::cout << "  t=" << std::fixed << std::setprecision(1) << t
                      << "s  pos=[" << std::setprecision(4)
                      << state.segment<3>(0).transpose()
                      << "]  att=[" << rec.euler_deg.transpose()
                      << "]°\n";
        }
    }

    std::cout << "  Simulation complete.\n";
    return records;
}

void write_csv(const std::string& filepath,
               const std::vector<SimulationRecord>& records) {
    std::ofstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open output file: " + filepath);
    }

    // Header
    file << "time,pos_x,pos_y,pos_z,vel_x,vel_y,vel_z,"
         << "qw,qx,qy,qz,roll_deg,pitch_deg,yaw_deg,"
         << "omega_x,omega_y,omega_z";

    if (!records.empty()) {
        int N = static_cast<int>(records[0].motor_commands.size());
        for (int i = 0; i < N; ++i) {
            file << ",motor_" << i + 1 << "_omega_sq";
        }
    }
    file << "\n";

    // Data
    file << std::fixed << std::setprecision(6);
    for (const auto& rec : records) {
        file << rec.time << ","
             << rec.position.x() << "," << rec.position.y() << ","
             << rec.position.z() << ","
             << rec.velocity.x() << "," << rec.velocity.y() << ","
             << rec.velocity.z() << ","
             << rec.quaternion.w() << "," << rec.quaternion.x() << ","
             << rec.quaternion.y() << "," << rec.quaternion.z() << ","
             << rec.euler_deg.x() << "," << rec.euler_deg.y() << ","
             << rec.euler_deg.z() << ","
             << rec.angular_velocity.x() << "," << rec.angular_velocity.y() << ","
             << rec.angular_velocity.z();

        for (int i = 0; i < rec.motor_commands.size(); ++i) {
            file << "," << rec.motor_commands(i);
        }
        file << "\n";
    }

    std::cout << "[CSV] Wrote " << records.size() << " records to " << filepath << "\n";
}

}  // namespace tricopter
