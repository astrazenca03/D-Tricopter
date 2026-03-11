#pragma once
// dynamics.hpp — Full 6DOF nonlinear rigid body dynamics for an N-rotor vehicle.
//
// State vector x (13 elements):
//   x = [pos_x, pos_y, pos_z,           // inertial position [m]
//        vel_x, vel_y, vel_z,            // inertial velocity [m/s]
//        q_w, q_x, q_y, q_z,            // attitude quaternion (scalar-first)
//        omega_x, omega_y, omega_z]      // body angular velocity [rad/s]
//
// Control input u (num_rotors elements):
//   u = [ω₁², ω₂², ..., ωₙ²]           // motor speed squared [(rad/s)²]

#include <Eigen/Dense>
#include "config.hpp"

namespace tricopter {

constexpr int STATE_DIM = 13;

// Full 6DOF dynamics: computes dx/dt given current state and control input.
//
// Translational: m*a = R * sum(k_Ti * ui * thrust_axis_i) + m*g_inertial
// Rotational:    J*omega_dot = tau_total - omega x (J*omega)
// Quaternion:    q_dot = 0.5 * q ⊗ [0, omega]
//
// Arguments:
//   state  — (13,) state vector
//   u      — (num_rotors,) motor speed squared
//   config — vehicle configuration
//
// Returns:
//   (13,) state derivative vector
Eigen::VectorXd dynamics(const Eigen::VectorXd& state,
                         const Eigen::VectorXd& u,
                         const VehicleConfig& config);

// Extract rotation matrix (3x3) from quaternion portion of state vector.
// Quaternion convention: scalar-first [w, x, y, z].
Eigen::Matrix3d quaternion_to_rotation(const Eigen::Quaterniond& q);

// Convert Euler angles [roll, pitch, yaw] in radians to quaternion.
// Uses ZYX convention (yaw * pitch * roll).
Eigen::Quaterniond euler_to_quaternion(double roll, double pitch, double yaw);

// Convert quaternion to Euler angles [roll, pitch, yaw] in radians.
Eigen::Vector3d quaternion_to_euler(const Eigen::Quaterniond& q);

// Normalize the quaternion portion of a state vector in-place.
void normalize_quaternion(Eigen::VectorXd& state);

}  // namespace tricopter
