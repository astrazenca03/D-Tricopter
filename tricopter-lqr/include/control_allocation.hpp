#pragma once
// control_allocation.hpp — Computes the 6×N control effectiveness matrix B.
//
// B maps motor speed squared [ω₁², ω₂², ..., ωₙ²] to body-frame wrench
// [Fx, Fy, Fz, Tx, Ty, Tz]:
//
//   Column i of B:
//     Force rows:  k_Ti * thrust_axis_i
//     Torque rows: (r_i × k_Ti * thrust_axis_i) + k_Qi * spin_dir_i * spin_axis_i
//
// The force contribution comes from thrust along each rotor's thrust axis.
// The torque contribution has two parts:
//   1. Moment arm: cross product of rotor position with thrust force
//   2. Reaction torque: aerodynamic drag torque about the spin axis

#include <Eigen/Dense>
#include "config.hpp"

namespace tricopter {

// Build the 6×N control effectiveness matrix.
// N = number of rotors (config.rotors.size())
//
// Returns: (6, N) matrix where rows are [Fx, Fy, Fz, Tx, Ty, Tz]
Eigen::MatrixXd build_control_matrix(const VehicleConfig& config);

// Analyze controllability of the B matrix.
// Prints rank, singular values, and flags weakly controllable axes.
// Returns the rank of B.
int analyze_controllability(const Eigen::MatrixXd& B);

}  // namespace tricopter
