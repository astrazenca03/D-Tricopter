#pragma once
// lqr.hpp — LQR controller for the tricopter.
//
// 1. Finds the hover trim point: motor speeds that produce thrust = mg, zero torque
// 2. Linearizes dynamics about hover to get (A, B_lin) matrices
// 3. Solves the Continuous Algebraic Riccati Equation (CARE) iteratively
// 4. Computes the optimal gain matrix K = R⁻¹ B^T P
//
// The linearized state is 12-dimensional (no quaternion norm constraint):
//   δx = [δpos(3), δvel(3), δφ(3), δω(3)]
// where δφ is a small-angle attitude error (roll, pitch, yaw perturbation).
//
// The input is the perturbation in motor speed squared:
//   δu = [δω₁², δω₂², δω₃²]  (num_rotors elements)

#include <Eigen/Dense>
#include "config.hpp"

namespace tricopter {

constexpr int LIN_STATE_DIM = 12;   // linearized state dimension

// Result of the trim solver
struct TrimResult {
    Eigen::VectorXd omega_sq_hover;   // (num_rotors,) hover motor speed squared
    Eigen::VectorXd omega_hover;      // (num_rotors,) hover motor speed [rad/s]
    bool feasible;                    // true if all omega² > 0
};

// Result of the LQR design
struct LQRResult {
    Eigen::MatrixXd K;               // (num_rotors, 12) gain matrix
    Eigen::MatrixXd P;               // (12, 12) solution to CARE
    Eigen::MatrixXd A;               // (12, 12) linearized state matrix
    Eigen::MatrixXd B_lin;           // (12, num_rotors) linearized input matrix
    TrimResult trim;
};

// Solve for the hover trim: find ω² such that total thrust = mg (along gravity)
// and total torque = 0.
//
// For N rotors, this is a linear system: B * u_hover = [0, 0, mg, 0, 0, 0]^T
// Uses least-squares if the system is over/under-determined.
TrimResult solve_hover_trim(const VehicleConfig& config,
                            const Eigen::MatrixXd& B_alloc);

// Form the linearized A matrix (12×12) about hover.
// Captures the coupling between attitude error and translational acceleration
// through the gravity vector, and the rotational dynamics through J⁻¹.
Eigen::MatrixXd linearize_A(const VehicleConfig& config,
                            const TrimResult& trim);

// Form the linearized B matrix (12×num_rotors) about hover.
// Maps perturbations in motor speed squared to state derivatives.
Eigen::MatrixXd linearize_B(const VehicleConfig& config,
                            const Eigen::MatrixXd& B_alloc);

// Solve the Continuous Algebraic Riccati Equation:
//   A^T P + P A - P B R⁻¹ B^T P + Q = 0
//
// Uses iterative Newton method (Kleinman iteration):
//   1. Start with a stabilizing K₀ (e.g., from LQR on simplified system or zero)
//   2. Solve Lyapunov equation: (A - B K_k)^T P_{k+1} + P_{k+1} (A - B K_k) = -(Q + K_k^T R K_k)
//   3. Update K_{k+1} = R⁻¹ B^T P_{k+1}
//   4. Repeat until convergence
//
// Returns the solution P matrix.
Eigen::MatrixXd solve_care(const Eigen::MatrixXd& A,
                           const Eigen::MatrixXd& B,
                           const Eigen::MatrixXd& Q,
                           const Eigen::MatrixXd& R,
                           int max_iter = 200,
                           double tol = 1e-10);

// Solve the continuous Lyapunov equation: A^T X + X A = -Q
// Uses vectorization: (I ⊗ A^T + A^T ⊗ I) vec(X) = vec(Q)
// This is the direct method — works for small state dimensions.
Eigen::MatrixXd solve_lyapunov(const Eigen::MatrixXd& A,
                               const Eigen::MatrixXd& Q);

// Full LQR design pipeline: trim → linearize → CARE → gain matrix.
LQRResult design_lqr(const Config& config,
                     const Eigen::MatrixXd& B_alloc);

}  // namespace tricopter
