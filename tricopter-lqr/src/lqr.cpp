// lqr.cpp — LQR controller design for the asymmetric tricopter.
//
// Pipeline:
//   1. Solve for hover trim (motor speeds that balance gravity with zero torque)
//   2. Linearize 6DOF dynamics about hover
//   3. Solve Continuous Algebraic Riccati Equation (CARE) via Kleinman iteration
//   4. Compute optimal gain K = R⁻¹ B^T P

#include "lqr.hpp"
#include "dynamics.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <stdexcept>

namespace tricopter {

TrimResult solve_hover_trim(const VehicleConfig& config,
                            const Eigen::MatrixXd& B_alloc) {
    // At hover, we need:
    //   B_alloc * u_hover = [0, 0, m*g, 0, 0, 0]^T
    //
    // B_alloc is (6, N). For N=3, this is a square system.
    // For N≠3, use least-squares.

    const int N = static_cast<int>(config.rotors.size());
    Eigen::VectorXd wrench_desired(6);
    wrench_desired << 0.0, 0.0, config.mass * 9.81, 0.0, 0.0, 0.0;

    TrimResult result;

    // Remove rows that are all-zero (uncontrollable axes like Fx, Fy for
    // all-vertical-thrust configurations). This avoids polluting the
    // least-squares with trivially satisfied constraints.
    std::vector<int> active_rows;
    for (int i = 0; i < 6; ++i) {
        if (B_alloc.row(i).norm() > 1e-12) {
            active_rows.push_back(i);
        }
    }

    int n_active = static_cast<int>(active_rows.size());
    Eigen::MatrixXd B_active(n_active, N);
    Eigen::VectorXd w_active(n_active);
    for (int i = 0; i < n_active; ++i) {
        B_active.row(i) = B_alloc.row(active_rows[i]);
        w_active(i) = wrench_desired(active_rows[i]);
    }

    // Solve the reduced system via SVD (handles square, over, or under-determined)
    result.omega_sq_hover =
        B_active.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV)
            .solve(w_active);

    // Check feasibility: all ω² must be non-negative
    result.feasible = true;
    result.omega_hover.resize(N);
    for (int i = 0; i < N; ++i) {
        if (result.omega_sq_hover(i) < 0.0) {
            result.feasible = false;
            result.omega_hover(i) = 0.0;
        } else {
            result.omega_hover(i) = std::sqrt(result.omega_sq_hover(i));
        }
    }

    // Print trim info
    std::cout << "\n=== Hover Trim Solution ===\n";
    for (int i = 0; i < N; ++i) {
        std::cout << "  Rotor " << i + 1 << " (" << config.rotors[i].name
                  << "): ω² = " << std::fixed << std::setprecision(2)
                  << result.omega_sq_hover(i)
                  << " (rad/s)² → ω = " << result.omega_hover(i)
                  << " rad/s\n";
    }

    // Verify: residual wrench
    Eigen::VectorXd residual = B_alloc * result.omega_sq_hover - wrench_desired;
    std::cout << "  Residual wrench norm: " << std::scientific
              << residual.norm() << "\n";

    if (!result.feasible) {
        std::cerr << "[WARNING] Trim solution has negative ω² — "
                  << "hover may not be achievable with this configuration.\n";
    }

    return result;
}

Eigen::MatrixXd linearize_A(const VehicleConfig& config,
                            const TrimResult& trim) {
    // Linearized state: δx = [δpos(3), δvel(3), δφ(3), δω(3)]  (12 states)
    //
    // The A matrix captures:
    //   d(δpos)/dt = δvel                         → A[0:3, 3:6] = I
    //   d(δvel)/dt ≈ -R_hover * [F_hover]× / m * δφ + ...
    //     For hover with all thrust along z, the gravity/thrust coupling gives:
    //     d(δvx)/dt ≈ -g * δθ (pitch → x acceleration)
    //     d(δvy)/dt ≈  g * δφ (roll → y acceleration)
    //   d(δφ)/dt = δω                             → A[6:9, 9:12] = I
    //   d(δω)/dt = J⁻¹ * (∂τ/∂φ) * δφ + J⁻¹ * (∂τ/∂ω) * δω
    //     At hover with zero angular velocity, ∂τ/∂ω ≈ 0 (gyroscopic terms vanish)
    //     and ∂τ/∂φ ≈ 0 (torques don't depend on attitude at first order)

    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(LIN_STATE_DIM, LIN_STATE_DIM);

    // d(δpos)/dt = δvel
    A.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity();

    // d(δvel)/dt coupling with attitude error
    // At hover, total thrust = mg along body z-axis.
    // Perturbation in attitude rotates the thrust vector away from vertical.
    // For small angles δφ = [δroll, δpitch, δyaw]:
    //   R(δφ) ≈ I + [δφ]×
    //   Thrust in inertial ≈ (I + [δφ]×) * [0, 0, mg]
    //   = [0, 0, mg] + δφ × [0, 0, mg]
    //   = [0, 0, mg] + [mg*δpitch, -mg*δroll, 0]  (approximately)
    //
    // But we need to be more precise for asymmetric vehicles.
    // Total hover force in body frame:
    double g = 9.81;
    int N = static_cast<int>(config.rotors.size());
    Eigen::Vector3d F_hover_body = Eigen::Vector3d::Zero();
    for (int i = 0; i < N; ++i) {
        F_hover_body += config.rotors[i].k_T * trim.omega_sq_hover(i) *
                        config.rotors[i].thrust_axis;
    }

    // At hover, R = I (identity), so F_hover_body ≈ [0, 0, mg].
    // The skew-symmetric matrix [F_hover_body]× maps attitude perturbation
    // to force perturbation in inertial frame:
    //   δF_inertial = [F_hover_body]× * δφ / m  (divided by mass for accel)
    // Actually: δa = (1/m) * [δφ]× * F_hover_body = -(1/m) * [F_hover_body]× * δφ
    Eigen::Matrix3d F_skew;
    F_skew <<  0,                -F_hover_body(2),  F_hover_body(1),
               F_hover_body(2),  0,                -F_hover_body(0),
              -F_hover_body(1),  F_hover_body(0),  0;

    // d(δvel)/dt += -(1/m) * [F_hover]× * δφ
    // But the sign convention: rotating body CW (positive angle) tilts thrust left.
    // The correct linearization for v_dot = (1/m)*R*F + g:
    //   δv_dot = (1/m) * [δR] * F_hover = (1/m) * R_hover * [δφ]× * F_hover
    // At hover R_hover = I, so:
    //   δv_dot = (1/m) * skew(δφ) * F_hover
    // In matrix form with δφ as state:
    //   A_vel_att = (1/m) * [0, F_z, -F_y; -F_z, 0, F_x; F_y, -F_x, 0]  (negative of [F]×)
    A.block<3, 3>(3, 6) = -F_skew / config.mass;

    // d(δφ)/dt = δω (small angle: attitude rate ≈ angular velocity)
    A.block<3, 3>(6, 9) = Eigen::Matrix3d::Identity();

    // d(δω)/dt: at hover with ω=0, the gyroscopic term -J⁻¹(ω × Jω) vanishes.
    // The only ω-dependent term at hover is zero since ω_hover = 0.
    // So A[9:12, 9:12] = 0 for the basic linearization.
    // (Damping would come from aerodynamic effects, not modeled here.)

    return A;
}

Eigen::MatrixXd linearize_B(const VehicleConfig& config,
                            const Eigen::MatrixXd& B_alloc) {
    // Linearized B matrix: maps δu (perturbation in ω²) to linearized state derivative
    //
    // B_lin is (12, N):
    //   Rows 0-2 (δpos_dot):   0 (motor speeds don't directly affect position)
    //   Rows 3-5 (δvel_dot):   (1/m) * B_alloc[0:3, :] (force → accel, at hover R=I)
    //   Rows 6-8 (δφ_dot):     0 (motor speeds don't directly affect attitude angle)
    //   Rows 9-11 (δω_dot):    J⁻¹ * B_alloc[3:6, :] (torque → angular accel)

    const int N = static_cast<int>(config.rotors.size());
    Eigen::MatrixXd B_lin = Eigen::MatrixXd::Zero(LIN_STATE_DIM, N);

    // Force → translational acceleration (at hover, R = I)
    B_lin.block(3, 0, 3, N) = B_alloc.block(0, 0, 3, N) / config.mass;

    // Torque → angular acceleration
    Eigen::Matrix3d J_inv = config.inertia.inverse();
    B_lin.block(9, 0, 3, N) = J_inv * B_alloc.block(3, 0, 3, N);

    return B_lin;
}

Eigen::MatrixXd solve_lyapunov(const Eigen::MatrixXd& A,
                               const Eigen::MatrixXd& Q) {
    // Solve continuous Lyapunov equation: A^T X + X A = -Q
    //
    // Vectorize: (I⊗A^T + A^T⊗I) vec(X) = -vec(Q)
    // where ⊗ is the Kronecker product.

    const int n = static_cast<int>(A.rows());
    const int n2 = n * n;

    Eigen::MatrixXd At = A.transpose();

    // Build Kronecker sum: (I⊗A^T + A^T⊗I)
    // Using the identity: vec(A^T X + X A) = (I⊗A^T + A^T⊗I) vec(X)
    Eigen::MatrixXd kron_sum = Eigen::MatrixXd::Zero(n2, n2);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int r = 0; r < n; ++r) {
                for (int s = 0; s < n; ++s) {
                    int row = i * n + r;
                    int col = j * n + s;
                    // I⊗A^T contribution: nonzero only when i==j
                    if (i == j) {
                        kron_sum(row, col) += At(r, s);
                    }
                    // A^T⊗I contribution: nonzero only when r==s
                    if (r == s) {
                        kron_sum(row, col) += At(i, j);
                    }
                }
            }
        }
    }

    // Vectorize Q (column-major) with negative sign: RHS = -vec(Q)
    Eigen::VectorXd vec_negQ(n2);
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            vec_negQ(j * n + i) = -Q(i, j);
        }
    }

    // Solve the linear system: kron_sum * vec(X) = -vec(Q)
    Eigen::VectorXd vec_X = kron_sum.fullPivLu().solve(vec_negQ);

    // Reshape vec_X back to matrix (column-major)
    Eigen::MatrixXd X(n, n);
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            X(i, j) = vec_X(j * n + i);
        }
    }

    // Symmetrize (numerical cleanup)
    X = 0.5 * (X + X.transpose());

    return X;
}

Eigen::MatrixXd solve_care(const Eigen::MatrixXd& A,
                           const Eigen::MatrixXd& B,
                           const Eigen::MatrixXd& Q,
                           const Eigen::MatrixXd& R,
                           int max_iter,
                           double tol) {
    // Solve CARE: A^T P + P A - P B R⁻¹ B^T P + Q = 0
    //
    // Method: Matrix Sign Function applied to the Hamiltonian matrix.
    //
    // The Hamiltonian is:
    //   H = [A,    -B R⁻¹ B^T]
    //       [-Q,   -A^T       ]
    //
    // The matrix sign function S = sign(H) satisfies S² = I and
    // converges via the iteration:
    //   Z₀ = H
    //   Z_{k+1} = 0.5 * (Z_k + Z_k⁻¹)
    //
    // The stable invariant subspace is: W = (I - S) / 2
    // Then P is extracted from W = [W1; W2] as P = W2 * W1⁻¹
    // (where W1, W2 are the n×n upper and lower blocks).

    const int n = static_cast<int>(A.rows());

    Eigen::MatrixXd R_inv = R.inverse();
    Eigen::MatrixXd BRinvBt = B * R_inv * B.transpose();

    std::cout << "\n=== CARE Solver (Matrix Sign Function) ===\n";

    // Form Hamiltonian matrix H (2n × 2n)
    Eigen::MatrixXd H(2 * n, 2 * n);
    H.block(0, 0, n, n) = A;
    H.block(0, n, n, n) = -BRinvBt;
    H.block(n, 0, n, n) = -Q;
    H.block(n, n, n, n) = -A.transpose();

    // Matrix sign function iteration with determinant scaling for faster convergence
    Eigen::MatrixXd Z = H;
    Eigen::MatrixXd I2n = Eigen::MatrixXd::Identity(2 * n, 2 * n);

    for (int iter = 0; iter < max_iter; ++iter) {
        Eigen::MatrixXd Z_inv = Z.fullPivLu().inverse();

        // Determinant scaling (accelerates convergence):
        // γ = |det(Z)|^(-1/(2n))
        double det_abs = std::abs(Z.determinant());
        double gamma = 1.0;
        if (det_abs > 1e-300 && std::isfinite(det_abs)) {
            gamma = std::pow(det_abs, -1.0 / (2.0 * n));
            // Clamp gamma to avoid extreme scaling
            gamma = std::max(0.5, std::min(2.0, gamma));
        }

        Eigen::MatrixXd Z_new = 0.5 * (gamma * Z + Z_inv / gamma);

        double delta = (Z_new - Z).norm();
        if (iter % 5 == 0 || delta < tol * 100) {
            std::cout << "  Iter " << std::setw(3) << iter
                      << ": ||ΔZ|| = " << std::scientific << delta << "\n";
        }

        Z = Z_new;

        // Check convergence: Z² should approach I
        if (delta < tol * 100) {
            Eigen::MatrixXd ZZ = Z * Z;
            double zz_err = (ZZ - I2n).norm();
            std::cout << "  ||Z² - I|| = " << std::scientific << zz_err << "\n";
            if (zz_err < tol * 10) {
                std::cout << "  Sign function converged at iter " << iter + 1 << ".\n";
                break;
            }
        }
    }

    // Extract P directly from the sign matrix blocks.
    // Partition sign(H) = Z = [Z11, Z12; Z21, Z22].
    // For a Hamiltonian, the sign matrix has the symplectic structure.
    // The CARE solution is: P = Z21 * (Z11 - I)⁻¹
    // Equivalently: P = -(I + Z22)⁻¹ * Z21
    Eigen::MatrixXd Z11 = Z.block(0, 0, n, n);
    Eigen::MatrixXd Z21 = Z.block(n, 0, n, n);
    Eigen::MatrixXd Z22 = Z.block(n, n, n, n);

    Eigen::MatrixXd P;

    // Try both formulas and pick the one with better conditioning
    Eigen::MatrixXd M1 = Z11 - Eigen::MatrixXd::Identity(n, n);
    Eigen::MatrixXd M2 = Eigen::MatrixXd::Identity(n, n) + Z22;

    double cond1 = M1.fullPivLu().rcond();
    double cond2 = M2.fullPivLu().rcond();
    std::cout << "  Block extraction: rcond(Z11-I)=" << cond1
              << ", rcond(I+Z22)=" << cond2 << "\n";

    if (cond1 > cond2 && cond1 > 1e-14) {
        P = Z21 * M1.fullPivLu().inverse();
        std::cout << "  Extracted P via Z21*(Z11-I)⁻¹\n";
    } else if (cond2 > 1e-14) {
        P = -M2.fullPivLu().inverse() * Z21;
        std::cout << "  Extracted P via -(I+Z22)⁻¹*Z21\n";
    } else {
        // Both nearly singular — use SVD of W = (I-Z)/2
        std::cout << "  Block formulas ill-conditioned, using SVD of W.\n";
        Eigen::MatrixXd W = 0.5 * (I2n - Z);
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(W, Eigen::ComputeFullU);
        Eigen::MatrixXd U_stable = svd.matrixU().leftCols(n);
        Eigen::MatrixXd U1 = U_stable.block(0, 0, n, n);
        Eigen::MatrixXd U2 = U_stable.block(n, 0, n, n);
        Eigen::JacobiSVD<Eigen::MatrixXd> svd_u1(
            U1, Eigen::ComputeFullU | Eigen::ComputeFullV);
        P = U2 * svd_u1.solve(Eigen::MatrixXd::Identity(n, n));
    }

    // Symmetrize
    P = 0.5 * (P + P.transpose());

    // Verify CARE residual
    Eigen::MatrixXd residual = A.transpose() * P + P * A -
                                P * BRinvBt * P + Q;
    double res_norm = residual.norm();
    std::cout << "  CARE residual: " << std::scientific << res_norm << "\n";

    // Check if the result gives a stabilizing controller
    Eigen::MatrixXd K = R_inv * B.transpose() * P;
    Eigen::MatrixXd A_cl = A - B * K;
    Eigen::EigenSolver<Eigen::MatrixXd> eig_check(A_cl);
    double max_real_cl = eig_check.eigenvalues().real().maxCoeff();
    std::cout << "  Closed-loop max Re(λ): " << max_real_cl << "\n";

    if (max_real_cl < 0.0) {
        // Sign function result is stabilizing — refine with Kleinman if needed
        if (res_norm > 1.0) {
            std::cout << "  Refining with Kleinman iteration...\n";
            Eigen::MatrixXd P_prev;
            for (int iter = 0; iter < max_iter; ++iter) {
                P_prev = P;
                A_cl = A - B * K;
                Eigen::MatrixXd Q_lyap = Q + K.transpose() * R * K;
                P = solve_lyapunov(A_cl, Q_lyap);
                K = R_inv * B.transpose() * P;
                double delta = (P - P_prev).norm();
                if (iter % 10 == 0 || delta < tol) {
                    std::cout << "    Iter " << std::setw(4) << iter
                              << ": ||ΔP|| = " << std::scientific << delta << "\n";
                }
                if (delta < tol) {
                    std::cout << "    Converged after " << iter + 1 << " iterations.\n";
                    break;
                }
            }
            P = 0.5 * (P + P.transpose());
        }
        return P;
    }

    // Sign function didn't give a stabilizing result.
    // Fall back to regularized Kleinman iteration.
    std::cout << "  Sign function P not stabilizing, trying regularized Kleinman...\n";
    double epsilon = 0.01;
    Eigen::MatrixXd A_reg = A - epsilon * Eigen::MatrixXd::Identity(n, n);
    K = Eigen::MatrixXd::Zero(B.cols(), n);

    Eigen::MatrixXd P_prev;
    for (int iter = 0; iter < max_iter; ++iter) {
        P_prev = P;
        Eigen::MatrixXd A_cl_k = A_reg - B * K;
        Eigen::MatrixXd Q_lyap = Q + K.transpose() * R * K;
        P = solve_lyapunov(A_cl_k, Q_lyap);
        K = R_inv * B.transpose() * P;
        double delta = (P - P_prev).norm();
        if (iter % 10 == 0 || delta < tol) {
            std::cout << "    Iter " << std::setw(4) << iter
                      << ": ||ΔP|| = " << std::scientific << delta << "\n";
        }
        if (delta < tol) {
            std::cout << "    Converged after " << iter + 1 << " iterations.\n";
            P = 0.5 * (P + P.transpose());
            return P;
        }
    }

    P = 0.5 * (P + P.transpose());
    std::cerr << "[WARNING] CARE solver did not fully converge.\n";
    return P;
}

LQRResult design_lqr(const Config& config,
                     const Eigen::MatrixXd& B_alloc) {
    LQRResult result;
    const int N = static_cast<int>(config.vehicle.rotors.size());

    // Step 1: Solve for hover trim
    result.trim = solve_hover_trim(config.vehicle, B_alloc);

    // Step 2: Linearize about hover
    result.A = linearize_A(config.vehicle, result.trim);
    result.B_lin = linearize_B(config.vehicle, B_alloc);

    std::cout << "\n=== Linearized System ===\n";
    std::cout << "A matrix (12x12):\n" << std::fixed << std::setprecision(4)
              << result.A << "\n\n";
    std::cout << "B_lin matrix (12x" << N << "):\n" << result.B_lin << "\n";

    // Step 3: Build Q and R weight matrices
    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(LIN_STATE_DIM, LIN_STATE_DIM);
    Q.diagonal().segment<3>(0)  = config.controller.q_position;
    Q.diagonal().segment<3>(3)  = config.controller.q_velocity;
    Q.diagonal().segment<3>(6)  = config.controller.q_attitude;
    Q.diagonal().segment<3>(9)  = config.controller.q_angular_velocity;

    Eigen::MatrixXd R_diag = config.controller.r_weights.asDiagonal();

    // Step 4: Input scaling for numerical conditioning
    //
    // The raw inputs u = ω² are O(300,000) while B_lin entries are O(1e-5).
    // This creates a poorly conditioned CARE problem.
    //
    // We define normalized inputs: ũ = δu / u_scale, where u_scale = ω²_hover.
    // The scaled B matrix is: B̃ = B_lin * diag(u_scale)
    // The R weights are interpreted as applying to the NORMALIZED perturbation ũ,
    // which represents fractional changes in motor speed squared relative to hover.
    // This gives R̃ = R (user weights directly, since ũ is O(1)).
    //
    // This means the cost function penalizes (δω² / ω²_hover)², which is the
    // physically meaningful quantity (fractional control effort).
    //
    // The gain for normalized inputs is: K̃ = R⁻¹ B̃^T P
    // The gain for original inputs is: K = K̃ * S_u⁻¹ → u = u_hover - K * δx
    // Equivalently: K = R⁻¹ B̃^T P S_u⁻¹ = R⁻¹ S_u B_lin^T P S_u⁻¹

    Eigen::VectorXd u_scale = result.trim.omega_sq_hover;
    Eigen::MatrixXd S_u = u_scale.asDiagonal();              // (N x N)
    Eigen::MatrixXd S_u_inv = u_scale.cwiseInverse().asDiagonal();

    Eigen::MatrixXd B_scaled = result.B_lin * S_u;            // (12 x N)
    Eigen::MatrixXd R_scaled = R_diag;                        // (N x N) — user weights on normalized input

    std::cout << "\nB_scaled (input-normalized) (12x" << N << "):\n"
              << std::fixed << std::setprecision(6) << B_scaled << "\n";
    std::cout << "Input scaling factors: " << u_scale.transpose() << "\n";

    // Step 5: Solve CARE with scaled system: A^T P + P A - P B̃ R⁻¹ B̃^T P + Q = 0
    result.P = solve_care(result.A, B_scaled, Q, R_scaled);

    // Step 6: Compute gain for original (unscaled) inputs
    // From normalized: ũ = -K̃ * δx → δu = S_u * ũ = -S_u * K̃ * δx
    // K̃ = R⁻¹ B̃^T P
    // K = S_u * K̃ = S_u * R⁻¹ * B̃^T * P = S_u * R⁻¹ * S_u * B_lin^T * P
    Eigen::MatrixXd K_tilde = R_scaled.inverse() * B_scaled.transpose() * result.P;  // (N x 12)
    result.K = S_u * K_tilde;  // (N x 12) — gain for original δω² inputs

    std::cout << "\n=== LQR Gain Matrix K (" << N << "x12) ===\n";
    std::cout << std::fixed << std::setprecision(4) << result.K << "\n";

    // Verify closed-loop stability
    Eigen::MatrixXd A_cl = result.A - result.B_lin * result.K;
    Eigen::EigenSolver<Eigen::MatrixXd> eig(A_cl);
    std::cout << "\nClosed-loop eigenvalues (Re, Im):\n";
    for (int i = 0; i < LIN_STATE_DIM; ++i) {
        std::cout << "  λ" << i + 1 << " = ("
                  << std::setw(10) << eig.eigenvalues()(i).real() << ", "
                  << std::setw(10) << eig.eigenvalues()(i).imag() << ")\n";
    }

    double max_real = eig.eigenvalues().real().maxCoeff();
    if (max_real < 0.0) {
        std::cout << "[OK] Closed-loop system is stable (max Re(λ) = "
                  << max_real << ")\n";
    } else {
        std::cerr << "[ERROR] Closed-loop system is UNSTABLE (max Re(λ) = "
                  << max_real << ")\n";
    }

    return result;
}

}  // namespace tricopter
