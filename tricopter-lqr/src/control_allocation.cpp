// control_allocation.cpp — Builds the 6×N control effectiveness matrix B
// and analyzes controllability.
//
// B maps [ω₁², ω₂², ..., ωₙ²] to [Fx, Fy, Fz, Tx, Ty, Tz] in body frame.

#include "control_allocation.hpp"
#include <iostream>
#include <iomanip>

namespace tricopter {

Eigen::MatrixXd build_control_matrix(const VehicleConfig& config) {
    const int N = static_cast<int>(config.rotors.size());

    // B is 6×N: top 3 rows = force, bottom 3 rows = torque
    Eigen::MatrixXd B(6, N);

    for (int i = 0; i < N; ++i) {
        const auto& rotor = config.rotors[i];

        // Force contribution (body frame): k_T * thrust_axis
        // When multiplied by ωᵢ², gives the thrust force vector
        Eigen::Vector3d force_col = rotor.k_T * rotor.thrust_axis;

        // Torque contribution (body frame):
        //   = (r_i × k_T * thrust_axis) + k_Q * spin_dir * spin_axis
        Eigen::Vector3d torque_col =
            rotor.position.cross(force_col) +
            rotor.k_Q * rotor.spin_direction * rotor.spin_axis;

        B.block<3, 1>(0, i) = force_col;   // rows 0-2: force
        B.block<3, 1>(3, i) = torque_col;   // rows 3-5: torque
    }

    return B;
}

int analyze_controllability(const Eigen::MatrixXd& B) {
    const int rows = static_cast<int>(B.rows());
    const int cols = static_cast<int>(B.cols());

    std::cout << "\n=== Control Allocation Matrix B (" << rows << "x" << cols << ") ===\n";
    std::cout << std::fixed << std::setprecision(6);

    // Print the matrix with labeled rows
    const char* row_labels[] = {"Fx", "Fy", "Fz", "Tx", "Ty", "Tz"};
    for (int i = 0; i < rows; ++i) {
        std::cout << "  " << row_labels[i] << ": ";
        for (int j = 0; j < cols; ++j) {
            std::cout << std::setw(12) << B(i, j);
        }
        std::cout << "\n";
    }

    // SVD for rank and condition analysis
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(B, Eigen::ComputeThinU | Eigen::ComputeThinV);
    auto singular_values = svd.singularValues();

    std::cout << "\nSingular values:\n";
    for (int i = 0; i < singular_values.size(); ++i) {
        std::cout << "  σ" << i + 1 << " = " << singular_values(i) << "\n";
    }

    // Determine rank with threshold
    double threshold = singular_values(0) * 1e-10;
    int rank = 0;
    for (int i = 0; i < singular_values.size(); ++i) {
        if (singular_values(i) > threshold) {
            ++rank;
        }
    }

    std::cout << "\nMatrix rank: " << rank << " (out of " << std::min(rows, cols) << ")\n";

    if (rank < std::min(rows, cols)) {
        std::cout << "[WARNING] B matrix is rank-deficient! Some axes may be uncontrollable.\n";

        // Identify weakly controllable axes via the U matrix
        Eigen::MatrixXd U = svd.matrixU();
        for (int i = 0; i < singular_values.size(); ++i) {
            if (singular_values(i) < threshold) {
                std::cout << "  Uncontrollable direction (U column " << i << "): [";
                for (int j = 0; j < rows; ++j) {
                    std::cout << (j > 0 ? ", " : "") << U(j, i);
                }
                std::cout << "]\n";
            }
        }
    }

    // Check for weak controllability (condition number)
    double cond = singular_values(0) / singular_values(singular_values.size() - 1);
    std::cout << "Condition number: " << std::scientific << cond << "\n";
    if (cond > 1e6) {
        std::cout << "[WARNING] High condition number — some axes are weakly controllable.\n";
    }

    // Check for individual axis controllability
    std::cout << "\nPer-axis force/torque authority:\n";
    for (int i = 0; i < rows; ++i) {
        double authority = B.row(i).norm();
        std::cout << "  " << row_labels[i] << ": ||B_row|| = " << std::fixed
                  << std::setprecision(8) << authority;
        if (authority < 1e-10) {
            std::cout << "  [UNCONTROLLABLE]";
        } else if (authority < 1e-6) {
            std::cout << "  [WEAK]";
        }
        std::cout << "\n";
    }

    return rank;
}

}  // namespace tricopter
