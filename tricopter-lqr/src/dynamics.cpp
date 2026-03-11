// dynamics.cpp — Full 6DOF nonlinear rigid body dynamics
//
// Implements the equations of motion for an asymmetric multirotor vehicle
// with arbitrary rotor placement, thrust axes, and reaction torques.

#include "dynamics.hpp"
#include <cmath>

namespace tricopter {

Eigen::Matrix3d quaternion_to_rotation(const Eigen::Quaterniond& q) {
    // Eigen's toRotationMatrix() returns R such that v_inertial = R * v_body
    return q.normalized().toRotationMatrix();
}

Eigen::Quaterniond euler_to_quaternion(double roll, double pitch, double yaw) {
    // ZYX convention: R = Rz(yaw) * Ry(pitch) * Rx(roll)
    Eigen::Quaterniond q =
        Eigen::AngleAxisd(yaw,   Eigen::Vector3d::UnitZ()) *
        Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) *
        Eigen::AngleAxisd(roll,  Eigen::Vector3d::UnitX());
    return q.normalized();
}

Eigen::Vector3d quaternion_to_euler(const Eigen::Quaterniond& q) {
    // Extract ZYX Euler angles from quaternion
    Eigen::Matrix3d R = quaternion_to_rotation(q);

    double roll, pitch, yaw;

    // pitch = -asin(R(2,0)), handling gimbal lock
    double sp = -R(2, 0);
    if (sp >= 1.0) {
        pitch = M_PI / 2.0;
    } else if (sp <= -1.0) {
        pitch = -M_PI / 2.0;
    } else {
        pitch = std::asin(sp);
    }

    // roll = atan2(R(2,1), R(2,2))
    roll = std::atan2(R(2, 1), R(2, 2));

    // yaw = atan2(R(1,0), R(0,0))
    yaw = std::atan2(R(1, 0), R(0, 0));

    return Eigen::Vector3d(roll, pitch, yaw);
}

void normalize_quaternion(Eigen::VectorXd& state) {
    // Quaternion is at indices 6..9 (w, x, y, z)
    double norm = state.segment<4>(6).norm();
    if (norm > 1e-10) {
        state.segment<4>(6) /= norm;
    }
    // Ensure scalar part is positive (canonical form)
    if (state(6) < 0.0) {
        state.segment<4>(6) *= -1.0;
    }
}

Eigen::VectorXd dynamics(const Eigen::VectorXd& state,
                         const Eigen::VectorXd& u,
                         const VehicleConfig& config) {
    // state: (13,) = [pos(3), vel(3), quat(4), omega(3)]
    // u:     (N,)  = [ω₁², ω₂², ..., ωₙ²]

    const int N = static_cast<int>(config.rotors.size());
    Eigen::VectorXd dxdt(STATE_DIM);

    // --- Extract state components ---
    // Position is state[0:3], velocity is state[3:6]
    Eigen::Vector3d vel   = state.segment<3>(3);

    // Quaternion (scalar-first): w=state[6], x=state[7], y=state[8], z=state[9]
    Eigen::Quaterniond q(state(6), state(7), state(8), state(9));
    q.normalize();

    // Body angular velocity
    Eigen::Vector3d omega = state.segment<3>(10);

    // Rotation matrix: body → inertial
    Eigen::Matrix3d R = quaternion_to_rotation(q);

    // --- Compute total force and torque in body frame ---
    Eigen::Vector3d total_force_body  = Eigen::Vector3d::Zero();  // [N]
    Eigen::Vector3d total_torque_body = Eigen::Vector3d::Zero();  // [N·m]

    for (int i = 0; i < N; ++i) {
        const auto& rotor = config.rotors[i];
        double ui = u(i);  // ωᵢ² [(rad/s)²]

        // Thrust force in body frame: T_i = k_T * ω² * thrust_axis
        Eigen::Vector3d thrust_force = rotor.k_T * ui * rotor.thrust_axis;
        total_force_body += thrust_force;

        // Torque from thrust moment arm: r_i × F_i
        Eigen::Vector3d thrust_torque = rotor.position.cross(thrust_force);

        // Reaction (drag) torque about spin axis: Q_i = k_Q * ω² * spin_dir * spin_axis
        // Sign: CW rotor (+1) produces CCW reaction torque, hence the spin_direction
        // captures the rotor spin sense, and the reaction opposes it.
        Eigen::Vector3d drag_torque =
            rotor.k_Q * ui * rotor.spin_direction * rotor.spin_axis;

        total_torque_body += thrust_torque + drag_torque;
    }

    // --- Translational dynamics (inertial frame) ---
    // m*a = R * F_body + m * g
    Eigen::Vector3d gravity(0.0, 0.0, -9.81);  // [m/s²] NED convention: z-up
    Eigen::Vector3d accel = (R * total_force_body) / config.mass + gravity;

    // --- Rotational dynamics (body frame) ---
    // J * omega_dot = tau - omega × (J * omega)
    Eigen::Vector3d J_omega = config.inertia * omega;
    Eigen::Vector3d omega_dot =
        config.inertia.inverse() * (total_torque_body - omega.cross(J_omega));

    // --- Quaternion kinematics ---
    // q_dot = 0.5 * q ⊗ [0, omega]
    // Using Eigen quaternion multiplication (Hamilton convention, scalar-first)
    Eigen::Quaterniond omega_quat(0.0, omega.x(), omega.y(), omega.z());
    Eigen::Quaterniond q_dot_quat = Eigen::Quaterniond(
        0.5 * (q * omega_quat).coeffs());

    // --- Assemble state derivative ---
    // d(position)/dt = velocity
    dxdt.segment<3>(0) = vel;

    // d(velocity)/dt = acceleration
    dxdt.segment<3>(3) = accel;

    // d(quaternion)/dt — note Eigen stores coeffs as [x, y, z, w]
    // We use scalar-first convention in our state vector [w, x, y, z]
    Eigen::Vector4d q_dot_coeffs;
    q_dot_coeffs(0) = q_dot_quat.w();  // dw/dt
    q_dot_coeffs(1) = q_dot_quat.x();  // dx/dt
    q_dot_coeffs(2) = q_dot_quat.y();  // dy/dt
    q_dot_coeffs(3) = q_dot_quat.z();  // dz/dt
    dxdt.segment<4>(6) = q_dot_coeffs;

    // d(omega)/dt = angular acceleration
    dxdt.segment<3>(10) = omega_dot;

    return dxdt;
}

}  // namespace tricopter
