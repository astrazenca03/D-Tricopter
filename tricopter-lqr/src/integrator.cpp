// integrator.cpp — 4th-order Runge-Kutta integrator
//
// Classic RK4 with fixed control input u held constant over the step.
// Normalizes the quaternion after each integration step.

#include "integrator.hpp"
#include "dynamics.hpp"

namespace tricopter {

Eigen::VectorXd rk4_step(const DynamicsFunc& f,
                         const Eigen::VectorXd& state,
                         const Eigen::VectorXd& u,
                         double dt) {
    // Classic RK4:
    //   k1 = f(x, u)
    //   k2 = f(x + dt/2 * k1, u)
    //   k3 = f(x + dt/2 * k2, u)
    //   k4 = f(x + dt * k3, u)
    //   x_new = x + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

    Eigen::VectorXd k1 = f(state, u);
    Eigen::VectorXd k2 = f(state + 0.5 * dt * k1, u);
    Eigen::VectorXd k3 = f(state + 0.5 * dt * k2, u);
    Eigen::VectorXd k4 = f(state + dt * k3, u);

    Eigen::VectorXd new_state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);

    // Normalize quaternion to maintain unit norm
    normalize_quaternion(new_state);

    return new_state;
}

}  // namespace tricopter
