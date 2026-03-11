#pragma once
// integrator.hpp — 4th-order Runge-Kutta integrator for the dynamics.
//
// Propagates the 13-state nonlinear dynamics forward by one timestep.
// Normalizes the quaternion after each step.

#include <Eigen/Dense>
#include <functional>
#include "config.hpp"

namespace tricopter {

// Type alias for the dynamics function signature:
//   f(state, control_input) -> state_derivative
using DynamicsFunc = std::function<Eigen::VectorXd(const Eigen::VectorXd&,
                                                    const Eigen::VectorXd&)>;

// Perform one RK4 integration step.
//
// Arguments:
//   f      — dynamics function f(x, u) -> dx/dt
//   state  — (13,) current state
//   u      — (num_rotors,) control input (held constant over the step)
//   dt     — timestep [s]
//
// Returns:
//   (13,) state at t + dt (quaternion normalized)
Eigen::VectorXd rk4_step(const DynamicsFunc& f,
                         const Eigen::VectorXd& state,
                         const Eigen::VectorXd& u,
                         double dt);

}  // namespace tricopter
