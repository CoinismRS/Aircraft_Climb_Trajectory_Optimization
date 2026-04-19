"""
Aircraft Climb Trajectory Optimization (SciPy-only fallback, no IPOPT)

Uses:
- numpy
- scipy.optimize.minimize with trust-constr
- matplotlib

State variables:
    x(t): horizontal position [m]
    h(t): altitude [m]
    V(t): airspeed [m/s]
    m(t): mass [kg]

Control variables:
    T(t): thrust [N]
    gamma(t): flight path angle [rad]

Goal:
    Minimize fuel consumption during climb while satisfying nonlinear dynamics
    and terminal constraints.

Notes:
- This is a direct-transcription NLP solved with SciPy.
- It is meant to be understandable and class-project friendly.
- Units are SI.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, Bounds, NonlinearConstraint


# ============================================================
# 1. Problem / aircraft parameters
# ============================================================

g = 9.81                 # gravity [m/s^2]
rho0 = 1.225             # sea-level air density [kg/m^3]
H_scale = 8500.0         # exponential atmosphere scale height [m]

S = 30.0                 # wing reference area [m^2]
CD0 = 0.02               # zero-lift drag coefficient
k_induced = 0.045        # induced drag factor
m_ref = 7500.0           # reference mass for simple CL estimate [kg]

T_max = 45000.0          # max thrust [N]
Isp = 3000.0             # effective specific impulse [s]
fuel_coeff = 1.0 / (Isp * g)   # m_dot = -fuel_coeff * T

# Time horizon
t0 = 0.0
tf = 240.0               # final time [s]
N = 60                   # number of nodes
dt = (tf - t0) / (N - 1)
time_grid = np.linspace(t0, tf, N)

# Initial conditions
x_init = 0.0             # [m]
h_init = 0.0             # [m]
V_init = 80.0            # [m/s]
m_init = 8000.0          # [kg]

# Target terminal conditions
h_target = 3000.0        # [m]
V_target = 140.0         # [m/s]

# Path bounds
h_min, h_max = 0.0, 12000.0
V_min, V_max = 60.0, 220.0
m_min = 6500.0

# Control bounds
T_min, T_max_bound = 5000.0, T_max
gamma_min_deg, gamma_max_deg = -3.0, 15.0
gamma_min = np.deg2rad(gamma_min_deg)
gamma_max = np.deg2rad(gamma_max_deg)

# Optional target for final horizontal progress
x_target_min = 12000.0   # [m]


# ============================================================
# 2. Helper functions
# ============================================================

def air_density(h):
    """Simple exponential atmosphere."""
    return rho0 * np.exp(-h / H_scale)


def drag(V, h, m):
    """
    Simple drag model:
        D = 0.5*rho*V^2*S*(CD0 + k*CL^2)

    Approximate CL from quasi-level support:
        CL ≈ (2*m*g)/(rho*V^2*S)
    """
    rho = air_density(h)
    V_safe = np.maximum(V, 1.0)
    q = 0.5 * rho * V_safe**2
    CL = (2.0 * m * g) / np.maximum(rho * V_safe**2 * S, 1e-6)
    CD = CD0 + k_induced * CL**2
    return q * S * CD


def pack_decision_variables(x, h, V, m, T, gamma):
    """Flatten all variables into a single vector z."""
    return np.concatenate([x, h, V, m, T, gamma])


def unpack_decision_variables(z):
    """Recover each trajectory/control array from decision vector z."""
    idx = 0
    x = z[idx:idx+N]; idx += N
    h = z[idx:idx+N]; idx += N
    V = z[idx:idx+N]; idx += N
    m = z[idx:idx+N]; idx += N
    T = z[idx:idx+N]; idx += N
    gamma = z[idx:idx+N]; idx += N
    return x, h, V, m, T, gamma


# ============================================================
# 3. Objective function
# ============================================================

def objective(z):
    """
    Minimize fuel burn with mild regularization on controls.

    Primary term:
        fuel_used = m_init - m_final

    Regularization:
        penalize large changes in T and gamma to encourage smooth controls
    """
    x, h, V, m, T, gamma = unpack_decision_variables(z)

    fuel_used = m_init - m[-1]

    # Smoothness penalties
    dT = np.diff(T)
    dgamma = np.diff(gamma)

    smooth_T = 1e-8 * np.sum(dT**2)
    smooth_gamma = 1e1 * np.sum(dgamma**2)

    # Mild penalty to encourage reaching a meaningful forward distance
    progress_penalty = 1e-6 * max(0.0, x_target_min - x[-1])**2

    return fuel_used + smooth_T + smooth_gamma + progress_penalty


# ============================================================
# 4. Dynamic equality constraints
# ============================================================

def dynamics_constraints(z):
    """
    Enforce discretized dynamics using forward Euler:

        x_{k+1} = x_k + dt * V_k cos(gamma_k)
        h_{k+1} = h_k + dt * V_k sin(gamma_k)
        V_{k+1} = V_k + dt * (T_k - D_k)/m_k - g*sin(gamma_k)
        m_{k+1} = m_k - dt * fuel_coeff*T_k

    Also enforce initial and terminal conditions.
    """
    x, h, V, m, T, gamma = unpack_decision_variables(z)

    cons = []

    # Initial conditions
    cons.append(x[0] - x_init)
    cons.append(h[0] - h_init)
    cons.append(V[0] - V_init)
    cons.append(m[0] - m_init)

    # Dynamics at each interval
    for k in range(N - 1):
        Dk = drag(V[k], h[k], m[k])

        x_next_pred = x[k] + dt * V[k] * np.cos(gamma[k])
        h_next_pred = h[k] + dt * V[k] * np.sin(gamma[k])
        V_next_pred = V[k] + dt * ((T[k] - Dk) / m[k] - g * np.sin(gamma[k]))
        m_next_pred = m[k] - dt * fuel_coeff * T[k]

        cons.append(x[k + 1] - x_next_pred)
        cons.append(h[k + 1] - h_next_pred)
        cons.append(V[k + 1] - V_next_pred)
        cons.append(m[k + 1] - m_next_pred)

    # Terminal conditions
    cons.append(h[-1] - h_target)
    cons.append(V[-1] - V_target)

    return np.array(cons)


# ============================================================
# 5. Inequality / path constraints
# ============================================================

def path_constraints(z):
    """
    Return vector c(z) for inequality bounds lb <= c(z) <= ub.

    We include:
    - altitude h
    - speed V
    - mass m
    - final horizontal progress x_final
    """
    x, h, V, m, T, gamma = unpack_decision_variables(z)

    c = np.concatenate([
        h,                  # bounded by [h_min, h_max]
        V,                  # bounded by [V_min, V_max]
        m,                  # bounded by [m_min, inf]
        np.array([x[-1]])   # bounded by [x_target_min, inf]
    ])
    return c


# ============================================================
# 6. Initial guess
# ============================================================

def build_initial_guess():
    """
    Construct a feasible-ish initial guess.
    It does not need to satisfy constraints exactly, but it should be reasonable.
    """
    x_guess = np.linspace(x_init, 15000.0, N)
    h_guess = np.linspace(h_init, h_target, N)
    V_guess = np.linspace(V_init, V_target, N)
    m_guess = np.linspace(m_init, 7600.0, N)
    T_guess = np.full(N, 0.65 * T_max)
    gamma_guess = np.full(N, np.deg2rad(5.0))

    return pack_decision_variables(
        x_guess, h_guess, V_guess, m_guess, T_guess, gamma_guess
    )


# ============================================================
# 7. Bounds on decision variables
# ============================================================

def build_variable_bounds():
    """
    Variable-wise bounds for all decision variables.
    """
    x_lb = np.full(N, 0.0)
    x_ub = np.full(N, 1.0e6)

    h_lb_arr = np.full(N, h_min)
    h_ub_arr = np.full(N, h_max)

    V_lb_arr = np.full(N, V_min)
    V_ub_arr = np.full(N, V_max)

    m_lb_arr = np.full(N, m_min)
    m_ub_arr = np.full(N, m_init)

    T_lb_arr = np.full(N, T_min)
    T_ub_arr = np.full(N, T_max_bound)

    gamma_lb_arr = np.full(N, gamma_min)
    gamma_ub_arr = np.full(N, gamma_max)

    lb = np.concatenate([
        x_lb, h_lb_arr, V_lb_arr, m_lb_arr, T_lb_arr, gamma_lb_arr
    ])

    ub = np.concatenate([
        x_ub, h_ub_arr, V_ub_arr, m_ub_arr, T_ub_arr, gamma_ub_arr
    ])

    return Bounds(lb, ub)


# ============================================================
# 8. Solve optimization problem
# ============================================================

def solve_problem():
    z0 = build_initial_guess()
    bounds = build_variable_bounds()

    # Equality constraints: dynamics + initial/final conditions
    dyn0 = dynamics_constraints(z0)
    eq_constraint = NonlinearConstraint(
        dynamics_constraints,
        lb=np.zeros_like(dyn0),
        ub=np.zeros_like(dyn0)
    )

    # Path constraints:
    # h in [h_min, h_max], V in [V_min, V_max], m in [m_min, inf], x_final >= x_target_min
    pc0 = path_constraints(z0)

    lb_path = np.concatenate([
        np.full(N, h_min),
        np.full(N, V_min),
        np.full(N, m_min),
        np.array([x_target_min])
    ])

    ub_path = np.concatenate([
        np.full(N, h_max),
        np.full(N, V_max),
        np.full(N, np.inf),
        np.array([np.inf])
    ])

    path_constraint = NonlinearConstraint(
        path_constraints,
        lb=lb_path,
        ub=ub_path
    )

    result = minimize(
        objective,
        z0,
        method="trust-constr",
        bounds=bounds,
        constraints=[eq_constraint, path_constraint],
        options={
            "maxiter": 100,
            "verbose": 3
        }
    )

    return result


# ============================================================
# 9. Post-processing and plotting
# ============================================================

def plot_results(z_opt):
    x, h, V, m, T, gamma = unpack_decision_variables(z_opt)

    fuel_used = m_init - m[-1]

    print("\nOptimization summary")
    print("-" * 40)
    print(f"Final horizontal distance : {x[-1]:.2f} m")
    print(f"Final altitude            : {h[-1]:.2f} m")
    print(f"Final speed               : {V[-1]:.2f} m/s")
    print(f"Final mass                : {m[-1]:.2f} kg")
    print(f"Fuel used                 : {fuel_used:.2f} kg")
    print(f"Average thrust            : {np.mean(T):.2f} N")
    print(f"Average gamma             : {np.rad2deg(np.mean(gamma)):.2f} deg")

    # ============================================================
    # Single Figure with Subplots
    # ============================================================

    fig, axs = plt.subplots(3, 2, figsize=(12, 10))

    # 1. Trajectory (x vs h)
    axs[0, 0].plot(x / 1000.0, h / 1000.0)
    axs[0, 0].set_title("Climb Trajectory")
    axs[0, 0].set_xlabel("Distance [km]")
    axs[0, 0].set_ylabel("Altitude [km]")
    axs[0, 0].grid(True)

    # 2. Velocity
    axs[0, 1].plot(time_grid, V)
    axs[0, 1].set_title("Velocity Profile")
    axs[0, 1].set_xlabel("Time [s]")
    axs[0, 1].set_ylabel("Velocity [m/s]")
    axs[0, 1].grid(True)

    # 3. Thrust
    axs[1, 0].plot(time_grid, T)
    axs[1, 0].set_title("Thrust Profile")
    axs[1, 0].set_xlabel("Time [s]")
    axs[1, 0].set_ylabel("Thrust [N]")
    axs[1, 0].grid(True)

    # 4. Flight Path Angle
    axs[1, 1].plot(time_grid, np.rad2deg(gamma))
    axs[1, 1].set_title("Flight Path Angle")
    axs[1, 1].set_xlabel("Time [s]")
    axs[1, 1].set_ylabel("Gamma [deg]")
    axs[1, 1].grid(True)

    # 5. Mass
    axs[2, 0].plot(time_grid, m)
    axs[2, 0].set_title("Mass Depletion")
    axs[2, 0].set_xlabel("Time [s]")
    axs[2, 0].set_ylabel("Mass [kg]")
    axs[2, 0].grid(True)

    # 6. Hide unused subplot (bottom-right)
    axs[2, 1].axis("off")

    # Improve spacing
    plt.tight_layout()

    # Show everything in ONE window
    plt.show()


# ============================================================
# 10. Main
# ============================================================

if __name__ == "__main__":
    result = solve_problem()

    print("\nSolver status:", result.message)
    print("Success:", result.success)
    print("Objective value:", result.fun)

    if result.success:
        plot_results(result.x)
    else:
        print("\nThe solver did not fully converge.")
        print("You can still inspect the best iterate returned so far.")
        plot_results(result.x)