"""Vehicle design parameters shared by the simulation and the PX4 airframe (FLU body frame, origin at the frame bottom)."""
import math
import os

RHO = 1.18  # kg/m^3
G = 9.81

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHARPI_USD = os.environ.get("CHARPI_USD", os.path.join(ROOT, "third_party", "charpi_vision_description", "usd", "charpi.usd"))

MASS = 2.0  # kg, all-up
INERTIA = (0.0078, 0.0100, 0.0180)  # kg m^2 about the COM, Charpi values scaled to 2.0 kg
COM = (-0.007, 0.0, 0.072)  # m, from the Charpi USD

PROP_D = 7 * 0.0254  # m
C_T, C_P = 0.10, 0.045  # typical static thrust and power coefficients of a 7x4 prop
K_T = C_T * RHO * PROP_D ** 4 / (4 * math.pi ** 2)  # N/(rad/s)^2
K_Q = C_P * RHO * PROP_D ** 5 / (8 * math.pi ** 3)  # N m/(rad/s)^2
W_MAX = 2094.0  # rad/s (20,000 rpm)
W_IDLE = 200.0  # rad/s at zero PX4 output when armed
KM = K_Q / K_T  # m, PX4 CA_ROTORn_KM

# PX4 quad-X order FR, RL, FL, RR; FLU positions (m); Pegasus rot_dir -1 = CCW from above
ARM = 0.10
ROTOR_Z = 0.04
ROTORS = [(ARM, -ARM, ROTOR_Z), (-ARM, ARM, ROTOR_Z), (ARM, ARM, ROTOR_Z), (-ARM, -ARM, ROTOR_Z)]
ROT_DIR = [-1, -1, 1, 1]

DRAG = (0.35, 0.35, 0.10)  # N/(m/s), linear body drag

# Cameras, FLU body frame (m); fisheyes run counter-clockwise from front-right, as D2VINS expects
FISHEYE_RADIUS = 0.105
FISHEYE_Z = 0.096
FISHEYE_YAW_DEG = [-45.0, 45.0, 135.0, -135.0]
FISHEYE_RES = (640, 360)  # per tile
BARCODE_RES = (4056, 3040)  # IMX477
BARCODE_F_PX = 7742.0  # px, 12 mm lens on 1.55 um pixels
BARCODE_POS = {"left": (0.0, 0.06, 0.06), "right": (0.0, -0.06, 0.06)}


def thrust_max():
    return K_T * W_MAX ** 2


def hover_omega(mass=MASS):
    return math.sqrt(mass * G / 4 / K_T)


def hover_output(mass=MASS):
    """PX4 motor output u in [0, 1] that hovers, with w = W_IDLE + (W_MAX - W_IDLE) u."""
    return (hover_omega(mass) - W_IDLE) / (W_MAX - W_IDLE)


def thr_mdl_fac():
    """PX4 THR_MDL_FAC f that best fits T(u)/T_max = (1 - f) u + f u^2 for the affine w(u)."""
    us = [i / 50 for i in range(51)]
    t = [((W_IDLE + (W_MAX - W_IDLE) * u) / W_MAX) ** 2 for u in us]
    # least squares for f in t - u = f (u^2 - u)
    num = sum((ti - u) * (u * u - u) for ti, u in zip(t, us))
    den = sum((u * u - u) ** 2 for u in us)
    return num / den


def hover_thrust_norm(mass=MASS):
    return mass * G / 4 / thrust_max()


if __name__ == "__main__":
    print(f"k_T {K_T:.3e} N/(rad/s)^2, k_Q {K_Q:.3e} N m/(rad/s)^2, KM {KM:.4f} m")
    print(f"max thrust {thrust_max():.1f} N per rotor, T/W {4 * thrust_max() / (MASS * G):.2f}")
    print(f"hover w {hover_omega():.0f} rad/s, u {hover_output():.3f}, normalised thrust {hover_thrust_norm():.3f}, "
          f"THR_MDL_FAC {thr_mdl_fac():.2f}")
    p_mech = 4 * K_Q * hover_omega() ** 3
    p_ideal = (MASS * G) ** 1.5 / math.sqrt(2 * RHO * 4 * math.pi * (PROP_D / 2) ** 2)
    print(f"hover shaft power {p_mech:.0f} W, ideal {p_ideal:.0f} W, figure of merit {p_ideal / p_mech:.2f}")

# Downward 1D lidar (TFmini-S class), body FLU (m); gives the barcode cameras a tighter height than VIO
RANGE_POS = (0.0, 0.0, 0.16)  # above the landing feet, clear of the 10 cm blind zone
RANGE_MIN, RANGE_MAX = 0.1, 12.0  # m
RANGE_SIGMA = 0.01  # m, 1 sigma
RANGE_RATE = 50.0  # Hz

# side lidars for aisle centring, body FLU (m), looking along +-y
SIDE_RANGE_Y, SIDE_RANGE_Z = 0.08, 0.06
