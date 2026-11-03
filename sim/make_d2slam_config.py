"""Write the D2VINS camchain and config for the simulated quad camera into d2slam/config/sim/.

    python3 sim/make_d2slam_config.py [--imu-rate 250] [--image-rate 25]
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(HERE, "vehicle"))
import fisheye_model as fm  # noqa: E402

OUT = os.path.join(ROOT, "d2slam", "config", "sim")
TEMPLATE = os.path.join(ROOT, "third_party", "D2SLAM", "config", "quadcam_drone_nxt_tmp", "quadcam_single.yaml")

# copy of the rig in sim/vehicle/cameras.py, which only imports inside Isaac; IMU at the body origin, FLU
FISHEYE_POS = [(0.0944, -0.0386, 0.0963), (0.0944, 0.0386, 0.0963), (-0.1074, 0.0386, 0.0963),
               (-0.1074, -0.0386, 0.0963)]
FISHEYE_YAW = [-45.0, 45.0, 135.0, -135.0]  # counter-clockwise, as D2VINS cross-camera tracking expects


def optical_to_body(yaw_deg):
    p = np.radians(yaw_deg)
    z = np.array([np.cos(p), np.sin(p), 0.0])
    y = np.array([0.0, 0.0, -1.0])
    return np.column_stack([np.cross(y, z), y, z])


def camchain():
    c = fm.mei_calibration()
    lines = []
    for i, (pos, yaw) in enumerate(zip(FISHEYE_POS, FISHEYE_YAW)):
        r_bc = optical_to_body(yaw)
        t = np.eye(4)
        t[:3, :3] = r_bc.T
        t[:3, 3] = -r_bc.T @ np.array(pos)
        lines.append(f"cam{i}:")
        lines.append("  T_cam_imu:")
        for row in t:
            lines.append("  - [" + ", ".join(f"{v:.9f}" for v in row) + "]")
        lines.append("  cam_overlaps: []")
        lines.append("  camera_model: omni")
        lines.append(f"  distortion_coeffs: [{c['k1']:.8f}, {c['k2']:.8f}, 0.0, 0.0]")
        lines.append("  distortion_model: radtan")
        # Isaac measures pixel positions from the image edge; OpenCV/camodocal from the first pixel centre
        lines.append(f"  intrinsics: [{c['xi']:.8f}, {c['gamma']:.6f}, {c['gamma']:.6f}, {fm.CX - 0.5:.1f}, {fm.CY - 0.5:.1f}]")
        lines.append(f"  resolution: [{fm.W}, {fm.H}]")
        lines.append(f"  rostopic: /quadcam/cam{i}")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imu-rate", type=int, default=250)
    ap.add_argument("--image-rate", type=int, default=25)
    ap.add_argument("--max-cnt", type=int, default=100)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "fisheye_cams.yaml"), "w") as f:
        f.write(camchain())
    text = open(TEMPLATE).read()
    sets = {
        "imu_topic": '"/imu"',
        "image0_topic": '"/quadcam/image"',
        "imu_freq": args.imu_rate,
        "image_freq": args.image_rate,
        "image_width": fm.W,
        "image_height": fm.H,
        "max_cnt": args.max_cnt,
        "max_superpoint_cnt": 0,  # CPU build: SuperPoint needs TensorRT
        "estimate_td": 0,  # sim stamps are exact
        "td": 0.0,
        "cnn_use_onnx": 0,
        # set above the Pegasus IMU noise to absorb model error
        "acc_n": 0.05,
        "gyr_n": 0.005,
        "acc_w": 0.001,
        "gyr_w": 0.0001,
        "g_norm": 9.81,
        "output_path": '"/root/output/"',
    }
    # keys added by patches/d2slam
    extra = {
        "enable_cross_cam_track": 1,
        # px; low-disparity cross-camera matches carry no depth and tilt the trajectory
        "cross_cam_min_disparity": 2.0,
        # the drone starts at rest on a calibrated IMU
        "add_vel_ba_prior": 1,
    }
    out = []
    for line in text.splitlines():
        key = line.split(":")[0].strip()
        if key in sets and not line.lstrip().startswith("#") and line[:1] != " ":
            out.append(f"{key}: {sets.pop(key)}")
        else:
            out.append(line)
    if sets:
        raise SystemExit(f"keys not found in template: {sorted(sets)}")
    out += [f"{k}: {v}" for k, v in extra.items()]
    with open(os.path.join(OUT, "quadcam_sim.yaml"), "w") as f:
        f.write("\n".join(out) + "\n")
    print("wrote", OUT, "MEI", {k: round(v, 4) for k, v in fm.mei_calibration().items()})


if __name__ == "__main__":
    main()
