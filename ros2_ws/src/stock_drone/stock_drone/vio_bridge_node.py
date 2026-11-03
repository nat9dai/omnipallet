"""Forward VIO odometry to PX4 EKF2 as external vision and to /nav/odom, with side-lidar drift corrections."""
import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from px4_msgs.msg import VehicleOdometry

from stock_drone import aisle_spec as spec
from sensor_msgs.msg import Range
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Float64

F = np.diag([1.0, -1.0, -1.0])  # VIO z-up FLU to PX4 z-down FRD: p_ned = F p, R_ned_frd = F R_flu F
PX4_QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                     history=HistoryPolicy.KEEP_LAST, depth=1)


def quat_to_rot(w, x, y, z):
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def rot_to_quat(r):
    """Rotation matrix to (w, x, y, z) with w >= 0."""
    t = np.trace(r)
    if t > 0:
        s = 2 * np.sqrt(t + 1)
        q = [s / 4, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(r)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2 * np.sqrt(1 + r[i, i] - r[j, j] - r[k, k])
        q = [0.0] * 4
        q[0] = (r[k, j] - r[j, k]) / s
        q[1 + i] = s / 4
        q[1 + j] = (r[j, i] + r[i, j]) / s
        q[1 + k] = (r[k, i] + r[i, k]) / s
    q = np.array(q)
    return q if q[0] >= 0 else -q


class VioBridge(Node):
    def __init__(self):
        super().__init__("vio_bridge")
        self.declare_parameter("odom_topic", "/d2vins/imu_propagation")
        self.declare_parameter("twist_in_world", True)  # D2VINS writes world-frame velocity in the twist
        self.declare_parameter("pos_std", 0.05)  # m
        self.declare_parameter("att_std", 0.03)  # rad
        self.declare_parameter("vel_std", 0.10)  # m/s
        self.declare_parameter("max_rate_hz", 100.0)
        self.declare_parameter("aisle_lock", True)
        self.declare_parameter("aisle_width", 2.5)  # m, between pallet faces
        self.declare_parameter("side_lidar_y", 0.08)  # m, side lidars at +-y in the body frame
        self.declare_parameter("down_lidar_z", 0.16)  # m, downward lidar above the body origin
        self.declare_parameter("lock_tau", 1.0)  # s, time constant of the sideways correction
        self.declare_parameter("width_tol", 0.06)  # m, max gap between left + right range and the aisle width
        self.declare_parameter("rack_lock", True)
        # "ned" takes the start heading as north, so with an EKF origin PX4 also gives a global position for a GCS map
        self.declare_parameter("pose_frame", "frd")
        self.declare_parameter("aisle_origin", [-3.0, 0.0])  # m, VIO origin in the aisle; VIO must start facing +x
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.lock = g("aisle_lock")
        self.pair_sum = g("aisle_width") - 2 * g("side_lidar_y")
        self.down_z, self.tau, self.width_tol = g("down_lidar_z"), g("lock_tau"), g("width_tol")
        self.side = {"L": None, "R": None}  # (t, range)
        self.down = None
        self.delta = 0.0  # m, VIO y minus lidar y
        self.t_lock = None
        self.n_lock = 0
        self.rack = g("rack_lock")
        self.origin = np.array(g("aisle_origin"), float)
        self.uprights = spec.RACK_X0 + spec.BAY_PITCH * np.arange(spec.N_BAYS + 1)
        self.near_max = spec.WIDTH / 2 + spec.FACE_GAP + 0.25 - g("side_lidar_y")  # m, max range of a rack-front return
        self.run = {"L": None, "R": None}  # current near-return run: [x_first, x_last, n, sum of ranges]
        self.dx = 0.0  # m, VIO x minus rack x
        self.n_rack = 0
        self.last = None  # (t, x_vio, vx) of the latest odometry
        self.twist_world = self.get_parameter("twist_in_world").value
        self.std = [self.get_parameter(n).value for n in ("pos_std", "att_std", "vel_std")]
        self.min_dt = 1.0 / self.get_parameter("max_rate_hz").value
        self.last_t = -1.0
        self.reset_counter = 0
        self.offset = 0.0  # s, sim time minus PX4 time; 0 on hardware (uXRCE-DDS time sync)
        self.pub = self.create_publisher(VehicleOdometry, "/fmu/in/vehicle_visual_odometry", PX4_QOS)
        self.create_subscription(Odometry, self.get_parameter("odom_topic").value, self.on_odom, 50)
        self.create_subscription(Float64, "/sim/px4_time_offset", self.on_offset, 10)
        self.create_subscription(Range, "/range_left", lambda m: self.on_range("L", m), 20)
        self.create_subscription(Range, "/range_right", lambda m: self.on_range("R", m), 20)
        self.create_subscription(Range, "/range", self.on_down, 20)
        self.pub_nav = self.create_publisher(Odometry, "/nav/odom", 50)
        self.pub_delta = self.create_publisher(Float64, "/nav/aisle_offset", 10)
        self.pub_dx = self.create_publisher(Float64, "/nav/rack_offset", 10)

    def on_range(self, side, msg):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.side[side] = (t, msg.range)
        if self.rack and self.last is not None:
            t0, x0, vx = self.last
            self.update_rack(side, x0 + vx * (t - t0), vx, msg.range)

    def update_rack(self, side, x_vio, vx, r):
        """Snap a 4 to 16 cm run of near side returns (an upright) to the rack layout to correct x drift."""
        if abs(vx) < 0.3:  # run lengths mean nothing while hovering
            self.run[side] = None
            return
        if 0.3 < r < self.near_max:
            run = self.run[side]
            self.run[side] = [x_vio, x_vio, 1, r] if run is None else [run[0], x_vio, run[2] + 1, run[3] + r]
            return
        run, self.run[side] = self.run[side], None
        if run is None or run[2] < 2:
            return
        length = abs(run[1] - run[0]) + abs(vx) * 0.02  # one sample spacing at 50 Hz
        if not 0.04 <= length <= 0.16:
            return
        x_c = 0.5 * (run[0] + run[1]) + self.origin[0]
        k = int(np.argmin(np.abs(self.uprights - (x_c - self.dx))))
        err = x_c - self.uprights[k]
        if abs(err - self.dx) > 0.6:
            return
        self.dx += 0.5 * (err - self.dx)
        self.n_rack += 1
        self.pub_dx.publish(Float64(data=float(self.dx)))

    def on_down(self, msg):
        if msg.min_range <= msg.range <= msg.max_range:
            self.down = msg.range

    def update_lock(self, t, y_vio):
        """Filter the VIO y offset with one left/right lidar pair that both hit pallet faces."""
        if not self.lock or self.side["L"] is None or self.side["R"] is None:
            return
        (tl, dl), (tr, dr) = self.side["L"], self.side["R"]
        if abs(tl - tr) > 0.03 or abs(t - tl) > 0.1 or abs(dl + dr - self.pair_sum) > self.width_tol:
            return
        y_true = (dr - dl) / 2
        dt = 0.02 if self.t_lock is None else min(max(t - self.t_lock, 0.0), 0.5)
        self.t_lock = t
        self.delta += (y_vio - y_true - self.delta) * min(1.0, dt / self.tau)
        self.n_lock += 1
        if self.n_lock % 50 == 1:
            self.pub_delta.publish(Float64(data=float(self.delta)))

    def on_offset(self, msg):
        self.offset = msg.data

    def on_odom(self, msg):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if t - self.last_t < self.min_dt:
            return
        if t < self.last_t:  # time went backwards: the VIO restarted
            self.reset_counter = (self.reset_counter + 1) % 256
        self.last_t = t
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        r_flu = quat_to_rot(q.w, q.x, q.y, q.z)
        self.update_lock(t, p.y)
        v_lin = msg.twist.twist.linear
        vx_w = v_lin.x if self.twist_world else (r_flu @ np.array([v_lin.x, v_lin.y, v_lin.z]))[0]
        self.last = (t, p.x, vx_w)
        pos = np.array([p.x - self.dx, p.y - self.delta, p.z])
        nav = Odometry()
        nav.header, nav.child_frame_id, nav.pose, nav.twist = msg.header, msg.child_frame_id, msg.pose, msg.twist
        nav.pose.pose.position.x = float(pos[0])
        nav.pose.pose.position.y = float(pos[1])
        if self.down is not None:  # flat floor: height from the down lidar
            nav.pose.pose.position.z = float((self.down - self.down_z) * r_flu[2, 2])
        self.pub_nav.publish(nav)
        v = msg.twist.twist.linear
        v_w = np.array([v.x, v.y, v.z]) if self.twist_world else r_flu @ np.array([v.x, v.y, v.z])
        w = msg.twist.twist.angular
        w_b = np.array([w.x, w.y, w.z])

        o = VehicleOdometry()
        o.timestamp = 0  # PX4 replaces 0 with its own time of arrival
        o.timestamp_sample = int((t - self.offset) * 1e6)  # us, PX4 time; PX4 clamps stamps from the future
        ned = self.get_parameter("pose_frame").value == "ned"
        o.pose_frame = VehicleOdometry.POSE_FRAME_NED if ned else VehicleOdometry.POSE_FRAME_FRD
        o.position = (F @ pos).astype(np.float32).tolist()
        o.q = rot_to_quat(F @ r_flu @ F).astype(np.float32).tolist()
        o.velocity_frame = VehicleOdometry.VELOCITY_FRAME_NED if ned else VehicleOdometry.VELOCITY_FRAME_FRD
        o.velocity = (F @ v_w).astype(np.float32).tolist()
        o.angular_velocity = (F @ w_b).astype(np.float32).tolist()
        sp, sa, sv = self.std
        o.position_variance = [sp ** 2] * 3
        o.orientation_variance = [sa ** 2] * 3
        o.velocity_variance = [sv ** 2] * 3
        o.reset_counter = self.reset_counter
        o.quality = 100
        self.pub.publish(o)


def main():
    rclpy.init()
    node = VioBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
