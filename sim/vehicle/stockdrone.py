"""The stock-counting quad in Pegasus: Charpi body, design mass and rotors, lidars and a ROS 2 backend."""
import os
import sys

import numpy as np
from pegasus.simulator.logic.backends.backend import Backend
from pegasus.simulator.logic.backends.px4_mavlink_backend import PX4MavlinkBackend, PX4MavlinkBackendConfig
from pegasus.simulator.logic.dynamics import LinearDrag
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
from pegasus.simulator.logic.sensors import IMU, Barometer
from pegasus.simulator.logic.sensors.sensor import Sensor
from pegasus.simulator.logic.thrusters import QuadraticThrustCurve
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
from pxr import Gf, UsdPhysics
from scipy.spatial.transform import Rotation

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import design as d  # noqa: E402

PX4_OVERLAY = os.path.join(os.path.dirname(HERE), "px4", "overlay")


class StockDrone(Multirotor):
    """Multirotor whose rotor forces act on /body at the design rotor positions."""

    def update(self, dt):
        if self._backends:
            ref = self._backends[0].input_reference()
        else:
            ref = [0.0] * 4
        self._thrusters.set_input_reference(ref)
        forces, _, yaw_moment = self._thrusters.update(self._state, dt)
        for i in range(4):
            self.apply_force([0.0, 0.0, forces[i]], pos=list(d.ROTORS[i]), body_part="/body")
        self.apply_torque([0.0, 0.0, yaw_moment], "/body")
        self.apply_force(self._drag.update(self._state, dt), body_part="/body")
        for b in self._backends:
            b.update(dt)


class Rangefinder(Sensor):
    """Downward 1D lidar over the flat warehouse floor (z = 0); slant range along body -z, metres."""

    def __init__(self, update_rate=d.RANGE_RATE):
        super().__init__(sensor_type="Rangefinder", update_rate=update_rate)
        self._mount = np.array(d.RANGE_POS)
        self._state = {"distance": -1.0}

    @property
    def state(self):
        return self._state

    @Sensor.update_at_rate
    def update(self, state, dt):
        rot = Rotation.from_quat(state.attitude)  # body FLU -> world ENU
        p = state.position + rot.apply(self._mount)
        down = rot.apply([0.0, 0.0, -1.0])
        r = p[2] / -down[2] if down[2] < -0.5 else -1.0  # no return beyond 60 deg tilt
        if r > 0:
            r += d.RANGE_SIGMA * np.random.randn()
        self._state = {"distance": float(r)}
        return self._state


class SideRangefinder(Sensor):
    """Side 1D lidar (side +1 looks along body +y, -1 along -y) ray-cast against the rack layout."""

    def __init__(self, side, pallets, update_rate=d.RANGE_RATE):
        super().__init__(sensor_type="RangeLeft" if side > 0 else "RangeRight", update_rate=update_rate)
        import aisle_spec as spec
        self._spec = spec
        self._side = side
        self._mount = np.array([0.0, side * d.SIDE_RANGE_Y, d.SIDE_RANGE_Z])
        tag = "L" if side > 0 else "R"
        self._loads = {}  # level -> array of (x centre, top of load above the level base)
        for p in pallets:
            if p.side == tag:
                self._loads.setdefault(p.level, []).append((p.x, spec.PALLET_H + p.load_h))
        self._loads = {k: np.array(v) for k, v in self._loads.items()}
        self._uprights = spec.RACK_X0 + spec.BAY_PITCH * np.arange(spec.N_BAYS + 1)
        self._state = {"distance": -1.0}

    @property
    def state(self):
        return self._state

    def _hit(self, o, dvec):
        spec, s = self._spec, self._side
        if dvec[1] * s < 0.3:  # rolled or yawed too far to see the rack
            return -1.0
        best = []
        face, front = s * spec.WIDTH / 2, s * (spec.WIDTH / 2 + spec.FACE_GAP)
        t = (face - o[1]) / dvec[1]
        x, z = o[0] + t * dvec[0], o[2] + t * dvec[2]
        for lv, loads in self._loads.items():
            zb = spec.level_base_z(lv)
            near = np.abs(loads[:, 0] - x) <= spec.PALLET_W / 2
            if np.any(near & (z >= zb) & (z <= zb + loads[:, 1])):
                best.append(t)
                break
        t = (front - o[1]) / dvec[1]
        x, z = o[0] + t * dvec[0], o[2] + t * dvec[2]
        on_upright = np.min(np.abs(self._uprights - x)) <= spec.UPRIGHT_W / 2 and 0 <= z <= spec.HEIGHT
        on_beam = any(abs(z - k * spec.LEVEL_H) <= spec.BEAM_H / 2 for k in range(1, spec.LEVELS))
        if on_upright or (on_beam and self._uprights[0] <= x <= self._uprights[-1]):
            best.append(t)
        best.append((s * 2.75 - o[1]) / dvec[1])  # side wall
        return min(best)

    @Sensor.update_at_rate
    def update(self, state, dt):
        rot = Rotation.from_quat(state.attitude)
        o = state.position + rot.apply(self._mount)
        r = self._hit(o, rot.apply([0.0, float(self._side), 0.0]))
        if r > 0:
            r += d.RANGE_SIGMA * np.random.randn()
        self._state = {"distance": float(r)}
        return self._state


class StockPX4Backend(PX4MavlinkBackend):
    """PX4 MAVLink backend that also forwards the downward lidar as DISTANCE_SENSOR."""

    def __init__(self, config):
        super().__init__(config)
        self._range = None

    def update_sensor(self, sensor_type, data):
        if sensor_type == "Rangefinder":
            self._range = data["distance"]
        else:
            super().update_sensor(sensor_type, data)

    def update(self, dt):
        super().update(dt)
        if self._range is None or not self._received_first_hearbeat:
            return
        r = self._range
        self._range = None
        if not (d.RANGE_MIN <= r <= d.RANGE_MAX):
            r = d.RANGE_MAX + 1.0  # PX4 treats out-of-bounds readings as invalid
        self._connection.mav.distance_sensor_send(
            int(self._current_utime / 1000), int(d.RANGE_MIN * 100), int(d.RANGE_MAX * 100), int(round(r * 100)),
            0, 0, 25, int(round((d.RANGE_SIGMA * 100) ** 2)))  # laser, id 0, PITCH_270 = down, variance cm^2


def set_body_mass(stage, prefix):
    """Override the Charpi body mass and inertia with the design values (before world.reset())."""
    prim = stage.GetPrimAtPath(prefix + "/body")
    m = UsdPhysics.MassAPI.Apply(prim)
    m.GetMassAttr().Set(d.MASS - 4 * 0.00245)  # props keep their own small mass
    m.GetDiagonalInertiaAttr().Set(Gf.Vec3f(*d.INERTIA))
    m.GetPrincipalAxesAttr().Set(Gf.Quatf(1, 0, 0, 0))
    m.GetCenterOfMassAttr().Set(Gf.Vec3f(*d.COM))


def scale_props(stage, prefix, scale=7.0 / 4.94):
    """Scale the 5 in prop visuals to 7 in, in the disc plane (local x and z; local +y is the motor axis)."""
    from pxr import UsdGeom
    for name in ("prop_front_right_1", "prop_rear_left_1", "prop_front_left_1", "prop_rear_right_1"):
        prim = stage.GetPrimAtPath(f"{prefix}/{name}/visuals")
        if prim.IsValid():
            UsdGeom.Xformable(prim).AddScaleOp(opSuffix="propsize").Set(Gf.Vec3f(scale, 1.0, scale))


def hide_airframe(stage, prefix):
    """Hide the airframe's instanced "visuals" roots from all cameras and return how many were hidden."""
    from pxr import Usd, UsdGeom
    n = 0
    for prim in Usd.PrimRange(stage.GetPrimAtPath(prefix)):
        if prim.GetName() == "visuals" and prim.IsA(UsdGeom.Imageable):
            UsdGeom.Imageable(prim).MakeInvisible()
            n += 1
    return n


class RosSimBackend(Backend):
    """Publishes the IMU, ground truth, clock and PX4 time offset with simulation-time stamps."""

    def __init__(self, px4_backend, node, gt_every=2):
        super().__init__(config=None)
        from nav_msgs.msg import Odometry
        from rosgraph_msgs.msg import Clock
        from sensor_msgs.msg import Imu
        from sensor_msgs.msg import Range
        from std_msgs.msg import Float64
        self._Range = Range
        self._pub_range = {k: node.create_publisher(Range, topic, 10) for k, topic in
                           (("Rangefinder", "/range"), ("RangeLeft", "/range_left"), ("RangeRight", "/range_right"))}
        self._px4 = px4_backend
        self._node = node
        self._Imu, self._Odom, self._Clock, self._F64 = Imu, Odometry, Clock, Float64
        qos = 50
        self._pub_imu = node.create_publisher(Imu, "/imu", qos)
        self._pub_gt = node.create_publisher(Odometry, "/gt/odom", qos)
        self._pub_clock = node.create_publisher(Clock, "/clock", 10)
        self._pub_off = node.create_publisher(Float64, "/sim/px4_time_offset", 10)
        self._gt_every = gt_every
        self._step = 0
        self._world = PegasusInterface().world

    def _stamp(self, stamp):
        """Write the current simulation time into a builtin_interfaces/Time."""
        t = self._world.current_time
        stamp.sec = int(t)
        stamp.nanosec = int(round((t - int(t)) * 1e9)) % 1_000_000_000
        return t

    def update_sensor(self, sensor_type, data):
        if sensor_type in ("Rangefinder", "RangeLeft", "RangeRight"):
            r = self._Range()
            self._stamp(r.header.stamp)
            r.header.frame_id = {"Rangefinder": "range", "RangeLeft": "range_left", "RangeRight": "range_right"}[sensor_type]
            r.radiation_type = r.INFRARED
            r.min_range, r.max_range, r.range = d.RANGE_MIN, d.RANGE_MAX, float(data["distance"])
            self._pub_range[sensor_type].publish(r)
            return
        if sensor_type != "IMU":
            return
        m = self._Imu()
        self._stamp(m.header.stamp)
        m.header.frame_id = "imu"
        w = data["angular_velocity"]  # FRD
        a = data["linear_acceleration"]  # FRD specific force, z = -g at rest
        m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z = float(w[0]), float(-w[1]), float(-w[2])
        m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z = (
            float(a[0]), float(-a[1]), float(-a[2]))
        m.orientation_covariance[0] = -1.0
        self._pub_imu.publish(m)

    def update_state(self, state):
        self._step += 1
        if self._step % self._gt_every:
            return
        o = self._Odom()
        self._stamp(o.header.stamp)
        o.header.frame_id, o.child_frame_id = "world", "body"
        p, q, v, w = state.position, state.attitude, state.linear_velocity, state.angular_velocity
        o.pose.pose.position.x, o.pose.pose.position.y, o.pose.pose.position.z = map(float, p)
        o.pose.pose.orientation.x, o.pose.pose.orientation.y, o.pose.pose.orientation.z, o.pose.pose.orientation.w = \
            map(float, q)
        o.twist.twist.linear.x, o.twist.twist.linear.y, o.twist.twist.linear.z = map(float, v)
        o.twist.twist.angular.x, o.twist.twist.angular.y, o.twist.twist.angular.z = map(float, w)
        self._pub_gt.publish(o)

    def update(self, dt):
        c = self._Clock()
        self._stamp(c.clock)
        self._pub_clock.publish(c)
        if self._step % 25 == 0 and self._px4 is not None:
            off = self._world.current_time - self._px4._current_utime * 1e-6
            self._pub_off.publish(self._F64(data=float(off)))

    def input_reference(self):
        return [0.0] * 4

    def update_graphical_sensor(self, sensor_type, data):
        pass

    def start(self):
        pass

    def stop(self):
        pass

    def reset(self):
        pass


def make_config(node, imu_rate=250.0, px4=True, pallets=None):
    cfg = MultirotorConfig()
    cfg.thrust_curve = QuadraticThrustCurve({
        "num_rotors": 4,
        "rotor_constant": [d.K_T] * 4,
        "rolling_moment_coefficient": [d.K_Q] * 4,
        "rot_dir": list(d.ROT_DIR),
        "min_rotor_velocity": [0.0] * 4,
        "max_rotor_velocity": [d.W_MAX] * 4,
    })
    cfg.drag = LinearDrag(list(d.DRAG))
    # GPS-denied: no GPS or magnetometer
    cfg.sensors = [Barometer(), IMU({"update_rate": imu_rate}), Rangefinder()]
    if pallets is not None:  # side lidars need the rack layout
        cfg.sensors += [SideRangefinder(1, pallets), SideRangefinder(-1, pallets)]
    backends = []
    px4_backend = None
    if px4:
        px4_backend = StockPX4Backend(PX4MavlinkBackendConfig({
            "vehicle_id": 0,
            "px4_autolaunch": True,
            "px4_dir": PX4_OVERLAY,
            "px4_vehicle_model": "pegasus_stockdrone",
            "enable_lockstep": True,
            "num_rotors": 4,
            "input_offset": [0.0] * 4,
            "input_scaling": [d.W_MAX - d.W_IDLE] * 4,
            "zero_position_armed": [d.W_IDLE] * 4,
            "update_rate": imu_rate,
        }))
        backends.append(px4_backend)
    backends.append(RosSimBackend(px4_backend, node))
    cfg.backends = backends
    cfg.graphical_sensors = []
    cfg.graphs = []
    return cfg


def spawn(stage, node, position=(0.0, 0.0, 0.02), yaw_deg=0.0, prefix="/World/stockdrone", px4=True, pallets=None):
    cfg = make_config(node, px4=px4, pallets=pallets)
    q = Rotation.from_euler("z", yaw_deg, degrees=True).as_quat()  # [x, y, z, w] as Pegasus wants
    drone = StockDrone(prefix, d.CHARPI_USD, 0, list(position), list(q), config=cfg)
    set_body_mass(stage, prefix)
    scale_props(stage, prefix)
    return drone
