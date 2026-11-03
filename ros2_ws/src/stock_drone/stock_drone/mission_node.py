"""Offboard mission: wait for a VIO estimate, arm, take off, fly the serpentine, land (use_sim_time:=true in sim)."""
import enum
import math

import numpy as np
import rclpy
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleLocalPosition, VehicleStatus
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from nav_msgs.msg import Odometry
from std_msgs.msg import String

from stock_drone import trajectory


class Phase(enum.Enum):
    WAIT_ESTIMATE = 0
    PRESTREAM = 1
    ARMING = 2
    TAKEOFF = 3
    MISSION = 4
    LAND = 5
    DONE = 6


PX4_QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                     history=HistoryPolicy.KEEP_LAST, depth=1)


class MissionNode(Node):
    def __init__(self):
        super().__init__("mission")
        p = self.declare_parameter
        p("home", [-3.0, 0.0])
        p("heights", [0.5, 2.16, 3.76, 5.36, 6.96])  # m, label heights from aisle_spec
        p("cam_z_offset", 0.06)  # m, barcode camera above the body origin
        p("x_start", 0.5)
        p("x_end", 79.5)
        p("v_scan", 1.0)
        p("v_transit", 2.0)
        p("v_climb", 0.5)
        p("a_max", 1.0)
        p("j_max", 2.0)
        p("hold", 4.0)
        p("takeoff_z", 1.0)
        p("rate_hz", 50.0)
        p("status_topic", "/fmu/out/vehicle_status_v1")
        # dock lat, lon, alt, only for a ground-station map; anything but 3 values = none
        p("ekf_origin", [0.0])
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        heights = [h - g("cam_z_offset") for h in g("heights")]
        self.home = np.array([*g("home"), 0.0])
        self.takeoff_z = g("takeoff_z")
        self.traj = trajectory.serpentine(g("x_start"), g("x_end"), heights, home=tuple(g("home")),
                                          v_scan=g("v_scan"), v_transit=g("v_transit"), v_climb=g("v_climb"),
                                          a_max=g("a_max"), j_max=g("j_max"), hold=g("hold"),
                                          takeoff_z=self.takeoff_z)
        self.get_logger().info(f"trajectory: {self.traj.duration:.0f} s, {self.traj.length():.0f} m "
                               f"({self.traj.length('scan'):.0f} m scanning)")

        self.pub_mode = self.create_publisher(OffboardControlMode, "/fmu/in/offboard_control_mode", PX4_QOS)
        self.pub_sp = self.create_publisher(TrajectorySetpoint, "/fmu/in/trajectory_setpoint", PX4_QOS)
        self.pub_cmd = self.create_publisher(VehicleCommand, "/fmu/in/vehicle_command", PX4_QOS)
        self.pub_phase = self.create_publisher(String, "/mission/phase", 10)
        self.pub_ref = self.create_publisher(Odometry, "/mission/reference", 10)  # aisle frame, for logging
        self.create_subscription(VehicleLocalPosition, "/fmu/out/vehicle_local_position", self.on_lpos, PX4_QOS)
        self.create_subscription(VehicleStatus, g("status_topic"), self.on_status, PX4_QOS)

        self.lpos = None
        self.status = None
        self.phase = Phase.WAIT_ESTIMATE
        self.t_phase = self.now()
        self.p0 = None  # NED position at arming
        self.yaw0 = 0.0
        self.takeoff = None
        self.create_timer(1.0 / g("rate_hz"), self.tick)

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def on_lpos(self, msg):
        self.lpos = msg

    def on_status(self, msg):
        self.status = msg

    def to_ned(self, v, is_point=True):
        """Aisle frame to PX4 local NED: p0 + R_z(yaw0) diag(1, -1, -1) (v - home), or rotation only for vectors."""
        c, s = math.cos(self.yaw0), math.sin(self.yaw0)
        rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        flip = np.diag([1.0, -1.0, -1.0])
        d = (v - self.home) if is_point else v
        out = rz @ flip @ d
        return out + self.p0 if is_point else out

    def set_phase(self, ph):
        self.phase = ph
        self.t_phase = self.now()
        self.get_logger().info(f"phase -> {ph.name}")

    def command(self, cmd, p1=0.0, p2=0.0, p7=0.0, p5=0.0, p6=0.0):
        m = VehicleCommand()
        m.timestamp = 0  # PX4 stamps 0 with its own time on arrival
        m.command, m.param1, m.param2, m.param7 = cmd, float(p1), float(p2), float(p7)
        m.param5, m.param6 = float(p5), float(p6)
        m.target_system, m.target_component, m.source_system, m.source_component = 1, 1, 1, 1
        m.from_external = True
        self.pub_cmd.publish(m)

    def stream(self, p_aisle, v_aisle=np.zeros(3), a_aisle=np.zeros(3)):
        ref = Odometry()
        ref.header.stamp = self.get_clock().now().to_msg()
        ref.header.frame_id = "aisle"
        ref.pose.pose.position.x, ref.pose.pose.position.y, ref.pose.pose.position.z = map(float, p_aisle)
        ref.twist.twist.linear.x, ref.twist.twist.linear.y, ref.twist.twist.linear.z = map(float, v_aisle)
        self.pub_ref.publish(ref)
        ts = 0
        mode = OffboardControlMode()
        mode.timestamp, mode.position, mode.velocity, mode.acceleration = ts, True, True, True
        self.pub_mode.publish(mode)
        sp = TrajectorySetpoint()
        sp.timestamp = ts
        sp.position = self.to_ned(p_aisle).astype(np.float32).tolist()
        sp.velocity = self.to_ned(v_aisle, False).astype(np.float32).tolist()
        sp.acceleration = self.to_ned(a_aisle, False).astype(np.float32).tolist()
        sp.yaw = float(self.yaw0)  # keep facing +x of the aisle
        sp.yawspeed = 0.0
        self.pub_sp.publish(sp)

    def estimate_ok(self):
        lp = self.lpos
        # no heading check: EKF2 never marks an FRD vision heading as north-aligned
        return lp is not None and lp.xy_valid and lp.z_valid and lp.v_xy_valid and lp.v_z_valid

    def tick(self):
        self.pub_phase.publish(String(data=self.phase.name))
        t = self.now() - self.t_phase
        if self.phase == Phase.WAIT_ESTIMATE:
            if self.estimate_ok() and self.status is not None:
                self.p0 = np.array([self.lpos.x, self.lpos.y, self.lpos.z])
                self.yaw0 = self.lpos.heading
                self.set_phase(Phase.PRESTREAM)
            return
        if self.phase in (Phase.PRESTREAM, Phase.ARMING):
            self.stream(self.home.copy())
            origin = self.get_parameter("ekf_origin").value
            if self.phase == Phase.PRESTREAM and len(origin) == 3 and int(t * 50) % 10 == 0 and t < 1.0:
                self.command(VehicleCommand.VEHICLE_CMD_SET_GPS_GLOBAL_ORIGIN, p5=origin[0], p6=origin[1], p7=origin[2])
            if self.phase == Phase.PRESTREAM and t > 1.5:  # PX4 wants setpoints before the mode switch
                self.command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)  # custom mode: offboard
                self.command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
                self.set_phase(Phase.ARMING)
            elif self.phase == Phase.ARMING:
                armed = self.status.arming_state == VehicleStatus.ARMING_STATE_ARMED
                offboard = self.status.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD
                if armed and offboard:
                    self.takeoff = trajectory.DoubleS(self.takeoff_z, 0.6, 1.0, 2.0)
                    self.set_phase(Phase.TAKEOFF)
                elif t > 3.0:
                    self.set_phase(Phase.PRESTREAM)
            return
        if self.phase == Phase.TAKEOFF:
            s, v, a = self.takeoff.sample(t)
            self.stream(self.home + [0, 0, s], np.array([0, 0, v]), np.array([0, 0, a]))
            if t > self.takeoff.T + 3.0:
                self.set_phase(Phase.MISSION)
            return
        if self.phase == Phase.MISSION:
            p, v, a, _, _ = self.traj.sample(t)
            self.stream(p, v, a)
            if t > self.traj.duration:
                self.command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
                self.set_phase(Phase.LAND)
            return
        if self.phase == Phase.LAND:
            if self.status.arming_state != VehicleStatus.ARMING_STATE_ARMED:
                self.set_phase(Phase.DONE)


def main():
    rclpy.init()
    node = MissionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
