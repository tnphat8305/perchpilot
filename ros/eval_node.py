"""Score repeated PerchPilot trials on one running PX4/Gazebo session."""
from __future__ import annotations

import argparse
from collections import deque
from itertools import product
import json
import math
from pathlib import Path
import random
import re
import subprocess
import time

import numpy as np

from perch_core import read_px4_param, rot_to_quat
from tag_camera_sim import parse_pose_v, simulated_tag_measurements, start_tag_camera

PROJECT = Path(__file__).resolve().parents[1]
TRUTH_ROOT = Path('/home/sup/data/perchpilot/sitl_eval')
WORLD = '/world/perch_world'
DRONE, CART = 'x500_perch_7', 'perch_cart'
CART_START = np.array([8.0, 0.0, 0.45])
IMPACT_TOPICS = {
    'ground': WORLD + '/model/ground_plane/link/link/sensor/eval_ground_contact/contact',
    'cart': WORLD + f'/model/{CART}/link/body/sensor/eval_cart_contact/contact',
    'plate': WORLD + f'/model/{CART}/link/plate/sensor/eval_plate_contact/contact',
}
TOUCH_FIELDS = ('pad_u_m', 'pad_w_m', 'pad_n_m', 'align_deg', 'vn_mps', 'vt_mps', 'omega_radps')
# HoldPad >= review D01/D02: vn/vt at the pad point (as the GPU sim), CoG values and the footprint support point
TOUCH_FIELDS_EXTRA = ('vn_cog_mps', 'vt_cog_mps', 'support_u_m', 'support_w_m')
PLATE_HALF = 0.225  # perch_world.sdf, half of the collision box width
HOLD_MAX_GAP_S = 0.05  # hold_truth is sampled every 0.005 s


def grid_values(text):
    try:
        # Comma-separated with decimal points, or semicolons with either decimal separator.
        parts = text.split(';') if ';' in text else text.split(',')
        values = tuple(float(part.replace(',', '.')) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError('expected a list of numbers') from exc
    if any(not math.isfinite(value) or value < 0 for value in values) or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError('grid values must be finite, non-negative and distinct')
    return values


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True, help='unique run and result name')
    parser.add_argument('--policy-run', type=Path, required=True, help='C1 policy directory')
    parser.add_argument('--trials', type=int, default=50)
    parser.add_argument('--eval-seed', type=int, default=12345)
    parser.add_argument('--speeds', type=grid_values, default=(0.0, 1.5, 3.0), help='cart speeds in m/s')
    parser.add_argument('--mu', type=grid_values, default=(0.9,))
    parser.add_argument('--f-hold-n', type=grid_values, default=(60.75,))
    parser.add_argument('--rest-s', type=float, default=10.0, help='seconds of sim time after disarm')
    parser.add_argument('--timeout-s', type=float, default=45.0, help='trial wall-clock timeout')
    parser.add_argument('--q2', action='store_true',
                        help='Q2 batch: a touch outside the envelope does not end the trial; score VERIFY decisions')
    parser.add_argument('--pad-grace-s', type=float, default=0.05,
                        help='sim seconds a body-plate impact may precede the pad touch before it counts as a crash')
    parser.add_argument('--recovery-s', type=float, default=45.0, help='return-to-hover wall-clock timeout')
    parser.add_argument('--startup-s', type=float, default=90.0)
    parser.add_argument('--check-config', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--continue', dest='cont', action='store_true',
                        help='append to an existing run after a SITL restart (skips the trials already in trials.jsonl)')
    args = parser.parse_args(argv)
    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_.-]*', args.name):
        parser.error('--name must start with a letter or digit and contain only letters, digits, _, . or -')
    if args.trials < 1 or not 0 <= args.eval_seed < 2**64:
        parser.error('--trials must be positive and --eval-seed must be an unsigned 64-bit integer')
    if max(args.speeds) > 3.0:
        parser.error('--speeds must be within [0, 3] m/s')
    if any(not math.isfinite(value) or value <= 0 for value in
           (args.rest_s, args.timeout_s, args.recovery_s, args.startup_s)) or args.timeout_s <= args.rest_s:
        parser.error('durations must be finite and positive, with --timeout-s greater than --rest-s')
    return args


def trial_grid(args):
    rng = random.Random(args.eval_seed)
    cells = list(product(args.speeds, args.mu, args.f_hold_n))
    for index in range(args.trials):
        if index % len(cells) == 0:
            rng.shuffle(cells)
        speed, mu, force = cells[index % len(cells)]
        yield {'speed_mps': speed, 'mu': mu, 'f_hold_n': force}


def wilson(k, n):
    if not 0 <= k <= n:
        raise ValueError('expected 0 <= k <= n')
    if n == 0:
        return {'p': None, 'lo': None, 'hi': None, 'n': 0}
    z = 1.96
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return {'p': round(p, 4), 'lo': round(max(0.0, c - h), 4), 'hi': round(min(1.0, c + h), 4), 'n': n}


def score_touch(touch):
    return (touch['align_deg'] <= 25.0 and touch['vn_mps'] < 1.5 and touch['vt_mps'] < 1.0
            and touch['omega_radps'] < 5.0 and max(abs(touch['pad_u_m']), abs(touch['pad_w_m'])) <= PLATE_HALF)


class TrialScore:
    """Consume the JSONL records in arrival order, with sim time for the hold window."""

    def __init__(self, rest_s, pad_grace_s=0.05, q2=False, q2_abort_watch_s=3.0):
        self.rest_s, self.pad_grace_s, self.q2_mode, self.q2_watch = rest_s, pad_grace_s, q2, q2_abort_watch_s
        self.q2 = self.q2_end_t = None
        self.plate_impact_collisions = None
        self.slip_path = 0.0
        self.touch = self.prev_diag = self.cut_t = self.last_hold_t = self.plate_impact = None
        self.crash = None
        self.aborted = self.hold_failed = self.hold_gap = self.rest_done = False
        self.perch_s = self.tag_lost_s = 0.0

    def feed(self, row):
        kind, sim_t = row['kind'], row.get('sim_t')
        if kind == 'touch' and self.touch is None:
            self.touch = {key: row[key] for key in TOUCH_FIELDS + TOUCH_FIELDS_EXTRA if key in row}
            self.touch['sim_t'] = sim_t
            self.touch['v_point'] = 'pad' if 'vn_cog_mps' in row else 'cog'
            # D04: a body/plate impact more than pad_grace_s (by sim stamp) before the pad touch is a miss
            if self.plate_impact is not None and sim_t - self.plate_impact > self.pad_grace_s:
                self.crash = self.crash or 'plate_missed_pad'
            self.touch['pad_error_cm'] = 100 * math.hypot(*(row[key] for key in TOUCH_FIELDS[:3]))
            if not score_touch(self.touch) and not self.q2_mode:   # Q2 batches score what happens after touch
                self.crash = self.crash or 'touch_envelope'
        elif kind == 'impact':
            if row['surface'] == 'plate':
                if self.plate_impact is None or sim_t < self.plate_impact:
                    self.plate_impact = sim_t
                    self.plate_impact_collisions = row.get('collisions')
            else:
                self.crash = self.crash or row['surface']
        elif kind == 'diag':
            diag = row['diag']
            if self.prev_diag is not None and self.prev_diag['phase'] == 'PERCH':
                dt = max(0.0, diag['t'] - self.prev_diag['t'])
                self.perch_s += dt
                if not self.prev_diag['tag_visible']:
                    self.tag_lost_s += dt
            self.prev_diag = diag
            self.aborted |= diag['phase'] == 'ABORT'
            q2 = diag.get('q2')
            if q2 and (self.q2 is None or q2.get('decision') != self.q2.get('decision')):
                self.q2 = q2           # a later SELF_ABORT (contact lost before disarm) overrides OFF
                if q2.get('decision') in ('ABORT', 'SELF_ABORT') and self.q2_end_t is None:
                    self.q2_end_t = sim_t
            if diag['phase'] == 'REST' and not diag['armed'] and self.cut_t is None:
                self.cut_t = sim_t
                self.last_hold_t = self.last_hold_t_prev = sim_t
        elif kind == 'hold' and self.cut_t is not None and sim_t > self.last_hold_t and not self.rest_done:
            self.hold_gap |= sim_t - self.last_hold_t > HOLD_MAX_GAP_S
            self.last_hold_t_prev, self.last_hold_t = self.last_hold_t, sim_t
            self.slip_path += row.get('slip_mps', 0.0) * (sim_t - self.last_hold_t_prev)
            self.hold_failed |= not row['engaged'] or row['margin_n'] < 0.0
            if not row['engaged']:
                self.crash = self.crash or 'detached_after_disarm'
            self.rest_done = sim_t - self.cut_t >= self.rest_s - 1e-9

    def outcome(self, sim_t):
        if self.crash:
            return 'crash'
        # D04: decide by stamps; wait 0.1 s more than the grace so a late-arriving touch row is still matched
        if self.touch is None and self.plate_impact is not None and sim_t - self.plate_impact >= self.pad_grace_s + 0.1:
            self.crash = 'plate_missed_pad'
            return 'crash'
        if self.aborted:
            return 'abort'
        if self.q2_end_t is not None and sim_t - self.q2_end_t >= self.q2_watch:
            return 'q2_abort' if self.q2['decision'] == 'ABORT' else 'self_abort'
        if self.rest_done:
            return 'rest'
        return None

    def result(self, outcome):
        motors_off = self.cut_t is not None
        held = None
        if motors_off:
            if self.hold_failed or outcome == 'crash':
                held = False
            elif self.rest_done and not self.hold_gap:
                held = True
        touch_ok = self.touch is not None and score_touch(self.touch)
        return {'outcome': outcome, 'crash_reason': self.crash, 'touch': self.touch,
                'touch_success': touch_ok,
                # D04: a cycle needs a valid touch (no missing / out-of-envelope touch) and no crash
                'motors_off': motors_off, 'held': held, 'success': held is True and touch_ok and self.crash is None,
                'plate_impact_collisions': self.plate_impact_collisions,
                'false_safe': not held if held is not None else None,
                'fell_after_off': motors_off and self.crash == 'detached_after_disarm',
                # Q2 valid perch (held_star): motors off, REST complete on the plate, total slip < 2 cm
                'valid_perch': motors_off and self.rest_done and self.crash is None and self.slip_path < 0.02,
                'rest_slip_m': round(self.slip_path, 4), 'q2': self.q2,
                'perch_s': round(self.perch_s, 4), 'tag_lost_s': round(self.tag_lost_s, 4),
                'tag_lost_pct': 100 * self.tag_lost_s / self.perch_s if self.perch_s else None}


def off_commanded(trials):
    # v1.2 includes contact lost after OFF was commanded, before PX4 finished disarming.
    return [t for t in trials if (t.get('q2') or {}).get('decision') == 'OFF'
            or 'after MOTORS_OFF' in ((t.get('q2') or {}).get('why') or '')]


def dashboard_batch(trials, q2=False):
    if q2 or any(t.get('q2') for t in trials):
        cut = off_commanded(trials)
        flags = [t.get('outcome') in ('crash', 'self_abort') for t in cut]
        metric = 'v1.2'
    else:
        cut = [t for t in trials if t.get('motors_off')]
        flags = [t.get('false_safe') for t in cut]
        metric = 'disarm'
    return {'trials': len(trials), 'touch_pass': sum(t.get('touch_success') is True for t in trials),
            'false_safe': None if None in flags else sum(flags), 'off': len(cut), 'metric': metric}


def summarize(trials):
    n = len(trials)
    cut = [trial for trial in trials if trial['motors_off']]
    unknown = sum(trial['held'] is None for trial in cut)
    touches = [trial['touch'] for trial in trials if trial['touch'] is not None]
    perch_s, lost_s = (sum(trial[key] for trial in trials) for key in ('perch_s', 'tag_lost_s'))
    result = {key: wilson(sum(trial.get(key) is True for trial in trials), n)
              for key in ('touch_success', 'success', 'motors_off', 'recovered', 'cycle_complete')}
    # re-arm on the plate after REST (denominator: trials whose motors were cut on the plate)
    result['rearm'] = wilson(sum(trial.get('rearm_ok') is True for trial in cut), len(cut))
    result['valid_perch'] = wilson(sum(trial.get('valid_perch') is True for trial in trials), n)
    result['false_safe_fall'] = wilson(sum(trial.get('fell_after_off') is True for trial in cut), len(cut))
    # spec v1.2: denominator = trials where MOTORS_OFF was commanded (incl. contact lost before PX4 finished
    # disarming, logged by the node as SELF_ABORT 'after MOTORS_OFF was commanded'); any loss of contact counts
    off_cmd = off_commanded(trials)
    result['off_commanded'] = wilson(len(off_cmd), n)
    result['false_safe_v12'] = wilson(sum(t['outcome'] in ('crash', 'self_abort') for t in off_cmd), len(off_cmd))
    for outcome in ('q2_abort', 'self_abort'):
        result[outcome] = wilson(sum(trial['outcome'] == outcome for trial in trials), n)
    aborted = [t for t in trials if t['outcome'] in ('q2_abort', 'self_abort')]
    result['recovered_after_abort'] = wilson(sum(t.get('recovered') is True for t in aborted), len(aborted))
    result['sitl_restarts'] = sum(trial.get('sitl_restart_after', False) is True for trial in trials)
    for outcome in ('crash', 'abort', 'timeout'):
        result[outcome] = wilson(sum(trial['outcome'] == outcome for trial in trials), n)
    result['held_if_motors_cut'] = wilson(sum(trial['held'] is True for trial in cut), n)
    result['false_safe'] = ({'p': None, 'lo': None, 'hi': None, 'n': len(cut)} if unknown else
                            wilson(sum(trial['false_safe'] is True for trial in cut), len(cut)))
    result['held_unknown_n'] = unknown
    result['perch_s'], result['tag_lost_s'] = round(perch_s, 4), round(lost_s, 4)
    result['tag_lost_pct'] = round(100 * lost_s / perch_s, 4) if perch_s else None
    result['touch_n'] = len(touches)
    for field, key in (('vn_mps', 'touch_vn_mean'), ('vt_mps', 'touch_vt_mean'), ('align_deg', 'touch_align_deg_mean'),
                       ('omega_radps', 'touch_omega_radps_mean')):
        result[key] = round(sum(t[field] for t in touches) / len(touches), 4) if touches else None
    result['touch_pos_cm_mean'] = round(sum(t['pad_error_cm'] for t in touches) / len(touches), 4) if touches else None
    return result


def stamp_s(msg):
    stamp = msg.header.stamp
    return stamp.sec + stamp.nsec * 1e-9


class Gazebo:
    def __init__(self):
        from gz.msgs10.contacts_pb2 import Contacts
        from gz.msgs10.double_v_pb2 import Double_V
        from gz.msgs10.pose_v_pb2 import Pose_V
        from gz.msgs10.twist_pb2 import Twist
        from gz.transport13 import Node

        self.Double_V, self.Twist = Double_V, Twist
        self.pending = deque()
        self.node = Node()
        self.velocity = self.node.advertise(f'/model/{CART}/cmd_vel', Twist)
        self.params = self.node.advertise('/perch/hold_params', Double_V)
        for kind, topic, message in (
            ('pose', WORLD + '/dynamic_pose/info', Pose_V),
            ('hold', '/perch/hold_truth', Double_V),
            ('sensor', '/perch/hold_sensors', Double_V),
            ('touch', '/perch/touch_truth', Double_V),
            ('load', '/perch/load_truth', Double_V),
            ('slip', '/perch/slip_truth', Double_V),
            ('params', '/perch/hold_params_applied', Double_V),
            *((surface, topic, Contacts) for surface, topic in IMPACT_TOPICS.items()),
        ):
            if not self.node.subscribe(message, topic, lambda msg, kind=kind:
                                       self.pending.append((kind, msg, time.monotonic()))):
                raise RuntimeError(f'Gazebo subscription failed {topic}')

    def cart_speed(self, speed):
        msg = self.Twist()
        msg.linear.x = speed
        if not self.velocity.publish(msg):
            raise RuntimeError('Gazebo cart command failed')

    def reset_cart(self):
        # Keep the blocking service call out of the Python transport subscriber thread.
        x, y, z = CART_START
        pose = f'name: "{CART}" position {{x: {x} y: {y} z: {z}}} orientation {{w: 1}}'
        response = subprocess.run(['gz', 'service', '-s', WORLD + '/set_pose',
                                   '--reqtype', 'gz.msgs.Pose', '--reptype', 'gz.msgs.Boolean',
                                   '--timeout', '2000', '--req', pose], capture_output=True, text=True, timeout=5)
        if response.returncode or response.stdout.strip() != 'data: true':
            raise RuntimeError(f'Gazebo set_pose rejected {CART}: {response.stderr or response.stdout}')


class EvalNode:
    """Own the ROS evaluator and the single Gazebo truth subscription set."""

    def __init__(self, node, executor, args, log_dir):
        from rclpy.qos import DurabilityPolicy, QoSProfile
        from std_msgs.msg import Float64MultiArray, String

        self.node, self.executor, self.args, self.log_dir = node, executor, args, log_dir
        self.String, self.Float64MultiArray = String, Float64MultiArray
        self.gz = Gazebo()
        self.cmd = node.create_publisher(String, '/perchpilot/cmd', 10)
        # Jazzy's ros_gz_bridge has no Double_V <-> Float64MultiArray conversion.
        self.sensors = node.create_publisher(Float64MultiArray, '/perch/hold_sensors', 10)
        self.dashboard = node.create_publisher(
            String, '/perchpilot/eval', QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.batch = dashboard_batch([], q2=args.q2)
        node.create_subscription(String, '/perchpilot/diag', self.on_diag, 10)
        self.sim_t = self.hold_wall = self.pose_wall = self.sensor_wall = self.rx_wall = 0.0
        self.diag = self.diag_wall = self.pose = self.camera_sample = self.cart_v = self.params_applied = None
        self.log = self.score = self.trial = None
        self.recovering = None

    def take_camera_poses(self):
        sample, self.camera_sample = self.camera_sample, None
        return sample

    def record(self, row):
        if self.log is not None:
            # wall = CLOCK_MONOTONIC at arrival (shared with perch_node's --tick-log) for offline alignment
            wall = self.rx_wall if row['kind'] in ('pose', 'hold', 'touch', 'impact') else time.monotonic()
            self.log.write(json.dumps({**row, 'wall': round(wall, 5)}, allow_nan=False) + '\n')
        if self.score is not None:
            previous_touch = self.score.touch
            self.score.feed(row)
            if previous_touch is None and self.score.touch is not None:
                self.publish_dashboard(self.trial, touch=self.score.touch, touch_pass=score_touch(self.score.touch))

    def publish_dashboard(self, trial=None, **fields):
        self.dashboard.publish(self.String(data=json.dumps({'trial': trial, 'batch': self.batch, **fields})))

    def dashboard_result(self, trials):
        self.batch = dashboard_batch(trials, q2=self.args.q2)
        last = trials[-1] if trials else {}
        self.publish_dashboard(last.get('trial'), touch=last.get('touch'), touch_pass=last.get('touch_success'),
                               outcome=last.get('outcome'), cycle_complete=last.get('cycle_complete'))

    def on_diag(self, msg):
        self.diag = json.loads(msg.data)
        self.diag_wall = time.monotonic()
        if self.recovering is not None and self.diag.get('phase') == 'DEPART' and self.diag.get('armed'):
            self.recovering['rearm'] = True
        if self.diag.get('trial') == self.trial:
            self.record({'kind': 'diag', 'sim_t': self.sim_t, 'diag': self.diag})

    def drain_gz(self):
        while self.gz.pending:
            kind, msg, self.rx_wall = self.gz.pending.popleft()
            if kind == 'pose':
                poses = parse_pose_v(msg, {DRONE, CART})
                if poses.keys() != {DRONE, CART}:
                    continue
                now, sim_t = time.monotonic(), stamp_s(msg)
                self.pose_wall = now
                if self.pose is not None and sim_t > self.pose[0]:
                    self.cart_v = (poses[CART][0] - self.pose[1][CART][0]) / (sim_t - self.pose[0])
                self.pose, self.camera_sample = (sim_t, poses), (self.rx_wall, poses)
                if self.log is not None:
                    self.record({'kind': kind, 'sim_t': sim_t, 'models': {
                        name: {'xyz': p.tolist(), 'wxyz': rot_to_quat(R).tolist()} for name, (p, R) in poses.items()}})
            elif kind == 'hold' and len(msg.data) == 4:
                t, engaged, margin, slip = msg.data
                self.sim_t, self.hold_wall = t, time.monotonic()
                self.record({'kind': kind, 'sim_t': t, 'engaged': bool(engaged), 'margin_n': margin, 'slip_mps': slip})
            elif kind == 'slip' and len(msg.data) == 4:
                if self.log is not None:     # C5 v2 slip sensor truth [mm per 5 ms sample, pad-plane u / w]
                    self.record({'kind': 'slip', 'sim_t': msg.data[0], 'd_u_mm': msg.data[1], 'd_w_mm': msg.data[2],
                                 'valid': msg.data[3]})
            elif kind == 'sensor' and len(msg.data) in (3, 5):
                self.sensor_wall = time.monotonic()
                self.sensors.publish(self.Float64MultiArray(data=msg.data))
            elif kind == 'touch' and len(msg.data) in (8, 12):
                self.record({'kind': kind, 'sim_t': msg.data[0],
                             **dict(zip(TOUCH_FIELDS + TOUCH_FIELDS_EXTRA, msg.data[1:]))})
            elif kind == 'load' and len(msg.data) == 4:
                if self.log is not None:     # HoldPad load-cell diagnostics: servo / collision normal force
                    self.record({'kind': 'load', 'sim_t': msg.data[0], 'servo_n': msg.data[1],
                                 'collision_n': msg.data[2], 'points': msg.data[3]})
            elif kind == 'params':
                self.params_applied = tuple(msg.data)
            elif kind in ('ground', 'cart', 'plate'):
                for contact in msg.contact:
                    names = [contact.collision1.name, contact.collision2.name]
                    if any(DRONE in name.split('::') for name in names):
                        self.record({'kind': 'impact', 'sim_t': stamp_s(msg), 'surface': kind, 'collisions': names})
                        break

    def wait(self, predicate, timeout, label):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.drain_gz()
            self.executor.spin_once(timeout_sec=0.01)
            result = predicate()
            if result:
                return result
        raise RuntimeError(f'timed out waiting for {label}')

    def hover_ready(self):
        # The node's "ready" is EKF2-based. After a crash the EKF can report a stable hover while the drone lies on
        # the ground (SITL, q2_none: 17 trials "crashed" at start), so the evaluator also checks the Gazebo truth.
        airborne = (self.pose is not None and self.pose[1][DRONE][0][2] > 1.0
                    and self.pose[1][DRONE][1][2, 2] > 0.9)
        return (self.diag is not None and self.diag.get('ready') and airborne and
                time.monotonic() - self.diag_wall < 0.3)

    def wait_for_hover(self, timeout):
        since = None

        def stable():
            nonlocal since
            if not self.hover_ready():
                since = None
                return False
            now = time.monotonic()
            if since is None:
                since = now
            return now - since >= 0.5

        self.wait(stable, timeout, 'stable EKF standby hover')

    def startup(self):
        def connected():
            return (self.diag is not None and self.pose is not None and time.monotonic() - self.hold_wall < 1.0
                    and self.cmd.get_subscription_count() and self.gz.velocity.has_connections()
                    and self.gz.params.has_connections() and time.monotonic() - self.sensor_wall < 1.0
                    and self.node.count_subscribers('/perch/hold_cmd'))

        self.wait(connected, self.args.startup_s, 'Gazebo truth, cart, HoldPad bridge and perch node')
        self.wait(lambda: set(IMPACT_TOPICS.values()) <= set(self.gz.node.topic_list()),
                  5.0, 'Gazebo collision topics')
        topics = {topic for topic, _ in self.node.get_subscriber_names_and_types_by_node('perchpilot_perch_node', '/')}
        allowed = {'/parameter_events', '/perch/tag_pose', '/perch/hold_sensors', '/perchpilot/cmd',
                   '/px4_7/fmu/out/vehicle_odometry', '/px4_7/fmu/out/vehicle_status',
                   '/px4_7/fmu/out/vehicle_control_mode', '/px4_7/fmu/out/sensor_combined'}
        if topics - allowed:
            raise RuntimeError(f'unexpected policy subscriptions {sorted(topics - allowed)}')
        self.gz.cart_speed(0.0)
        self.wait_for_hover(self.args.startup_s)

    def setup_trial(self, config):
        self.gz.cart_speed(0.0)
        self.wait(lambda: self.cart_v is not None and np.linalg.norm(self.cart_v) < 0.05, 5.0, 'cart stop')
        self.gz.reset_cart()
        self.wait(lambda: np.linalg.norm(self.pose[1][CART][0] - CART_START) < 0.025
                  and np.linalg.norm(self.cart_v) < 0.05, 5.0, 'cart set_pose and settle')
        params = (config['mu'], config['f_hold_n'])
        self.params_applied = None
        next_send = 0.0

        def applied():
            nonlocal next_send
            now = time.monotonic()
            if self.params_applied == params:
                return True
            if now >= next_send:
                if not self.gz.params.publish(self.gz.Double_V(data=params)):
                    raise RuntimeError('Gazebo hold_params publish failed')
                next_send = now + 0.3
            return False

        self.wait(applied, 5.0, 'HoldPad runtime parameter acknowledgement')
        self.wait_for_hover(self.args.recovery_s)
        self.gz.cart_speed(config['speed_mps'])
        self.wait(lambda: abs(self.cart_v[0] - config['speed_mps']) < 0.15, 5.0, 'cart speed')

    def run_trial(self, index, config):
        self.setup_trial(config)
        self.trial = index
        self.score = TrialScore(self.args.rest_s, self.args.pad_grace_s, q2=self.args.q2)
        self.publish_dashboard(index)
        try:
            with (self.log_dir / f'truth_{index}.jsonl').open('w', buffering=1) as self.log:
                self.record({'kind': 'start', 'sim_t': self.sim_t, 'trial': index, **config})
                # the node only accepts "start" while its standby hover is "ready" (flickers): resend until acked
                next_send = [0.0]

                def acked():
                    if self.diag['trial'] == self.trial:
                        return True
                    if time.monotonic() >= next_send[0]:
                        self.cmd.publish(self.String(data=f'start:{index}'))
                        next_send[0] = time.monotonic() + 0.5
                    return False

                self.wait(acked, 10.0, 'perch node start acknowledgement')
                self.record({'kind': 'start_ack', 'sim_t': self.sim_t})
                deadline = time.monotonic() + self.args.timeout_s

                def finished():
                    now = time.monotonic()
                    if now - min(self.diag_wall, self.hold_wall, self.pose_wall, self.sensor_wall) > 1.0:
                        raise RuntimeError('lost diag, Gazebo pose, hold_truth or hold_sensors during trial')
                    return self.score.outcome(self.sim_t) or ('timeout' if now >= deadline else None)

                outcome = self.wait(finished, self.args.timeout_s + 1.0, 'trial termination')
                self.record({'kind': 'end', 'sim_t': self.sim_t, 'outcome': outcome})
                return {'trial': index, **config, **self.score.result(outcome), 'recovered': False}
        finally:
            self.log = self.score = self.trial = None

    def recover(self, trial):
        """Return to standby; records rearm_ok (armed again in DEPART) and recovered in the trial dict."""
        self.recovering = {'rearm': False}
        try:
            self.cmd.publish(self.String(data='reset'))
            self.wait_for_hover(self.args.recovery_s)
            self.gz.cart_speed(0.0)
            trial['recovered'] = True
        finally:
            trial['rearm_ok'] = self.recovering['rearm'] if trial['motors_off'] else None
            trial['cycle_complete'] = bool(trial['success'] and trial['recovered'] and trial['rearm_ok'])
            self.recovering = None


def sha256(path):
    import hashlib
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def provenance(args):
    """Review D08: binaries, policy, generated world and the effective configuration of this batch."""
    import os
    env = {k: os.environ.get(k) for k in ('PERCH_NODE_ARGS', 'PERCH_PAD_FOOTPRINT', 'PERCH_YAW_SCALE',
                                          'PERCH_FREEZE_PLATE_V', 'PERCH_DEPART_PUSHOFF', 'PERCH_AIRMODE',
                                          'PERCH_EFFECTIVE_NODE_ARGS', 'PERCH_LOAD_CELL', 'PERCH_SLIP_SENSOR',
                                          'PERCH_Q2_LOCK')}
    so = PROJECT / 'gz' / 'plugins' / 'hold_pad' / 'build' / 'libHoldPad.so'
    world = Path('/home/sup/data/perchpilot/sitl/perch_world.sdf')
    px4_dir = Path(os.environ.get('PX4_DIR', '/home/sup/data/perchpilot/PX4-perch'))
    fp = re.findall(r'<footprint_half_(x|y)>([^<]*)<', world.read_text()) if world.exists() else []
    try:
        px4_git = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=px4_dir, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        px4_git = None
    node_src = (PROJECT / 'ros' / 'perch_node.py').read_text()
    qos = re.search(r'^HOLD_SENSORS_QOS = "([^"]*)"', node_src, re.M)
    depth = re.search(r'^HOLD_SENSORS_QOS_DEPTH = (\d+)', node_src, re.M)
    lock = Path(env['PERCH_Q2_LOCK']) if env.get('PERCH_Q2_LOCK') else None
    return {'holdpad_so_sha256': sha256(so),
            'q2_lock_file_sha256': sha256(lock) if lock else None,
            'q2_lock_sha256_field': json.loads(lock.read_text()).get('sha256') if lock and lock.exists() else None,
            'perch_node_hold_sensors_qos': f'{qos.group(1)}/{depth.group(1)}' if qos and depth
            else 'best_effort/keep_last/1',
            'holdpad_so_has_footprint': so.exists() and b'footprint_half_x' in so.read_bytes(),
            'px4_bin_sha256': sha256(px4_dir / 'build' / 'px4_sitl_default' / 'bin' / 'px4'), 'px4_git': px4_git,
            'policy_ts_sha256': sha256(Path(args.policy_run) / 'policy.ts'),
            'world_sdf_sha256': sha256(world), 'world_footprint': dict(fp), 'env': env,
            'px4_params_readback': {k: read_px4_param(k) for k in ('MC_AIRMODE', 'MPC_THR_HOVER', 'CBRK_FLIGHTTERM')}}


def write_result(path, args, policy, trials, error):
    bins = {}
    for trial in trials:
        key = f"speed_{trial['speed_mps']}_mu_{trial['mu']}_f_hold_n_{trial['f_hold_n']}"
        bins.setdefault(key, []).append(trial)
    git = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PROJECT, text=True).strip()
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
              if key not in ('name', 'check_config', 'eval_seed')}
    result = {'name': args.name, 'phase': 'P5', 'git': git, 'seed': policy.get('seed'),
              'provenance': provenance(args),
              'config': {**config, 'tilt_deg': 65.0, 'drone_model': DRONE, 'policy_git': policy.get('git')},
              'eval': {'n': len(trials), 'eval_seed': args.eval_seed, 'nominal': summarize(trials), 'hard': None,
                       'bins': {key: summarize(cell) for key, cell in bins.items()}},
              'train': {'iterations': 0, 'env_steps': 0, 'wall_min': 0, 'best_iter': 0},
              'status': 'incomplete' if error else 'complete', 'error': error}
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def main(argv=None):
    args = parse_args(argv)
    args.policy_run = args.policy_run.resolve()
    policy = json.loads((args.policy_run / 'policy.json').read_text())
    if not (args.policy_run / 'policy.ts').is_file():
        raise FileNotFoundError(args.policy_run / 'policy.ts')
    log_dir = TRUTH_ROOT / args.name
    result = PROJECT / 'results' / f'sitl_{args.name}.json'
    done = []
    if args.cont and (log_dir / 'trials.jsonl').exists():
        done = [json.loads(line) for line in (log_dir / 'trials.jsonl').open()]
    elif result.exists() or log_dir.exists():
        raise FileExistsError(f'run {args.name} already has logs or results')
    if args.check_config:
        print(args.policy_run)
        print(log_dir)
        return 0

    import os
    os.environ.setdefault('GZ_PARTITION', 'perchpilot')
    os.environ.setdefault('ROS_DOMAIN_ID', '77')
    import rclpy
    from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
    from rclpy.node import Node

    log_dir.mkdir(parents=True, exist_ok=args.cont)
    rclpy.init(args=[])
    node, camera = Node('perchpilot_eval_node'), Node('tag_camera_sim')
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    executor.add_node(camera)
    trials, error = list(done), None
    restart = False
    try:
        evaluator = EvalNode(node, executor, args, log_dir)
        evaluator.dashboard_result(trials)
        measure = simulated_tag_measurements(evaluator.take_camera_poses, DRONE, CART, args.eval_seed)
        start_tag_camera(camera, measure)
        evaluator.startup()
        for index, config in enumerate(trial_grid(args), 1):
            if index <= len(done):
                continue
            try:
                trial = evaluator.run_trial(index, config)
            except RuntimeError as exc:
                if 'start acknowledgement' not in str(exc):
                    raise
                restart = True          # node never left standby: rerun this trial after a SITL restart
                raise RuntimeError(f'trial {index} not started: {exc}') from exc
            trials.append(trial)
            evaluator.dashboard_result(trials)
            node.get_logger().info(f"Trial {index}/{args.trials} {config} -> {trial['outcome']}")
            try:
                evaluator.recover(trial)
            except RuntimeError as exc:
                # counted as not recovered; run_sitl_eval.sh restarts PX4 + Gazebo and continues (exit code 3)
                trial['sitl_restart_after'] = index < args.trials
                restart = index < args.trials
                raise RuntimeError(f'trial {index} recovery failed: {exc}') from exc
            finally:
                evaluator.dashboard_result(trials)
    except (RuntimeError, OSError, ValueError, subprocess.TimeoutExpired, KeyboardInterrupt,
            ExternalShutdownException) as exc:
        error = str(exc) or type(exc).__name__
        node.get_logger().error(error)
    finally:
        executor.shutdown()
        camera.destroy_node()
        node.destroy_node()
        rclpy.try_shutdown()
        with (log_dir / 'trials.jsonl').open('w') as summary:
            for trial in trials:
                summary.write(json.dumps(trial, allow_nan=False) + '\n')
        write_result(result, args, policy, trials, None if restart else error)
    print(result)
    return 3 if restart else int(error is not None)


if __name__ == '__main__':
    raise SystemExit(main())
