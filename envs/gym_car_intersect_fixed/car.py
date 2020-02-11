from typing import Dict, Union

import numpy as np
import math
import Box2D
from Box2D.b2 import fixtureDef, polygonShape, revoluteJointDef

from envs.gym_car_intersect_fixed.utils import DataSupporter
from shapely.geometry import Point, Polygon


SIZE = 80 / 1378.0
MC = SIZE / 0.02
ENGINE_POWER = 100000000 * SIZE * SIZE / MC / MC
WHEEL_MOMENT_OF_INERTIA = 4000 * SIZE * SIZE / MC / MC
FRICTION_LIMIT = 1000000 * SIZE * SIZE / MC / MC / 2
WHEEL_R = 27 / MC
WHEEL_W = 14 / MC
CENTROID = 220
WHEELPOS = [
    (-45, +60 - CENTROID), (+45, +60 - CENTROID),
    (-45, -70 - CENTROID), (+45, -70 - CENTROID)
]
HULL_POLY4 = [
    (-45, -105 - CENTROID), (+45, -105 - CENTROID),
    (-45, +105 - CENTROID), (+45, +105 - CENTROID)
]
SENSOR_SHAPE = [
    (-45, -105 - CENTROID), (+45, -105 - CENTROID),
    (-45, +105 - CENTROID), (+45, +105 - CENTROID)
]
## Point sensor:
# SENSOR_BOT = [
#     (-10,350-CENTROID), (+10,350-CENTROID),
#     (-10,+360-CENTROID),  (+10,+360-CENTROID)
# ]
SENSOR_BOT = [
    (-50, +110 - CENTROID), (+50, +110 - CENTROID),
    (-10, +300 - CENTROID), (+10, +300 - CENTROID)
]
# SENSOR_ADD = [
#     (-1,+110-CENTROID), (+1,+110-CENTROID),
#     (-50,+200-CENTROID),  (+50,+200-CENTROID)
# ]
WHEEL_COLOR = (0.0, 0.0, 0.0)
WHEEL_WHITE = (0.3, 0.3, 0.3)

class DummyCar:
    """
    Class of car for Carracing fixed environment.
    Unfortunately there are a lot of legacy magic constants,
        but if you remove or change them, everything will simply don't work correctly.
    """

    def __init__(
            self,
            world: Box2D,
            bot: bool = False,
            car_image=None,
            track=None,
            data_loader=None,
    ):
        """ Constructor to define Car.
        Parameters
        ----------
        world : Box2D World

        """
        global SIZE
        self.car_image = car_image
        self.track = track
        self.data_loader = data_loader
        self.is_bot = bot
        self._bot_state = {
            'was_break': False,
            'stop_for_next': -1,
        }

        # all coordinates in XY format, not in IMAGE coordinates
        init_x, init_y = DataSupporter.get_track_initial_position(self.track['line'])
        init_angle = DataSupporter.get_track_angle(track) - np.pi / 2
        width_y, height_x = self.car_image.size

        CAR_HULL_POLY4 = [
            (-height_x / 2, -width_y / 2), (+height_x / 2, -width_y / 2),
            (-height_x / 2, +width_y / 2), (+height_x / 2, +width_y / 2)
        ]
        N_SENSOR_BOT = [
            (-height_x / 2 * 1.11, +width_y / 2 * 1.0), (+height_x / 2 * 1.11, +width_y / 2 * 1.0),
            # (-height_x/2*1.11, +width_y/2*1.11), (+height_x/2*1.11, +width_y/2*1.11),
            (-height_x / 2 * 0.8, +width_y / 2 * 3), (+height_x / 2 * 0.8, +width_y / 2 * 3)
            # (-height_x/2*0.22, +width_y/2*2), (+height_x/2*0.22, +width_y/2*2)
        ]
        WHEELPOS = [
            (-height_x / 2, +width_y / 2 / 2), (+height_x / 2, +width_y / 2 / 2),
            (-height_x / 2, -width_y / 2 / 2), (+height_x / 2, -width_y / 2 / 2)
        ]

        LEFT_SENSOR = [
            (0, 0), (+height_x, 0),
            (0, +width_y * 1.5), (+height_x, +width_y * 1.5)
        ]

        RIGHT_SENSOR = [
            (-height_x, 0), (0, 0),
            (-height_x, +width_y * 1.5), (0, +width_y * 1.5)
        ]

        N_SENSOR_SHAPE = CAR_HULL_POLY4

        SENSOR = N_SENSOR_BOT if bot else N_SENSOR_SHAPE
        self.world = world
        # SIZE *= 0.5
        self._hull = self.world.CreateDynamicBody(
            position=(init_x, init_y),
            angle=init_angle,
            fixtures=[
                fixtureDef(shape=polygonShape(
                    vertices=[(x * SIZE, y * SIZE) for x, y in CAR_HULL_POLY4]),
                    density=1.0,
                    userData='body',
                ),
                fixtureDef(shape=polygonShape(
                    vertices=[(x * SIZE, y * SIZE) for x, y in SENSOR]),
                    isSensor=True,
                    userData='sensor',
                ),
                fixtureDef(shape=polygonShape(
                    vertices=[(x * SIZE, y * SIZE) for x, y in RIGHT_SENSOR]),
                    isSensor=True,
                    userData='right_sensor',
                ),
                fixtureDef(shape=polygonShape(
                    vertices=[(x * SIZE, y * SIZE) for x, y in LEFT_SENSOR]),
                    isSensor=True,
                    userData='left_sensor',
                ),
            ]
        )
        # SIZE *= 2

        self._hull.name = 'bot_car' if bot else 'car'
        self._hull.cross_time = float('inf')
        self._hull.stop = False
        self._hull.left_sensor = False
        self._hull.right_sensor = False
        self._hull.collision = False
        self._hull.userData = self._hull
        self.wheels = []
        self.fuel_spent = 0.0
        WHEEL_POLY = [
            (-WHEEL_W, +WHEEL_R), (+WHEEL_W, +WHEEL_R),
            (+WHEEL_W, -WHEEL_R), (-WHEEL_W, -WHEEL_R)
        ]

        for wx, wy in WHEELPOS:
            front_k = 1.0 if wy > 0 else 1.0
            w = self.world.CreateDynamicBody(
                position=(init_x + wx * SIZE, init_y + wy * SIZE),
                angle=init_angle,
                fixtures=fixtureDef(
                    shape=polygonShape(vertices=[(x * front_k * SIZE, y * front_k * SIZE) for x, y in WHEEL_POLY]),
                    density=0.1,
                    categoryBits=0x0020,
                    maskBits=0x001,
                    restitution=0.0)
            )
            w.wheel_rad = front_k * WHEEL_R * SIZE
            w.is_front = front_k
            w.gas = 0.0
            w.brake = 0.0
            w.steer = 0.0
            w.phase = 0.0  # wheel angle
            w.omega = 0.0  # angular velocity
            rjd = revoluteJointDef(
                bodyA=self._hull,
                bodyB=w,
                localAnchorA=(wx * SIZE, wy * SIZE),
                localAnchorB=(0, 0),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=180 * 900 * SIZE * SIZE,
                motorSpeed=0,
                lowerAngle=-0.4,
                upperAngle=+0.4,
            )
            w.joint = self.world.CreateJoint(rjd)
            w.tiles = set()
            w.name = 'wheel'
            w.collision = False
            w.penalty = False
            w.userData = w
            self.wheels.append(w)
        self.drawlist = self.wheels + [self._hull]
        self.target = (0, 0)

        self._time: int = 0
        self.userData = self
        self._track_point: int = 0
        self._old_track_point: int = 0
        self._state_data = None
        self._last_action = [0, 0, 0]
        self._flush_stats()

    def get_vector_state(self) -> np.ndarray:
        state = []
        self.update_stats()
        CAR_FEATURES = {
            'hull_position', 'hull_angle', 'car_speed', 'wheels_positions', 'track_sensor',
            'road_sensor', 'finish_sensor',
        }
        if len(set(self.data_loader.car_features_list) - CAR_FEATURES) > 0:
            raise ValueError(
                f"incorrect car features list\n"
                f"you pass : {set(self.data_loader.car_features_list)}\n"
                f"we expect some of {CAR_FEATURES}"
            )

        if 'hull_position' in self.data_loader.car_features_list:
            state.extend([
                self._hull.position.x / self.data_loader.playfield_size[0],
                self._hull.position.y / self.data_loader.playfield_size[1],
            ])

        if 'hull_angle' in self.data_loader.car_features_list:
            state.extend([self._hull.angle, np.sin(self._hull.angle), np.cos(self._hull.angle)])

        if 'car_speed' in self.data_loader.car_features_list:
            state.extend([
                self._hull.linearVelocity.x / 1000,
                self._hull.linearVelocity.y / 1000,
            ])

        if 'wheels_positions' in self.data_loader.car_features_list:
            for wheel in self.wheels:
                state.extend([
                    wheel.position.x / self.data_loader.playfield_size[0],
                    wheel.position.y / self.data_loader.playfield_size[1],
                ])

        if 'track_sensor' in self.data_loader.car_features_list:
            state.append(1.0 if self._state_data['is_out_of_track'] else 0.0)

        if 'road_sensor' in self.data_loader.car_features_list:
            state.append(
                1.0
                if self._state_data['is_out_of_road'] or self._state_data['is_out_of_map']
                else 0.0
            )

        if 'finish_sensor' in self.data_loader.car_features_list:
            state.append(1.0 if self._state_data['is_finish'] else 0.0)

        return np.array(state)

    @property
    def angle_index(self) -> int:
        return int((int(self._hull.angle * 180 / np.pi) % 360) / 8)

    @property
    def angle_degree(self) -> float:
        return self._hull.angle * 180 / np.pi

    @property
    def position_PLAY(self) -> np.ndarray:
        return np.array([self._hull.position.x, self._hull.position.y])

    @property
    def position_IMG(self) -> np.ndarray:
        return self.data_loader.convertPLAY2IMG(self.position_PLAY)

    @property
    def stats(self) -> Dict[str, Union[bool, int, float]]:
        return self._state_data

    @property
    def wheels_positions_PLAY(self) -> np.array:
        return np.array([
            np.array([wheel.position.x, wheel.position.y])
            for wheel in self.wheels
        ])

    @property
    def wheels_position_IMG(self) -> np.array:
        return self.data_loader.convertPLAY2IMG(self.wheels_positions_PLAY)

    def _is_car_closely_to(self, point, threshold=0.5) -> bool:
        if point.shape != (2, ):
            raise ValueError
        return (
            np.any(np.sqrt(((self.wheels_positions_PLAY - point)**2).sum(axis=1)) < threshold)
            or
            ((self.position_PLAY - point)**2).sum() < threshold
        )

    def _flush_stats(self):
        """
        Set car statistic data to initial state.
        """
        self._state_data = {
            'new_tiles_count': 0,
            'is_finish': False,
            'is_collided': False,
            'is_on_cross_road': False,
            'is_out_of_track': False,
            'is_out_of_map': False,
            'is_out_of_road': False,

            'left_sensor': False,
            'right_sensor': False,

            'speed': 0.0,
            'time': 0,
            'track_progress': 0.0,
            'last_action': [0.0, 0.0, 0.0]
        }

    def update_stats(self):
        """
        Update car statistic with current car state.
        """
        self._flush_stats()
        cur_points = [
            np.array([wheel.position.x, wheel.position.y])
            for wheel in self.wheels
        ]
        # self._state_data['car_position'] = np.mean(cur_points, axis=0)

        for wheel_position in cur_points:
            if np.any(wheel_position < 0):
                self._state_data['is_out_of_map'] = True
            if wheel_position[0] > self.data_loader.playfield_size[0]:
                self._state_data['is_out_of_map'] = True
            if wheel_position[1] > self.data_loader.playfield_size[1]:
                self._state_data['is_out_of_map'] = True

        cur_points = [Point(x) for x in cur_points]

        for wheel_position in cur_points:
            if not self._state_data['is_out_of_road']:
                for polygon in self.world.restricted_world['not_road']:
                    if polygon.contains(wheel_position):
                        self._state_data['is_out_of_road'] = True
                        break

            if not self._state_data['is_on_cross_road']:
                for polygon in self.world.restricted_world['cross_road']:
                    if polygon.contains(wheel_position):
                        self._state_data['is_on_cross_road'] = True
                        break

            if not self._state_data['is_out_of_track']:
                if not self.track['polygon'].contains(wheel_position):
                    self._state_data['is_out_of_track'] = True

        # update track progress
        self._update_track_point()
        self.update_finish()
        self._state_data['track_progress'] = self._track_point / (
                len(self.track['line']) - 1 if len(self.track['line']) >= 2 else 1
        )
        self._state_data['last_action'] = self._last_action

        # update collision from contact listener
        self._state_data['is_collided'] = self._hull.collision
        self._state_data['right_sensor'] = self._hull.right_sensor
        self._state_data['left_sensor'] = self._hull.left_sensor

        # add extra info to data:
        self._state_data['speed'] = np.sqrt(np.sum(
            np.array([
                self._hull.linearVelocity.x,
                self._hull.linearVelocity.y,
            ])**2
        ))
        self._state_data['time'] = self._time

    def update_finish(self):
        self._state_data['is_finish'] = False
        if self._is_car_closely_to(self.track['line'][-1], 0.5):
            self._state_data['is_finish'] = True

    def _update_track_point(self, hard=False):
        """
        Move car goal point in accordance with car track progress.
        """
        for track_index in range(self._track_point, len(self.track['line']), 1):
            if self._is_car_closely_to(self.track['line'][track_index], 0.75):
                continue
            if hard:
                self._old_track_point = self._track_point
                self._track_point = track_index
            break
        self._state_data['new_tiles_count'] = self._track_point - self._old_track_point

    def gas(self, gas):
        """
        Car control: rear wheel drive
        """
        gas = np.clip(gas, 0, 1)
        self._last_action[0] = gas
        gas /= 10
        for w in self.wheels[2:4]:
            diff = gas - w.gas
            if diff > 0.01:
                diff = 0.01  # gradually increase, but stop immediately
            w.gas += diff

    def brake(self, b):
        """
        Car control: brake b=0..1, more than 0.9 blocks wheels to zero rotation
        """
        b = np.clip(b, 0, 1)
        self._last_action[2] = float(b)
        for w in self.wheels:
            w.brake = b

    def steer(self, s):
        """
        Car control: steer s=-1..1, it takes time to rotate steering wheel from side to side, s is target position
        """
        s = np.clip(s, -1, 1)
        self._last_action[1] = float(s)
        self.wheels[0].steer = s
        self.wheels[1].steer = s

    def steer_by_angle(self, s):
        self._hull.angle += s

    def go_to_target(self):
        """
        Set car params to move one step to current goal. Used for bot cars_full.
        """
        self.update_stats()

        if self._state_data['right_sensor']:
            self.brake(0.8)
            self._bot_state['stop_for_next'] = 10
            return

        if self._bot_state['stop_for_next'] > 0:
            self.brake(0.8)
            self._bot_state['stop_for_next'] -= 1
            return

        x, y = round(self._hull.position.x, 2), round(self._hull.position.y, 2)

        x_pos, y_pos = self.track['line'][self._track_point]
        target_angle = -math.atan2(x_pos - x, y_pos - y)

        x_pos_next, y_pos_next = self.track['line'][self._track_point + 1]
        target_angle_next = -math.atan2(x_pos_next - x, y_pos_next - y)

        direction = -math.pi / 2 + target_angle - self._hull.angle
        direction_next = -math.pi / 2 + target_angle_next - self._hull.angle

        steer_value = math.cos(direction * 0.6 + direction_next * 0.4)
        self.steer(steer_value)

        if abs(steer_value) >= 0.2 and not self._bot_state['was_break']:
            self.brake(0.8)
            self._bot_state['was_break'] = True
        else:
            self.brake(0.0)
            self.gas(0.1)

        if abs(steer_value) < 0.1:
            self._bot_state['was_break'] = False
            self.gas(0.3)

    def step(self, dt):
        """
        Compute forces and apply them to car wheels in accordance with gas/brake/steer state.
        This function must be called once in pyBox2D step.
        """
        self._update_track_point(hard=True)
        self._time += 1

        if self.is_bot:
            self.go_to_target()

        for wheel_index, w in enumerate(self.wheels):
            # Steer each wheel
            dir = np.sign(w.steer - w.joint.angle)
            val = abs(w.steer - w.joint.angle) * 5
            if val < 1e-3:
                val = 0
            w.joint.motorSpeed = dir * val

            # Position => friction_limit
            grass = True
            friction_limit = FRICTION_LIMIT * 0.6  # Grass friction if no tile
            for tile in w.tiles:
                friction_limit = max(friction_limit, FRICTION_LIMIT * tile.road_friction)
                grass = False

            # Force
            forw = w.GetWorldVector((0, 1))
            side = w.GetWorldVector((1, 0))
            v = w.linearVelocity

            vf = forw[0] * v[0] + forw[1] * v[1]  # forward speed
            vs = side[0] * v[0] + side[1] * v[1]  # side speed

            # WHEEL_MOMENT_OF_INERTIA*np.square(w.omega)/2 = E -- energy
            # WHEEL_MOMENT_OF_INERTIA*w.omega * domega/dt = dE/dt = W -- power
            # domega = dt*W/WHEEL_MOMENT_OF_INERTIA/w.omega
            w.omega += dt * ENGINE_POWER * w.gas / WHEEL_MOMENT_OF_INERTIA / (
                        abs(w.omega) + 5.0)  # small coef not to divide by zero
            self.fuel_spent += dt * ENGINE_POWER * w.gas

            if w.brake >= 0.9:
                w.omega = 0
            elif w.brake > 0:
                BRAKE_FORCE = 15  # radians per second
                dir = -np.sign(w.omega)
                val = BRAKE_FORCE * w.brake
                if abs(val) > abs(w.omega):
                    val = abs(w.omega)  # low speed => same as = 0
                w.omega += dir * val
            w.phase += w.omega * dt

            vr = w.omega * w.wheel_rad  # rotating wheel speed
            f_force = -vf + vr  # force direction is direction of speed difference
            p_force = -vs

            # Physically correct is to always apply friction_limit until speed is equal.
            # But dt is finite, that will lead to oscillations if difference is already near zero.
            f_force *= 205000 * SIZE * SIZE  # Random coefficient to cut oscillations in few steps (have no effect on friction_limit)
            p_force *= 205000 * SIZE * SIZE
            force = np.sqrt(np.square(f_force) + np.square(p_force))

            if abs(force) > friction_limit:
                f_force /= force
                p_force /= force
                force = friction_limit  # Correct physics here
                f_force *= force
                p_force *= force

            w.omega -= dt * f_force * w.wheel_rad / WHEEL_MOMENT_OF_INERTIA

            w.ApplyForceToCenter((
                p_force * side[0] + f_force * forw[0],
                p_force * side[1] + f_force * forw[1]), True)


    def destroy(self):
        """
        Remove car property from pyBox2D world.
        """
        self.world.DestroyBody(self._hull)
        del self._hull
        for w in self.wheels:
            self.world.DestroyBody(w)
        del self.wheels
