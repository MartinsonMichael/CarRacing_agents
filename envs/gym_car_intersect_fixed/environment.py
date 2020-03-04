import json
from functools import lru_cache

import Box2D
import cv2
import gym
import numpy as np
import torch

from gym import spaces
from gym.utils import seeding, EzPickle

from envs.gym_car_intersect_fixed.car import DummyCar
from envs.gym_car_intersect_fixed.contact_listner import RefactoredContactListener
from envs.gym_car_intersect_fixed.rewards import Rewarder

from envs.gym_car_intersect_fixed.utils import DataSupporter
from shapely import geometry
from typing import List, Union, Dict

FPS = 60


class CarRacingHackatonContinuousFixed(gym.Env, EzPickle):
    metadata = {
        'render.modes': ['human', 'rgb_array', 'state_pixels'],
        'video.frames_per_second': FPS
    }

    def __init__(self, settings_file_path):
        EzPickle.__init__(self)
        self._settings = json.load(open(settings_file_path))

        # load env resources
        self._data_loader = DataSupporter(self._settings)

        # init world
        self.seed()
        self.contactListener_keepref = RefactoredContactListener(self)
        self.world = Box2D.b2World((0, 0), contactListener=self.contactListener_keepref)
        self._was_done: bool = False
        self._init_world()
        self._static_env_state_cache = None

        # init agent data
        self.car = None
        self._preseted_agent_track = None
        self.create_agent_car()
        self.rewarder = Rewarder(self._settings)

        # init bots data
        self.num_bots = self._settings['bot_number']
        self.bot_cars = []

        self._preseted_render_mode = 'human'

        # init gym properties
        self.picture_state = np.zeros_like(self._data_loader.get_background(), dtype=np.uint8)
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0]),
            high=np.array([+1.0, +1.0, +1.0]),
            dtype=np.float32
        )  # steer, gas, brake
        test_car = DummyCar(
            world=self.world,
            car_image=self._data_loader.peek_car_image(is_for_agent=True),
            track=DataSupporter.do_with_points(
                self._data_loader.peek_track(is_for_agent=True, expand_points=200, index=self._preseted_agent_track),
                self._data_loader.convertIMG2PLAY,
            ),
            data_loader=self._data_loader,
            bot=False,
        )
        self.observation_space = spaces.Dict(
            picture=spaces.Box(
                low=0,
                high=255,
                shape=self._data_loader.get_background().shape,
                dtype=np.uint8,
            ),
            car_vector=spaces.Box(
                low=-5,
                high=+5,
                shape=(len(test_car.get_vector_state()), ),
                dtype=np.float32,
            ),
            env_vector=spaces.Box(
                low=-5,
                high=+5,
                shape=(len(self._create_vector_env_static_description()),),
                dtype=np.float32,
            ),
        )
        self._need_draw_picture = self._settings['state_config']['picture']
        self.reset()

    def set_render_mode(self, mode):
        self._preseted_render_mode = mode

    def set_bot_number(self, bot_number):
        self.num_bots = bot_number

    def set_agent_track(self, index):
        """
        Set agent track.
        :param index: index from 0 to number of tracks (smt like 12)
        :return: void
        """
        if index is None:
            print('agent track set to random')
            self._preseted_agent_track = None
            return
        if index < 0 or index > self._data_loader.track_count:
            raise ValueError(f'index must be from 0 to {self._data_loader.track_count}')
        print(f'agent track set to {index}')
        self._preseted_agent_track = index

    def _init_world(self):
        """
        function to create shapely polygons, which define road zones, not road zone
        :return: void
        """
        self.world.restricted_world = {
            'not_road': [],
            'cross_road': [],
        }
        for polygon in self._data_loader.data.get_polygons(0):
            polygon_name = polygon['label']
            polygon_points = polygon['points']
            if polygon_name in {'not_road', 'cross_road'}:
                self.world.restricted_world[polygon_name].append(geometry.Polygon(
                    self._data_loader.convertIMG2PLAY(polygon_points)
                ))

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _destroy(self):
        """
        destroy b2world
        :return: void
        """
        if self.car is not None:
            self.car.destroy()
            del self.car
        if self.bot_cars is not None:
            for bot_car in self.bot_cars:
                bot_car.destroy()
                del bot_car

    def reset(self, force=False):
        """
        recreate agent car and bots cars_full
        :return: initial state
        """
        self._destroy()
        self.time = 0
        self.create_agent_car()
        self.rewarder = Rewarder(self._settings)
        self._need_draw_picture = self._settings['state_config']['picture']

        self.bot_cars = []
        for bot_index in range(self.num_bots):
            self.create_bot_car()

        if force:
            return self.step(None)[0], 0, False, {'was_reset': True}

        return self.step(None)[0]

    def create_agent_car(self):
        self.car = DummyCar(
            world=self.world,
            car_image=self._data_loader.peek_car_image(is_for_agent=True),
            track=DataSupporter.do_with_points(
                self._data_loader.peek_track(
                    is_for_agent=True,
                    expand_points=self._settings['reward']['track_checkpoint_expanding'],
                    index=self._preseted_agent_track
                ),
                self._data_loader.convertIMG2PLAY,
            ),
            data_loader=self._data_loader,
            bot=False,
        )
        self.car.update_stats()
        self.car.update_stats()

    def create_bot_car(self):
        track = DataSupporter.do_with_points(
            self._data_loader.peek_track(is_for_agent=False, expand_points=50),
            self._data_loader.convertIMG2PLAY,
        )
        collided_indexes = self.initial_track_check(track)
        if len(collided_indexes) == 0:
            bot_car = DummyCar(
                world=self.world,
                car_image=self._data_loader.peek_car_image(is_for_agent=False),
                track=track,
                data_loader=self._data_loader,
                bot=True,
            )
            self.bot_cars.append(bot_car)

    def initial_track_check(self, track) -> List[int]:
        """
        Check if initial track position intersect some existing car. Return list of bots car indexes, which
        collide with track initial position. For agent car return -1 as index.
        :return: list of integers
        """
        init_pos = DataSupporter.get_track_initial_position(track)
        collided_indexes = []
        for bot_index, bot_car in enumerate(self.bot_cars):
            if DataSupporter.dist(init_pos, bot_car.position_PLAY) < 5:
                collided_indexes.append(bot_index)

        if self.car is not None:
            if DataSupporter.dist(self.car.position_PLAY, init_pos) < 5:
                collided_indexes.append(-1)

        return collided_indexes

    def step(self, action: List[float]):
        if self._was_done:
            self._was_done = False
            return self.reset(), 0.0, False, {'was_reset': True}

        info = {}
        if action is not None:
            if self._settings['steer_policy']['angle_steer']:
                self.car.steer_by_angle(
                    action[0] * np.pi / 180 * self._settings['steer_policy']['angle_steer_multiplication']
                )
                self.car.gas(action[1])
                self.car.brake(action[2])
            else:
                self.car.steer(action[0])
                self.car.gas(action[1])
                self.car.brake(action[2])

        delta_time = 1.0 / FPS
        self.car.step(delta_time)
        for bot_car in self.bot_cars:
            bot_car.step(delta_time)

        self.world.Step(delta_time, 6 * 30, 2 * 30)
        self.time += delta_time
        self.car.update_stats()

        for index, bot_car in enumerate(self.bot_cars):
            bot_car.update_stats()

            # print(f'BOT {index}')
            # print(bot_car.stats)
            # print()

            if bot_car.stats['is_finish'] or bot_car.stats['is_out_of_road'] or bot_car.stats['is_out_of_map']:
                bot_car.destroy()
                del bot_car
                self.bot_cars.pop(index)

        if len(self.bot_cars) < self.num_bots:
            self.create_bot_car()

        if self._settings['state_config']['picture']:
            try:
                self.picture_state = self.render(self._preseted_render_mode)
            except:
                return self.reset(force=True)
        else:
            self.picture_state = None

        done = self.rewarder.get_step_done(self.car.stats)
        step_reward = self.rewarder.get_step_reward(self.car.stats)
        info.update(self.car.stats)

        self._was_done = done
        return self._create_state(), step_reward, done, info

    def _create_state(self) -> Union[np.ndarray, Dict[str, Union[None, np.ndarray]]]:
        return {
            'picture':
                self.picture_state.astype(np.uint8)
                if self._settings['state_config']['picture']
                else None,
            'car_vector':
                self.car.get_vector_state().astype(np.float32)
                if len(self._settings['state_config']['vector_car_features']) != 0
                else None,
            'env_vector': self._create_vector_env_static_description().astype(np.float32),
        }

    def _get_car_state_vector(self) -> np.ndarray:
        if len(self._data_loader.car_features_list - {'car_radar_1', 'car_radar_2', 'car_radar_3'}):
            return self.car.get_vector_state().astype(np.float32)

        dists = []
        for bot in self.bot_cars:
            dists.append(None)

    @lru_cache(maxsize=None)
    def _create_vector_env_static_description(self) -> np.ndarray:
        params_to_use = self._settings['state_config']['vector_env_features']
        # that a have, just to remaind:
        # * agent car - NOT in this function
        # * bots car [optionaly] - NOT in this function
        # agent track info: check points, main goal
        # road info: for current track we have its bounds, but them in a form of polygon coordinates
        # not road info, coordinates of polygons defined not road surface
        POSSIBLE_PARAMS = {
            'track_goal', 'track_line', 'track_polygon', 'not_road', 'cross_road'
        }
        if len(set(params_to_use) - POSSIBLE_PARAMS) > 0:
            raise ValueError(
                f'some params of vector env description are incorrect,\n \
                should be some of: {POSSIBLE_PARAMS}, \n \
                you pass : {params_to_use}'
            )

        env_vector = []

        if 'track_goal' in params_to_use:
            env_vector.extend(self.car.track['line'][-1] / self._data_loader.playfield_size)

        if 'track_line' in params_to_use:
            for point in self.car.track['line']:
                env_vector.extend(point / self._data_loader.playfield_size)

        if 'track_polygon' in params_to_use:
            for point in zip(*list(map(list, self.car.track['polygon'].exterior.coords.xy))):
                env_vector.extend(np.array(point) / self._data_loader.playfield_size)

        for polygon in self._data_loader.data.get_polygons(0):
            polygon_name = polygon['label']
            polygon_points = polygon['points']
            if polygon_name in params_to_use:
                for point in polygon_points:
                    env_vector.extend(point / self._data_loader.playfield_size)
        return np.array(env_vector, dtype=np.float32)

    def render(self, mode='human', full_image=False) -> np.array:
        background_image = self._data_loader.get_background(true_size=full_image)
        background_mask = np.zeros(
            shape=(background_image.shape[0], background_image.shape[1]),
            dtype='uint8'
        )

        self.draw_car(
            background_image,
            background_mask,
            self.car,
            full_image=full_image,
        )
        # self.debug_draw_hull(background_image, self.car)
        for bot_car in self.bot_cars:
            self.draw_car(
                background_image,
                background_mask,
                bot_car,
                full_image=full_image,
            )
            if mode == 'debug':
                self.debug_draw_track(
                    background_image=background_image,
                    car=bot_car,
                    point_size=10,
                    color='green',
                )

        # if mode == 'debug':
        # self.debug_draw_track(
        #     background_image,
        #     car=self.car,
        #     point_size=3,
        #     color='red'
        # )
            # self.debug_draw_restrictions(background_image)
        return background_image

    def draw_car(self, background_image, background_mask, car: DummyCar, full_image=False):
        # check dimensions
        if background_image.shape[0] != background_mask.shape[0]:
            raise ValueError('background image and mask have different shape')
        if background_image.shape[1] != background_mask.shape[1]:
            raise ValueError('background image and mask have different shape')
        if car.car_image.mask.shape[0] != car.car_image.image.shape[0]:
            raise ValueError('car image and mask have different shape')
        if car.car_image.mask.shape[1] != car.car_image.image.shape[1]:
            raise ValueError('car image and mask have different shape')

        # rotate car image and mask of car image, and compute bounds of rotated image
        masked_image, car_mask_image = self._data_loader.get_rotated_car_image(car, true_size=full_image)
        bound_y, bound_x = masked_image.shape[:2]

        # car position in image coordinates (in pixels)
        car_x, car_y = car.position_IMG * (
            self._data_loader._background_image_scale
            if not full_image else self._data_loader.FULL_RENDER_COEFF
        )

        # bounds of car image on background image, MIN/MAX in a case of position near the background image boarder
        start_x = min(
            max(
                int(car_x - bound_x / 2),
                0,
            ),
            background_image.shape[1],
        )
        start_y = min(
            max(
                int(car_y - bound_y / 2),
                0,
            ),
            background_image.shape[0],
        )
        end_x = max(
            min(
                int(car_x + bound_x / 2),
                background_image.shape[1]
            ),
            0,
        )
        end_y = max(
            min(
                int(car_y + bound_y / 2),
                background_image.shape[0],
            ),
            0,
        )

        # in a case there car image is out of background, just return backgraund image
        if start_x == end_x or start_y == end_y:
            return background_image, background_mask

        # compute bounds of car image, in case then car near the bord of backgraund image,
        #    and so displayed car image
        #    less then real car image
        mask_start_x = start_x - int(car_x - bound_x / 2)
        mask_start_y = start_y - int(car_y - bound_y / 2)
        mask_end_x = mask_start_x + end_x - start_x
        mask_end_y = mask_start_y + end_y - start_y

        # finally crop car mask and car image, and insert them to background
        cropped_mask = (car_mask_image[
                        mask_start_y: mask_end_y,
                        mask_start_x: mask_end_x,
                        :,
                        ] > 240)

        cropped_image = (
            masked_image[
                mask_start_y: mask_end_y,
                mask_start_x: mask_end_x,
                :,
            ]
        )
        # back = background_image[start_y:end_y, start_x:end_x, :]
        background_image[start_y:end_y, start_x:end_x, :] = (
                background_image[start_y:end_y, start_x:end_x, :] * (1 - cropped_mask) +
                cropped_image * cropped_mask
        )
        # background_image[start_y:end_y, start_x:end_x, :][cropped_mask] = cropped_image[cropped_mask]

    def close(self):
        del self.car
        del self.bot_cars
        del self._data_loader
        del self.world

    def debug_draw_hull(self, background_image, car, point_size=10):
        color = {
            'body': 'red',
            'sensor': 'green',
            'right_sensor': 'blue',
            'left_sensor': [255, 255, 0],
        }
        for fixture in car._hull.fixtures:
            for point in fixture.shape.vertices:
                pnt = DataSupporter.convert_XY2YX(self._data_loader.convertPLAY2IMG(point) + car.position_IMG)
                # print(f'fhull point: {pnt}')
                CarRacingHackatonContinuousFixed.debug_draw_sized_point(
                    background_image,
                    pnt,
                    point_size,
                    color[fixture.userData],
                )

    def debug_draw_track(self, background_image, car, point_size=10, color='blue'):
        for point in DataSupporter.convert_XY2YX(self._data_loader.convertPLAY2IMG(car.track['line'])) * self._data_loader._background_image_scale:
            CarRacingHackatonContinuousFixed.debug_draw_sized_point(
                background_image,
                point,
                point_size,
                color,
            )
        CarRacingHackatonContinuousFixed.debug_draw_sized_point(
            background_image,
            DataSupporter.convert_XY2YX(self._data_loader.convertPLAY2IMG(car.track['line'][car._track_point])) * self._data_loader._background_image_scale,
            point_size,
            'blue',
        )

    def debug_draw_polygon(self,
                           background_image,
                           polygon: np.array,
                           point_size: int = 10,
                           color='red',
                           ):
        x, y = polygon.exterior.coords.xy
        for point in zip(x, y):
            CarRacingHackatonContinuousFixed.debug_draw_sized_point(
                background_image,
                DataSupporter.convert_XY2YX(self._data_loader.convertPLAY2IMG(np.array(point))),
                point_size,
                color,
            )

    @staticmethod
    def debug_draw_sized_point(
            background_image,
            coordinate: np.array,
            size: int,
            color: Union[np.array, str]
    ):
        if isinstance(color, str):
            color = {
                'red': np.array([255, 0, 0]),
                'green': np.array([0, 255, 0]),
                'blue': np.array([0, 0, 255]),
                'black': np.array([0, 0, 0]),
                'while': np.array([255, 255, 255]),
            }[color]
        y, x = coordinate
        for dx in range(int(-size / 2), int(size / 2) + 1, 1):
            for dy in range(int(-size / 2), int(size / 2) + 1, 1):
                background_image[
                int(np.clip(y + dy, 0, background_image.shape[0] - 1)),
                int(np.clip(x + dx, 0, background_image.shape[1] - 1)),
                :,
                ] = color
