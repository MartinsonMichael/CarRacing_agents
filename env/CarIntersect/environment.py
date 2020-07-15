import json
from functools import lru_cache

import yaml
from shapely import geometry
from typing import List, Union, Dict, Tuple, Optional
import Box2D
import gym
import time
import numpy as np
from gym import spaces
from gym.utils import seeding, EzPickle
from PIL import Image

from env.CarIntersect.car import DummyCar
from env.CarIntersect.contact_listner import RefactoredContactListener
from env.CarIntersect.rewards import Rewarder
from env.CarIntersect.utils import DataSupporter, Geom, Img


class CarIntersect(gym.Env, EzPickle):

    def __init__(self, settings_file_path_or_settings):
        EzPickle.__init__(self)
        if isinstance(settings_file_path_or_settings, dict):
            self._settings = settings_file_path_or_settings
        else:
            # noinspection PyBroadException
            try:
                self._settings = json.load(open(settings_file_path_or_settings, 'r'))
            except:
                self._settings = yaml.load(open(settings_file_path_or_settings, 'r'))

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
        self.car: DummyCar = None
        self.bot_cars = []
        self.create_agent_car()
        self.rewarder = Rewarder(self._settings)

        # init bots data
        self.num_bots = self._settings['bot_number']

        self.bot_cars = []
        # init gym properties
        self.picture_state = np.zeros_like(
            a=(*self._data_loader.get_target_image_size(), 3),
            dtype=np.uint8,
        )
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0]),
            high=np.array([+1.0, +1.0, +1.0]),
            dtype=np.float32
        )  # steer, gas, brake
        test_car = DummyCar(
            world=self.world,
            car_image=self._data_loader.peek_car_image(is_for_agent=True),
            track=DataSupporter.do_with_points(
                self._data_loader.peek_track(is_for_agent=True, expand_points_pixels=10),
                self._data_loader.convertIMG2PLAY,
            ),
            data_loader=self._data_loader,
            bot=False,
        )
        self.observation_space = spaces.Dict(
            picture=spaces.Box(
                low=0,
                high=255,
                shape=(*self._data_loader.get_target_image_size(), 3),
                dtype=np.uint8,
            ),
            car_vector=spaces.Box(
                low=-5,
                high=+5,
                shape=(len(test_car.get_vector_state()),),
                dtype=np.float32,
            ),
            # env_vector=spaces.Box(
            #     low=-5,
            #     high=+5,
            #     shape=(len(self._create_vector_env_static_description()),),
            #     dtype=np.float32,
            # ),
        )
        self.time = 0
        # self.reset(first=True)
        # time.sleep(0.5)
        # self.seed()
        # self.reset(first=True)

        if len(set(self._data_loader.car_features_list) - DummyCar.car_features_set()) > 0:
            raise ValueError(
                f"incorrect car features list\n"
                f"you pass : {set(self._data_loader.car_features_list)}\n"
                f"we expect some of {DummyCar.car_features_set()}"
            )

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
                    self._data_loader.convertIMG2PLAY(polygon_points) * DataSupporter.image_coef()
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

    def reset(self, force=False, first=False, eval=False):
        """
        recreate agent car and bots cars_full
        :return: initial state
        """
        if eval:
            self._data_loader.start_eval()
        else:
            self._data_loader.stop_eval()

        self._was_done = False
        self._destroy()
        self.time = 0
        self.create_agent_car()
        self.rewarder = Rewarder(self._settings)

        self.bot_cars = []
        for bot_index in range(self.num_bots):
            self.create_bot_car()

        if force:
            return self.step(None)[0], 0, False, {'was_reset': True}

        if first:
            delta_time = 1.0 / 60
            for _ in range(50):
                for car in [self.car] + self.bot_cars:
                    car.brake(1)
                    car.step(delta_time, test=True)
                self.world.Step(delta_time, 6 * 30, 2 * 30)

        return self.step(None)[0]

    def create_agent_car(self):
        self.car = DummyCar(
            world=self.world,
            car_image=self._data_loader.peek_car_image(is_for_agent=True),
            track=self._data_loader.peek_track(
                is_for_agent=True,
                expand_points_pixels=self._settings['reward']['track_checkpoint_expanding'],
            ),
            data_loader=self._data_loader,
            bot=False,
        )

    def create_bot_car(self):
        track = self._data_loader.peek_track(is_for_agent=False, expand_points_pixels=50)
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
        init_pos = Geom.get_track_initial_position(track)
        collided_indexes = []
        for bot_index, bot_car in enumerate(self.bot_cars):
            if Geom.dist(init_pos, bot_car.position_PLAY) < 13:
                collided_indexes.append(bot_index)

        if self.car is not None:
            if Geom.dist(self.car.position_PLAY, init_pos) < 13:
                collided_indexes.append(-1)

        return collided_indexes

    def step(self, action: Union[None, List[float]]) \
            -> Tuple[Dict[str, Union[None, np.ndarray]], float, bool, Dict]:
        if self._was_done:
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

        delta_time = 1.0 / 60
        self.car.step(delta_time)
        for bot_car in self.bot_cars:
            bot_car.step(delta_time)

        self.world.Step(delta_time, 6 * 30, 2 * 30)
        self.time += delta_time
        self.car.after_world_step()

        for index, bot_car in enumerate(self.bot_cars):
            bot_car.after_world_step()

            if bot_car.stats['is_finish'] or bot_car.stats['is_out_of_road'] or bot_car.stats['is_out_of_map']:
                bot_car.destroy()
                del bot_car
                self.bot_cars.pop(index)

        if len(self.bot_cars) < self.num_bots:
            self.create_bot_car()

        done = self.rewarder.get_step_done(self.car.stats)
        step_reward = self.rewarder.get_step_reward(self.car.stats)

        info.update(self.car.stats)
        info.update(self.car.DEBUG_create_radar_state(2, self.bot_cars))

        self._was_done = done
        return (
            {
                'picture': self._create_image_state().astype(np.uint8),
                'car_vector':
                    self.car.get_vector_state(self.bot_cars).astype(np.float32)
                    if len(self._settings['state']['vector_car_features']) != 0
                    else None,
            },
            step_reward,
            done,
            info,
        )

    def _create_image_state(self) -> Union[None, np.ndarray]:
        if not self._settings['state']['picture']:
            return None
        return self.render()

    @lru_cache(maxsize=None)
    def _create_vector_env_static_description(self) -> np.ndarray:
        params_to_use = self._settings['state']['vector_env_features']
        # that I have, just to remaind:
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

    def render(self, mode: str = 'human', full_image: Optional[bool] = False) -> np.array:
        background_image = self._data_loader.get_background()

        self.draw_car(background_image, self.car)
        for bot_car in self.bot_cars:
            self.draw_car(background_image, bot_car)

        if self._settings['state'].get('checkpoints', {'show': False})['show']:
            self.draw_track(
                background_image=background_image,
                car=self.car,
                hide_on_reach=self._settings['state']['checkpoints']['hide_on_reach'],
            )

        if self._settings['state'].get('crop_near_agent', {'use': False})['use']:
            car_x, car_y = self.car.position_IMG
            bounds = self._settings['state']['crop_near_agent']['size']

            background_image = background_image.crop((
                car_x - bounds / 2,
                car_y - bounds / 2,
                car_x + bounds / 2,
                car_y + bounds / 2,
            ))

        if full_image:
            if self._data_loader.get_animation_target_size() is not None:
                # noinspection PyTypeChecker
                return np.asarray(
                    Img.resize_pil_image(
                        background_image.convert('RGB'),
                        size=self._data_loader.get_animation_target_size(),
                    ),
                )
            # noinspection PyTypeChecker
            return np.asarray(background_image.convert('RGB'))

        # noinspection PyTypeChecker
        return np.asarray(
            Img.resize_pil_image(
                background_image.convert('RGB'),
                size=self._data_loader.get_target_image_size(),
            ),
        )

    def draw_car(self, background_image: Image.Image, car: DummyCar) -> None:
        # rotate car image and mask of car image, and compute bounds of rotated image
        masked_image, mask = self._data_loader.get_rotated_car_image(car)
        bound_y, bound_x = masked_image.size

        # car position in image coordinates (in pixels)
        car_x, car_y = car.position_IMG

        background_image.paste(
            im=masked_image,
            box=(
                int(car_x - bound_y / 2),
                int(car_y - bound_x / 2),
            ),
            mask=mask,
        )

        # DEBUG DRAW car points: center and wheels
        # self.draw_point(background_image, self.car.position_IMG)
        # for pnt in self.car.wheels_positions_IMG:
        #     self.draw_point(background_image, pnt)

    def close(self):
        del self.car
        del self.bot_cars
        del self._data_loader
        del self.world

    def draw_track(self, background_image: Image.Image, car: DummyCar, hide_on_reach=True) -> None:
        points = (
            car.track['line']
            if not hide_on_reach else
            car.track['line'][car.track_index():]
        )
        for point in points:
            p = self._data_loader.convertPLAY2IMG(point)
            self.draw_point(background_image, p)

    def draw_point(self, background_image: Image.Image, point: np.ndarray) -> None:
        ch_image = self._data_loader.checkpoint_image()
        background_image.paste(
            im=self._data_loader.checkpoint_image(),
            box=(
                int(point[0] - ch_image.size[0] / 2),
                int(point[1] - ch_image.size[1] / 2)
            ),
            mask=ch_image,
        )
