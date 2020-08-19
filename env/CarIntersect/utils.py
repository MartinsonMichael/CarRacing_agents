import os
from collections import defaultdict
from typing import List, NamedTuple, Optional, Tuple, Union, Dict, Set
import cv2
import numpy as np
from shapely.geometry import Polygon
from skimage.measure import label, regionprops
import copy
from PIL import Image

from .cvat_loader import CvatDataset
from .geometry_utils import TrackType
from . import geometry_utils as Geom
from . import image_utils as Img


class CarImage(NamedTuple):
    size: np.ndarray
    hashable_obj: str

    pil_image: Image.Image
    mask: Image.Image


class DataSupporter:
    """
    Class with bunch of static function for processing, loading ect of tracks, background image, cars_full image.
    Also all convertations from normal world XY coordinate system to image -YX system should be
       provided via this class functions.

    """
    def __init__(self, settings):
        self._settings = settings

        self._image_scale = self._settings['image_scale']
        self._background_image = DataSupporter._load_pil_image(self._settings['background_path'])
        self._background_image = self._background_image.convert('RGBA')

        # in XY coordinates, not in a IMAGE coordinates
        self._original_image_size = np.array([self._background_image.size[1], self._background_image.size[0]])
        assert abs(self._original_image_size[0] - self._original_image_size[1]) < 10, \
            "AAA, panic, we didn't expect such difficult case"

        # NOTE we want to use target image size
        assert isinstance(self._image_scale['image_target_size'], (list, tuple))
        assert len(self._image_scale['image_target_size']) == 2

        if self._image_scale['relative_car_scale'] is None:
            self._image_scale['relative_car_scale'] = 1.0

        self._image_scale['image_target_size'] = tuple(self._image_scale['image_target_size'])

        if 'animation_target_size' not in self._image_scale.keys():
            self._image_scale['animation_target_size'] = (200, 200)
        else:
            if isinstance(self._image_scale['animation_target_size'], (tuple, list)):
                assert len(self._image_scale['animation_target_size']) == 2
                self._image_scale['animation_target_size'] = tuple(self._image_scale['animation_target_size'])

        # just two numbers of field in pyBox2D coordinate system
        self.PLAYFIELD_SIZE = 100
        self._playfield_size = np.array([
            self.PLAYFIELD_SIZE * self._background_image.size[1] / self._background_image.size[0],
            self.PLAYFIELD_SIZE
        ])
        # technical field
        self._data = CvatDataset()
        self._data.load(self._settings['annotation_path'])

        # list of car images
        self._cars: List[CarImage] = []
        self._car_image_memory = {}
        self._load_car_images(self._settings['cars_path'])
        print(f'Loaded {len(self._cars)} car images')

        # Preparing tracks
        # _tracks is for all tracks: dict with track name as key and Dict as track description
        #   track description is also Dict with two (or one) keys : "line", "polygon"(optional)
        #       "line" is numpy with track points
        #       "polygon" is numpy array with points if surrounding polygon (may be not present for bots tracks)
        self._tracks: Dict[str, TrackType] = {}
        # _agent_track_list is just list of string - track names that can be used for agent
        self._agent_track_list = self._settings['agent_tracks']
        # _bot_track_list is just list of string - track names that can be used for bots
        self._bot_track_list = self._settings['bots_tracks']
        self._extract_tracks()

        self._track_point_image: Optional[Image.Image] = None
        self.load_track_point_image()
        if self._track_point_image is not None:
            self._track_point_image = Img.resize_pil_image(
                self._track_point_image,
                coef=self._get_checkpoint_size_coef(),
            )

        print(f'cheskpoint size : {self.get_checkpoint_size()}')

        self._eval_mode: bool = False

    def start_eval(self) -> None:
        self._eval_mode = True

    def stop_eval(self) -> None:
        self._eval_mode = False

    def get_track_start_position(self, track_obj: TrackType, is_bot: bool) -> Tuple[int, int, float, int]:
        """Return start x, y, and angle"""
        ind = 0
        if not self._eval_mode:
            if is_bot and self._settings.get('random_start_position_bots', False):
                ind = np.random.choice(len(track_obj['line']) - 1)
            if not is_bot and self._settings.get('random_start_position', False):
                ind = np.random.choice(len(track_obj['line']) - 1)

        return [
            *track_obj['line'][ind],
            Geom.angle_by_2_points(track_obj['line'][ind], track_obj['line'][ind + 1]) - np.pi / 2,
            ind
        ]

    def get_checkpoint_size(self) -> float:
        if self._track_point_image is None:
            scale = 10
        else:
            scale = self._track_point_image.size[0]
        return 1.0 * scale / 2 * self.playfield_size[0] / self._original_image_size[0]

    @staticmethod
    def image_coef() -> float:
        return 0.4

    @staticmethod
    def _load_pil_image(path: str) -> Image.Image:
        return Img.resize_pil_image(
            Image.open(path),
            coef=DataSupporter.image_coef(),
        )

    def get_target_image_size(self) -> Tuple[int, int]:
        return self._image_scale['image_target_size']

    def get_animation_target_size(self) -> Union[None, Tuple[int, int]]:
        if isinstance(self._image_scale['animation_target_size'], (list, tuple)):
            return self._image_scale['animation_target_size']
        return None

    def load_track_point_image(self) -> None:
        path_to_tp = os.path.join('.', 'env_data', 'track_point.png')
        if os.path.exists(path_to_tp):
            self._track_point_image = DataSupporter._load_pil_image(path_to_tp)
            return

        path_to_tp = os.path.join('.', 'env', 'env_data', 'track_point.png')
        if os.path.exists(path_to_tp):
            self._track_point_image = DataSupporter._load_pil_image(path_to_tp)
            return

        path_to_tp = os.path.join('..', 'env', 'env_data', 'track_point.png')
        if os.path.exists(path_to_tp):
            self._track_point_image = DataSupporter._load_pil_image(path_to_tp)
            return

        if self._settings['state']['checkpoints']['show']:
            raise ValueError("can't find checkpoint image!")

    def _get_checkpoint_size_coef(self) -> float:
        if not self._settings['state'].get('checkpoints', {'show': False})['show']:
            return 1.0
        return float(self._settings['state']['checkpoints']['size'])

    def checkpoint_image(self) -> Image.Image:
        assert self._track_point_image is not None
        return self._track_point_image

    @property
    def car_features_list(self) -> Set[str]:
        return set(self._settings['state'].get('vector_car_features', []))

    @property
    def get_background_image_scale(self) -> float:
        return self._image_scale['back_image_scale_factor']

    @property
    def track_count(self) -> int:
        return len(self._tracks)

    @property
    def playfield_size(self) -> np.array:
        return self._playfield_size

    def set_playfield_size(self, size: np.array):
        if size.shape != (2,):
            raise ValueError
        self._playfield_size = size

    @staticmethod
    def convert_XY2YX(points: np.ndarray):
        if len(points.shape) == 2:
            return np.array([points[:, 1], points[:, 0]]).T
        if points.shape == (2, ):
            return np.array([points[1], points[0]])
        raise ValueError

    @staticmethod
    def do_with_points(track_obj: TrackType, func) -> TrackType:
        """
        Perform given functions under 'line' array.
        :param track_obj: dict with two keys:
            'polygon' - shapely.geometry.Polygon object
            'line' - np.array with line points coordinates
        :param func: function to be permorm under track_obj['line']
        :return: track_obj
        """
        track_obj['line'] = func(track_obj['line'])
        return track_obj

    def convertIMG2PLAY(self, points: Union[np.array, Tuple[float, float]]) -> np.array:
        """
        Convert points from IMG pixel coordinate to pyBox2D coordinate.
        NOTE! This function doesn't flip Y to -Y, just scale coordinates.
        :param points: np.array
        :return: np.array
        """
        points = np.array(points)
        if len(points.shape) == 1:
            return self._convertXY_IMG2PLAY(points)
        else:
            return np.array([self._convertXY_IMG2PLAY(coords) for coords in points])

    def convertPLAY2IMG(self, points: Union[np.array, Tuple[float, float]]) -> np.array:
        """
        Convert points from pyBox2D coordinate to IMG pixel coordinate.
        NOTE! This function doesn't flip Y to -Y, just scale coordinates.
        :param points: np.array
        :return: np.array
        """
        points = np.array(points)
        if len(points.shape) == 1:
            return self._convertXY_PLAY2IMG(points)
        else:
            return np.array([self._convertXY_PLAY2IMG(coords) for coords in points])

    def _convertXY_IMG2PLAY(self, coords: np.ndarray):
        """
        Technical function for IMG to pyBox2D coordinates convertation.
        """
        if coords.shape != (2, ):
            raise ValueError
        return coords * self._playfield_size / self._original_image_size

    def _convertXY_PLAY2IMG(self, coords: np.ndarray):
        """
        Technical function for pyBox2D to IMG coordinates convertation.
        """
        if coords.shape != (2, ):
            raise ValueError
        return coords * self._original_image_size / self._playfield_size

    @property
    def data(self):
        return self._data

    def get_background(self) -> Image.Image:
        return self._background_image.copy()

    def _load_car_images(self, cars_path):
        """
        Technical function for car image loading.
        """
        import os
        for folder in sorted(os.listdir(cars_path)):
            pil_image = DataSupporter._load_pil_image(os.path.join(cars_path, folder, 'image.jpg'))
            mask = DataSupporter._load_pil_image(os.path.join(cars_path, folder, 'mask.bmp')).convert('1')
            if mask.size != pil_image.size:
                continue

            r, g, b = map(np.array, pil_image.convert('RGB').split())

            # noinspection PyTypeChecker
            alpha = np.where(np.array(mask) == 0, 0, 255).astype('uint8')

            pil_image = cv2.merge((r, g, b, alpha))
            pil_image = Image.fromarray(pil_image, mode='RGBA')

            car = CarImage(
                size=np.array(pil_image.size, dtype=np.int32),
                hashable_obj=f"{cars_path}--{folder}",
                pil_image=pil_image,
                mask=mask,
            )

            self._cars.append(car)
            self._car_image_memory[car.hashable_obj] = dict()

    def get_rotated_car_image(self, car) -> Tuple[Image.Image, Image.Image]:
        if car.car_image.hashable_obj not in self._car_image_memory.keys():
            self._car_image_memory[car.car_image.hashable_obj] = dict()
        angle_index = car.angle_index

        if angle_index in self._car_image_memory[car.car_image.hashable_obj].keys():
            return self._car_image_memory[car.car_image.hashable_obj][angle_index]

        pil_rotated_resized = Img.rotate_pil_image(
            Img.resize_pil_image(car.car_image.pil_image, self._image_scale['relative_car_scale']),
            -car.angle_degree - 90,
        )

        pil_rotated_resized_mask = Img.rotate_pil_image(
            Img.resize_pil_image(car.car_image.mask, self._image_scale['relative_car_scale']),
            -car.angle_degree - 90,
        )

        self._car_image_memory[car.car_image.hashable_obj][angle_index] = (
            pil_rotated_resized,
            pil_rotated_resized_mask.convert('1'),
        )
        return self._car_image_memory[car.car_image.hashable_obj][angle_index]

    def _extract_tracks(self):
        """
        Technical function for track loading.
        """
        tracks = defaultdict(lambda: {'line': None, 'polygon': None}, {})
        for item in self._data.get_polylines(0):
            if item['label'] == 'track_line':
                # noinspection PyTypeChecker
                tracks[item['attributes']['title']]['line'] = np.array(item['points']) * DataSupporter.image_coef()

        for item in self._data.get_polygons(0):
            if item['label'] == 'track_polygon':
                # noinspection PyTypeChecker
                tracks[item['attributes']['title']]['polygon'] = np.array(item['points']) * DataSupporter.image_coef()

        self._agent_track_list = set(self._agent_track_list)

        for track_title, track_object in tracks.items():

            if track_object['line'] is None:
                print(f"Skip track {track_title}, because it hasn't got track line")
                continue

            if track_object['polygon'] is None:
                self._agent_track_list = self._agent_track_list - {track_title}
                self._tracks[track_title] = {
                    'polygon': None,
                    'line': self.convertIMG2PLAY(track_object['line']),
                }
                continue

            self._tracks[track_title] = {
                'polygon': Polygon(self.convertIMG2PLAY(track_object['polygon'])),
                'line': self.convertIMG2PLAY(track_object['line']),
            }

        self._agent_track_list = np.array(list(self._agent_track_list))
        self._bot_track_list = np.array(list(self._bot_track_list))

    def peek_car_image(self, is_for_agent: bool, index: Optional[int] = None):
        """
        Return random car image.
        :param is_for_agent: bool
        :param index: integer, if provided function return index'th car image
        :return: car image, named tuple
        """
        if is_for_agent and len(self._settings['agent_image_indexes']) != 0:
            index = np.random.choice(self._settings['agent_image_indexes'])
        if index is None:
            index = np.random.choice(np.arange(len(self._cars)))
        return copy.deepcopy(self._cars[index])

    def peek_track(self, is_for_agent, expand_points_pixels: Optional[float] = 50) -> TrackType:
        """
        Return random track object.
        :param is_for_agent:
        :param expand_points_pixels: if provided increase number of points in 'line' part of track object
        :return: TrackType
        """
        if is_for_agent:
            index = np.random.choice(self._agent_track_list)
            track = self._tracks[index]
        else:
            index = np.random.choice(self._bot_track_list)
            track = self._tracks[index]

        if expand_points_pixels is not None:
            return Geom.expand_track(
                track,
                expand_points_pixels * self._playfield_size[0] / self._original_image_size[0],
            )
        return track
