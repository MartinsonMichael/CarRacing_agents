import os
from collections import defaultdict
from typing import List, NamedTuple, Any, Optional, Tuple, Union, Dict, Set
import cv2
import numpy as np
from shapely.geometry import Polygon
from skimage.measure import label, regionprops
import copy
from PIL import Image

from env.CarIntersect.cvat_loader import CvatDataset


class CarImage(NamedTuple):
    # image: np.ndarray
    # mask: np.ndarray
    # real_image: np.ndarray
    # real_size: np.ndarray
    # car_image_center_displacement: np.ndarray
    size: np.ndarray
    # center: np.ndarray
    hashable_obj: str

    pil_image: Image.Image
    mask: np.ndarray


class DataSupporter:
    """
    Class with bunch of static function for processing, loading ect of tracks, background image, cars_full image.
    Also all convertations from normal world XY coordinate system to image -YX system should be
       provided via this class functions.

    """
    # for training
    def __init__(self, settings):
        self._settings = settings

        self._image_scale = self._settings['image_scale']
        self._background_image: Image.Image = Image.open(self._settings['background_path'])
        self._background_image.convert('RGBA')
        self._sended_background = None

        # in XY coordinates, not in a IMAGE coordinates
        self._original_image_size = np.array([self._background_image.size[1], self._background_image.size[0]])

        # two cases of image scaling
        # case one, we have tow float number for scaling background image and for scaling car images
        # base on both cases we wiil compute two entities: target size of image and scaling factor for car images
        if 'image_target_size' not in self._image_scale.keys() or self._image_scale['image_target_size'] is None:
            assert self._image_scale['back_image_scale_factor'] is not None
            assert self._image_scale['car_image_scale_factor'] is not None

            self._image_scale['image_target_size'] = tuple(map(int,
                self._original_image_size * self._image_scale['back_image_scale_factor']
            ))
            print(self._image_scale['image_target_size'])
        # second case, we have target size of image, and relative car image scaling
        else:
            assert isinstance(self._image_scale['image_target_size'], (list, tuple))
            assert len(self._image_scale['image_target_size']) == 2

            if self._image_scale['relative_car_scale'] is None:
                self._image_scale['relative_car_scale'] = 1.0

            self._image_scale['back_image_scale_factor'] = (
                self._image_scale['image_target_size'][0] / self._original_image_size[0]
            )

            # this parameter will be used for car image scaling
            self._image_scale['car_image_scale_factor'] = (
                self._image_scale['back_image_scale_factor'] * self._image_scale['relative_car_scale']
            )

            self._image_scale['image_target_size'] = tuple(self._image_scale['image_target_size'])

        # just two numbers of field in pyBox2D coordinate system
        # BLACK_MAGIC_CONST = 335
        self.BLACK_MAGIC_CONST = 100
        self._playfield_size = np.array([
            self.BLACK_MAGIC_CONST * self._background_image.size[1] / self._background_image.size[0],
            self.BLACK_MAGIC_CONST
        ])
        # technical field
        self._data = CvatDataset()
        self._data.load(self._settings['annotation_path'])

        # list of car images
        self._cars: List[CarImage] = []
        self._image_memory = {}
        self._load_car_images(self._settings['cars_path'])

        # Preparing tracks
        # _tracks is for all tracks: dict with track name as key and Dict as track description
        #   track description is also Dict with two (or one) keys : "line", "polygon"(optional)
        #       "line" is numpy with track points
        #       "polygon" is numpy array with points if surrounding polygon (may be not present for bots tracks)
        self._tracks: Dict[str, Dict[str, Union[np.ndarray, Polygon]]] = {}
        # _agent_track_list is just list of string - track names that can be used for agent
        self._agent_track_list = self._settings['agent_tracks']
        # _bot_track_list is just list of string - track names that can be used for bots
        self._bot_track_list = self._settings['bots_tracks']
        self._extract_tracks()

        # index of image -> [dict of angle index -> [image] ]
        # like a memory cache for rotated car images

        # coefficients of images scaling for animation records, not so small as for RL image state
        self.FULL_RENDER_COEFF = self._settings['image_scale']['image_scale_for_animation_records']
        self._render_back = None
        self._true_image_memory = {}

        # print some statistics
        # print(f'count of car images: {self.car_image_count}')
        # print(f'count of track count: {self.track_count}')
        # print(f'background image shape: {self._image_size * self._background_image_scale}')
        # print(f'play field shape: {self._playfield_size}')

        self._track_point_image: Optional[Image.Image] = None
        self.load_track_point_image()

    def load_track_point_image(self) -> None:
        path_to_tp = os.path.join('.', 'env_data', 'track_point.png')

        # print(os.listdir(path_to_tp))

        if os.path.exists(path_to_tp):
            self._track_point_image = Image.open(path_to_tp)

    def checkpoint_image(self, full_image: bool = True) -> Image.Image:
        assert self._track_point_image is not None
        if full_image:
            return self._track_point_image

        if not hasattr(self, '_track_point_image_not_full'):
            self._track_point_image_not_full: Image.Image = DataSupporter.resize_pil_image(
                self._track_point_image,
                coef=self.get_background_image_scale / self.FULL_RENDER_COEFF,
            )
        return self._track_point_image_not_full

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
    def convert_XY2YX(points: np.array):
        if len(points.shape) == 2:
            return np.array([points[:, 1], points[:, 0]]).T
        if points.shape == (2, ):
            return np.array([points[1], points[0]])
        raise ValueError

    @staticmethod
    def do_with_points(track_obj, func):
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

    def get_background(self, true_size=False) -> Image.Image:
        if self._sended_background is None:
            self._sended_background = DataSupporter.resize_pil_image(
                self._background_image,
                size=self._image_scale['image_target_size'],
            ).convert('RGBA')
        if true_size:
            if self._render_back is None:
                self._render_back = DataSupporter.resize_pil_image(
                    self._background_image,
                    coef=self.FULL_RENDER_COEFF,
                ).convert('RGBA')
            return self._render_back.copy()
        return self._sended_background.copy()

    def _load_car_images(self, cars_path):
        """
        Technical function for car image loading.
        """
        import os
        for folder in sorted(os.listdir(cars_path)):
            try:
                mask = cv2.imread(os.path.join(cars_path, folder, 'mask.bmp'))
                real_image = cv2.imread(os.path.join(cars_path, folder, 'image.jpg'))

                if mask.shape[:2] != real_image.shape[:2]:
                    continue

                del real_image

                label_image = label(mask[:, :, 0])
                region = regionprops(label_image)[0]
                min_y, min_x, max_y, max_x = region.bbox

                region_size_y = (max_y - min_y)
                region_size_x = (max_x - min_x)

                pil_image: Image.Image = Image.open(os.path.join(cars_path, folder, 'image.jpg'))
                pil_image.convert('RGB')
                r, g, b = map(np.array, pil_image.split())
                alpha = np.where(mask == 0, 0, 255).astype('uint8')[:, :, 0]
                pil_image = cv2.merge((r, g, b, alpha))
                pil_image = Image.fromarray(pil_image, mode='RGBA')

                car = CarImage(
                    size=np.array([region_size_x, region_size_y]),
                    hashable_obj=os.path.join(cars_path, folder, 'mask.bmp'),
                    pil_image=pil_image,
                    mask=Image.open(os.path.join(cars_path, folder, 'mask.bmp')),
                )

                self._cars.append(car)
                self._image_memory[car.hashable_obj] = dict()
            except Exception as e:
                raise e
                # print(f'error while parsing car image source: {os.path.join(cars_path, folder)}')

    def get_rotated_car_image(self, car, true_size=False) -> Tuple[Image.Image, Image.Image]:
        if true_size:
            if car.car_image.hashable_obj not in self._true_image_memory.keys():
                self._true_image_memory[car.car_image.hashable_obj] = dict()
            angle_index = car.angle_index

            if angle_index in self._true_image_memory[car.car_image.hashable_obj].keys():
                return self._true_image_memory[car.car_image.hashable_obj][angle_index]

            pil_rotated_resized = DataSupporter.rotate_pil_image(
                DataSupporter.resize_pil_image(car.car_image.pil_image, self.FULL_RENDER_COEFF),
                -car.angle_degree - 90,
            )
            pil_rotated_resized_mask = DataSupporter.rotate_pil_image(
                DataSupporter.resize_pil_image(car.car_image.mask, self.FULL_RENDER_COEFF),
                -car.angle_degree - 90,
            )

            self._true_image_memory[car.car_image.hashable_obj][angle_index] = \
                (pil_rotated_resized, pil_rotated_resized_mask.convert('1'))
            return self._true_image_memory[car.car_image.hashable_obj][angle_index]

        if car.car_image.hashable_obj not in self._image_memory.keys():
            self._image_memory[car.car_image.hashable_obj] = dict()

        angle_index = car.angle_index

        if angle_index in self._image_memory[car.car_image.hashable_obj].keys():
            return self._image_memory[car.car_image.hashable_obj][angle_index]

        pil_rotated_resized = DataSupporter.rotate_pil_image(
            DataSupporter.resize_pil_image(car.car_image.pil_image, self._image_scale['car_image_scale_factor']),
            -car.angle_degree - 90,
        )
        pil_rotated_resized_mask = DataSupporter.rotate_pil_image(
            DataSupporter.resize_pil_image(car.car_image.mask, self._image_scale['car_image_scale_factor']),
            -car.angle_degree - 90,
        )
        self._image_memory[car.car_image.hashable_obj][angle_index] =\
            (pil_rotated_resized, pil_rotated_resized_mask.convert('1'))

        return self._image_memory[car.car_image.hashable_obj][angle_index]

    @staticmethod
    def resize_pil_image(image: Image, coef: Optional[float] = None, size: Optional[Tuple[int, int]] = None) -> Image:
        if size is None:
            assert coef is not None
            (width, height) = (int(image.width * coef), int(image.height * coef))
        else:
            assert isinstance(size, (list, tuple))
            assert len(size) == 2
            (width, height) = size
        im_resized = image.resize((width, height))
        return im_resized

    @staticmethod
    def rotate_pil_image(image: Image, angle_degree: float) -> Image:
        return image.rotate(angle=angle_degree, fillcolor=(0, 0, 0, 0), expand=True)

    @staticmethod
    def rotate_image(image, angle, scale=1.0):
        # grab the dimensions of the image and then determine the
        # center
        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)

        # grab the rotation matrix (applying the negative of the
        # angle to rotate clockwise), then grab the sine and cosine
        # (i.e., the rotation components of the matrix)
        M = cv2.getRotationMatrix2D((cX, cY), -angle, scale)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        # compute the new bounding dimensions of the image
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))

        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY

        # perform the actual rotation and return the image
        return cv2.warpAffine(image, M, (nW, nH))

    def _extract_tracks(self):
        """
        Technical function for track loading.
        """
        tracks = defaultdict(lambda: {'line': None, 'polygon': None}, {})
        for item in self._data.get_polylines(0):
            if item['label'] == 'track_line':
                tracks[item['attributes']['title']]['line'] = np.array(item['points'])

        for item in self._data.get_polygons(0):
            if item['label'] == 'track_polygon':
                tracks[item['attributes']['title']]['polygon'] = np.array(item['points'])

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
                # 'polygon': Polygon(track_object['polygon']),
                'line': self.convertIMG2PLAY(track_object['line']),
            }

        self._agent_track_list = np.array(list(self._agent_track_list))
        self._bot_track_list = np.array(list(self._bot_track_list))

    @staticmethod
    def _dist(pointA, pointB) -> float:
        """
        Just Euclidean distance.
        """
        return np.sqrt(np.sum((pointA - pointB)**2))

    def _expand_track(self, track_obj, max_dist: float = 10.0) -> Dict[str, Any]:
        """
        Insert point in existing polyline, while dist between point more then max_dist.
        As a result track_obj['line'] will contain more points.
        """
        max_dist /= self._background_image.size[0] / self._playfield_size[0]
        track = np.array(track_obj['line'])
        expanded_track = [track[0]]

        for index in range(1, len(track)):
            first, second = track[index - 1], track[index]
            num_points_to_insert = int((DataSupporter._dist(first, second) - max_dist * 0.7) / max_dist)
            if num_points_to_insert == 0:
                expanded_track.append(second)
                continue

            vector_to_add = (second - first) / DataSupporter._vector_len(second - first) * max_dist
            for i in range(1, num_points_to_insert + 1):
                expanded_track.append(first + vector_to_add * i)
            expanded_track.append(second)

        return {
            'polygon': track_obj['polygon'],
            'line': np.array(expanded_track),
        }

    @staticmethod
    def _vector_len(v: np.ndarray) -> float:
        return float(np.sqrt((v**2).sum()))

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

    def peek_track(self, is_for_agent, expand_points: Optional[float] = 50):
        """
        Return random track object.
        :param is_for_agent:
        :param expand_points: if provided increase number of points in 'line' part of track object
        :return:
        """
        if is_for_agent:
            index = np.random.choice(self._agent_track_list)
            track = self._tracks[index]
        else:
            index = np.random.choice(self._bot_track_list)
            track = self._tracks[index]

        # track['line'] = self.convertIMG2PLAY(track['line'])

        if expand_points is not None:
            return self._expand_track(track, expand_points)
        return track

    @staticmethod
    def dist(pointA: np.array, pointB: np.array) -> float:
        """
        Just another Euclidean distance.
        """
        if pointA.shape != (2, ) or pointB.shape != (2, ):
            raise ValueError('incorrect points shape')
        return np.sqrt(np.sum((pointA - pointB)**2))

    @staticmethod
    def get_track_angle(track_obj: np.array, index=0) -> float:
        """
        Return angle between OX and track_obj['line'][index] -> track_obj['line'][index + 1] points
        """
        track = track_obj
        if isinstance(track_obj, dict):
            track = track_obj['line']
        if index == len(track):
            index -= 1
        angle = DataSupporter.angle_by_2_points(
            track[index],
            track[index + 1]
        )
        return angle

    @staticmethod
    def get_track_initial_position(track: Union[np.array, Dict[str, Any]]) -> np.array:
        """
        Just return starting position for track object.
        """
        if isinstance(track, dict):
            return track['line'][0]
        if isinstance(track, (np.ndarray, list)):
            return track[0]

        raise ValueError('unknown track type, fix me!')

    @staticmethod
    def angle_by_2_points(
            pointA: np.array,
            pointB: np.array,
    ) -> float:
        return DataSupporter.angle_by_3_points(
            np.array(pointA) + np.array([1.0, 0.0]),
            np.array(pointA),
            np.array(pointB),
        )

    @staticmethod
    def angle_by_3_points(
            pointA: np.array,
            pointB: np.array,
            pointC: np.array) -> float:
        """
        compute angle
        :param pointA: np.array of shape (2, )
        :param pointB: np.array of shape (2, )
        :param pointC: np.array of shape (2, )
        :return: angle in radians between AB and BC
        """
        if pointA.shape != (2,) or pointB.shape != (2,) or pointC.shape != (2,):
            raise ValueError('incorrect points shape')

        def unit_vector(vector):
            return vector / np.linalg.norm(vector)

        def angle_between(v1, v2):
            v1_u = unit_vector(v1)
            v2_u = unit_vector(v2)
            return np.arctan2(np.cross(v1_u, v2_u), np.dot(v1_u, v2_u))

        return angle_between(pointA - pointB, pointC - pointB)
