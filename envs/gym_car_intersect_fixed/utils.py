import os
from typing import List, NamedTuple, Type, Any, Optional, Tuple, Union, Dict
import cv2
import numpy as np
from shapely.geometry import Polygon
from skimage.measure import label, regionprops
import copy

from envs.gym_car_intersect_fixed.cvat import CvatDataset


class CarImage(NamedTuple):
    image: np.ndarray
    mask: np.ndarray
    real_image: np.ndarray
    real_size: np.array
    car_image_center_displacement: np.ndarray
    size: np.array
    center: np.array
    hashable_obj: str


class DataSupporter:
    """
    Class with bunch of static function for processing, loading ect of tracks, background image, cars_full image.
    Also all convertations from normal world XY coordinate system to image -YX system should be
       provided via this class functions.

    """
    # for training
    def __init__(self, settings):
        self._settings = settings

        self._background_image_scale = self._settings['image_scale']['back_image_scale_factor']
        self._car_image_scale = self._settings['image_scale']['car_image_scale_factor']

        self._background_image = cv2.imread(self._settings['background_path'])
        self._sended_background = None

        # in XY coordinates, not in a IMAGE coordinates
        self._image_size = np.array([self._background_image.shape[1], self._background_image.shape[0]])
        # just two numbers of field in pyBox2D coordinate system
        self._playfield_size = np.array([80 * self._background_image.shape[1] / self._background_image.shape[0], 80])
        # technical field
        self._data = CvatDataset()
        self._data.load(self._settings['annotation_path'])
        # list of car images
        self._cars: List[CarImage] = []
        self._load_car_images(self._settings['cars_path'])
        # list of tracks
        self._tracks: List[Dict[str, Union[np.ndarray, Polygon]]] = []
        self._agent_track_list = []
        self._bot_track_list = []
        self._extract_tracks()

        # index of image -> [dict of angle index -> [image] ]
        self._image_memory = {}

        self.FULL_RENDER_COEFF = 0.5
        self._render_back = None
        self._true_image_memory = {}

        # print some statistics
        # print(f'count of car images: {self.car_image_count}')
        # print(f'count of track count: {self.track_count}')
        # print(f'background image shape: {self._image_size * self._background_image_scale}')
        # print(f'play field shape: {self._playfield_size}')

    @property
    def car_features_list(self) -> List[str]:
        if 'vector_car_features' in self._settings['state_config'].keys():
            return self._settings['state_config']['vector_car_features']
        return []

    @property
    def get_state_picture_shape(self):
        if 'picture' in self._settings['state_config'].keys():
            return self._settings['state_config']['picture']
        return None

    @property
    def track_count(self):
        return len(self._tracks)

    @property
    def car_image_count(self):
        return len(self._cars)

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
        return coords * self._playfield_size / self._image_size

    def _convertXY_PLAY2IMG(self, coords: np.ndarray):
        """
        Technical function for pyBox2D to IMG coordinates convertation.
        """
        if coords.shape != (2, ):
            raise ValueError
        return coords * self._image_size / self._playfield_size

    @property
    def data(self):
        return self._data

    def get_background(self, true_size=False):
        if self._sended_background is None:
            self._sended_background = cv2.resize(
                self._background_image,
                None,
                fx=self._background_image_scale,
                fy=self._background_image_scale,
            )
        if true_size:
            if self._render_back is None:
                self._render_back = cv2.resize(
                    self._background_image,
                    None,
                    fx=self.FULL_RENDER_COEFF,
                    fy=self.FULL_RENDER_COEFF,
                )
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

                label_image = label(mask[:, :, 0])
                region = regionprops(label_image)[0]
                min_y, min_x, max_y, max_x = region.bbox

                region_size_y = (max_y - min_y)
                region_size_x = (max_x - min_x)

                car = CarImage(
                    mask=mask,
                    real_image=real_image,
                    real_size=np.array([real_image.shape[1], real_image.shape[0]]),
                    center=np.array([real_image.shape[1], real_image.shape[0]]),
                    car_image_center_displacement=region.centroid - np.array([real_image.shape[0], real_image.shape[1]]) / 2,
                    image=cv2.bitwise_and(real_image, mask),
                    size=np.array([region_size_x, region_size_y]),
                    hashable_obj=os.path.join(cars_path, folder, 'mask.bmp'),
                )

                self._cars.append(car)
                self._image_memory[car.hashable_obj] = dict()
            except:
                pass
                # print(f'error while parsing car image source: {os.path.join(cars_path, folder)}')

    def get_rotated_car_image(self, car, true_size=False):
        if true_size:
            if car.car_image.hashable_obj not in self._true_image_memory.keys():
                self._true_image_memory[car.car_image.hashable_obj] = dict()
            angle_index = car.angle_index
            if angle_index in self._true_image_memory[car.car_image.hashable_obj].keys():
                return self._true_image_memory[car.car_image.hashable_obj][angle_index]
            masked_image = DataSupporter.rotate_image(
                cv2.resize(
                    car.car_image.image,
                    None,
                    fx=self.FULL_RENDER_COEFF,
                    fy=self.FULL_RENDER_COEFF,
                ),
                car.angle_degree + 90,
            )
            car_mask_image = DataSupporter.rotate_image(
                cv2.resize(
                    car.car_image.mask,
                    None,
                    fx=self.FULL_RENDER_COEFF,
                    fy=self.FULL_RENDER_COEFF,
                ),
                car.angle_degree + 90,
            )
            self._true_image_memory[car.car_image.hashable_obj][angle_index] = (masked_image, car_mask_image)
            return self._true_image_memory[car.car_image.hashable_obj][angle_index]

        if car.car_image.hashable_obj not in self._image_memory.keys():
            self._image_memory[car.car_image.hashable_obj] = dict()

        angle_index = car.angle_index

        if angle_index in self._image_memory[car.car_image.hashable_obj].keys():
            return self._image_memory[car.car_image.hashable_obj][angle_index]
        try:
            masked_image = DataSupporter.rotate_image(
                cv2.resize(
                    car.car_image.image,
                    None,
                    fx=self._car_image_scale,
                    fy=self._car_image_scale,
                ),
                car.angle_degree + 90
            )
            car_mask_image = DataSupporter.rotate_image(
                cv2.resize(
                    car.car_image.mask,
                    None,
                    fx=self._car_image_scale,
                    fy=self._car_image_scale,
                ),
                car.angle_degree + 90,
            )
        except:
            print('ERROR in resize')
            print(f'car image shape : {car.car_image.image.shape}')
            print(f'car mask shape : {car.car_image.mask.shape}')
            print(f'angle : {car.angle_degree + 90}')
            print(f'scale : {self._car_image_scale}')

        self._image_memory[car.car_image.hashable_obj][angle_index] = (masked_image, car_mask_image)
        return self._image_memory[car.car_image.hashable_obj][angle_index]

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
        track_lines = {}
        for item in self._data.get_polylines(0):
            if item['label'] == 'track_line':
                track_lines[item['attributes']['index']] = np.array(item['points'])
        track_polygons = {}
        for item in self._data.get_polygons(0):
            if item['label'] == 'track':
                track_polygons[item['attributes']['index']] = np.array(item['points'])
        for index, track_line in track_lines.items():
            if index not in track_polygons.keys():
                print(f'skip track index index {index}')
                continue

            # fill inner indexes for preset track indexes
            if index in self._settings['agent_tracks'] or len(self._settings['agent_tracks']) == 0:
                self._agent_track_list.append(len(self._tracks))
            if index in self._settings['bots_tracks'] or len(self._settings['bots_tracks']) == 0:
                self._bot_track_list.append(len(self._tracks))

            self._tracks.append({
                'polygon': Polygon(self.convertIMG2PLAY(track_polygons[index])),
                'line': track_line,
            })

    @staticmethod
    def _dist(pointA, pointB) -> float:
        """
        Just Euclidean distance.
        """
        return np.sqrt(np.sum((pointA - pointB)**2))

    @staticmethod
    def _expand_track(track_obj, max_dist: float = 10.0) -> Dict[str, Any]:
        """
        Insert point in existing polyline, while dist between point more then max_dist.
        As a result track_obj['line'] will contain more points.
        """
        track = np.array(track_obj['line'])
        first_point = track[0].copy()
        expanded_track = [first_point.copy()]

        for index in range(1, len(track)):
            next_point = track[index]
            vector = next_point - first_point
            vector_len = np.sqrt(np.sum((vector ** 2)))
            if vector_len < 1e-8:
                print(vector)
                raise ValueError('oops, something went wrong, ask Michael Martinson, tg @MichaelMD')
            vector = vector / vector_len * max_dist

            if index == len(track) - 1 or DataSupporter._dist(next_point, first_point) > 1.2 * max_dist:
                expanded_point = first_point + vector
                while DataSupporter._dist(next_point, expanded_point) > 1.3 * max_dist:
                    expanded_track.append(expanded_point.copy())
                    expanded_point += vector
                expanded_track.append(expanded_point.copy())
            expanded_track.append(next_point.copy())
            first_point = next_point.copy()

        expanded_track.append(track[-1])
        return {
            'polygon': track_obj['polygon'],
            'line': np.array(expanded_track),
        }

    def peek_car_image(self, is_for_agent, index: Optional[int] = None):
        """
        Return random car image.
        :param index: integer, if provided function return index'th car image
        :return: car image, named tuple
        """
        if is_for_agent and len(self._settings['agent_image_indexes']) != 0:
            index = np.random.choice(self._settings['agent_image_indexes'])
        if index is None:
            index = np.random.choice(np.arange(len(self._cars)))
        return copy.deepcopy(self._cars[index])

    def peek_track(self, is_for_agent, expand_points: Optional[float] = 50, index: Optional[int] = None):
        """
        Return random track object.
        :param is_for_agent:
        :param expand_points: if provided increase number of points in 'line' part of track object
        :param index: if provided, function return index'th track.
        :return:
        """
        if is_for_agent:
            index = np.random.choice(self._agent_track_list)
        if not is_for_agent:
            index = np.random.choice(self._bot_track_list)

        if expand_points is not None:
            return DataSupporter._expand_track(self._tracks[index], expand_points)
        return self._tracks[index]

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
        return track[0]

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
