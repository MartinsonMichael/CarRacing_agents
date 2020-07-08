from typing import Tuple, Optional, Dict, Union

import numpy as np
from shapely.geometry import Polygon


TrackType = Dict[str, Union[Polygon, np.ndarray]]


def is_on_segment(x, a, b) -> bool:
    """
    if point x lief on the segmetn [a, b], there x, a, b - point in 2d
    """
    assert len(x) == 2 and len(a) == 2 and len(b) == 2
    AB = np.array(a) - np.array(b)
    AX = np.array(x) - np.array(a)
    BX = np.array(x) - np.array(b)
    return np.sign(np.dot(AB, AX)) == np.sign(np.dot(-AB, BX))


def get_distanced_point_on_segment(X, A, B, max_dist: float) -> Tuple[bool, Optional[np.ndarray]]:
    """

    """
    assert len(X) == 2 and len(A) == 2 and len(B) == 2
    B_vect = np.array(A) - np.array(X)
    B_dist = np.linalg.norm(B_vect)

    seg_vect = np.array(A) - np.array(B)

    cos_alpha = np.dot(B_vect, seg_vect) / B_dist / np.linalg.norm(seg_vect)

    det_sqrt = np.sqrt(B_dist ** 2 * cos_alpha ** 2 - B_dist ** 2 + max_dist ** 2)

    c1 = B_dist * cos_alpha + det_sqrt
    P1 = np.array(A) - c1 * seg_vect / np.linalg.norm(seg_vect)
    p1_on_seg = is_on_segment(P1, A, B)

    c2 = B_dist * cos_alpha - det_sqrt
    P2 = np.array(A) - c2 * seg_vect / np.linalg.norm(seg_vect)
    p2_on_seg = is_on_segment(P2, A, B)

    if p1_on_seg and not p2_on_seg:
        return True, P1
    if not p1_on_seg and p2_on_seg:
        return True, P2
    if not p1_on_seg and not p2_on_seg:
        return False, None

    # case P1 and P2 on segmnt, return more coused to segment end

    if np.sum((P1 - B) ** 2) < np.sum((P2 - B) ** 2):
        return True, P1
    return True, P2


def expand_track(track_obj: TrackType, max_dist: float = 10.0) -> TrackType:
    """
    Insert point in existing polyline, while dist between point more then max_dist.
    As a result track_obj['line'] will contain more points.
    """
    track = np.array(track_obj['line'])
    expanded_track = [track[0]]

    last_point = track[0]
    cur_track_index = 1
    cur_state = 'on-segment'

    while True:

        if cur_track_index >= len(track):
            break

        if cur_state == 'on-segment':
            if dist(last_point, track[cur_track_index]) >= max_dist:
                vector = track[cur_track_index] - last_point
                last_point = last_point + vector * max_dist / np.linalg.norm(vector)
                expanded_track.append(last_point)
                continue
            cur_state = 'find-next-segment'
            cur_track_index += 1
            continue

        if cur_state == 'find-next-segment':
            ok, point = get_distanced_point_on_segment(
                last_point,
                track[cur_track_index - 1], track[cur_track_index],
                max_dist,
            )
            if not ok:
                cur_track_index += 1
                continue
            last_point = point
            expanded_track.append(last_point)
            cur_state = 'on-segment'
            continue

    if dist(expanded_track[-1], track[-1]) > 0.75 * max_dist:
        expanded_track.append(track[-1])

    return {
        'polygon': track_obj['polygon'],
        'line': np.array(expanded_track),
    }


def dist(pointA: np.array, pointB: np.array) -> float:
    """
    Just another Euclidean distance.
    """
    if pointA.shape != (2,) or pointB.shape != (2,):
        raise ValueError('incorrect points shape')
    return np.sqrt(np.sum((pointA - pointB) ** 2))


def get_track_angle(track_obj: TrackType) -> float:
    """
    Return angle between OX and track_obj['line'][index] -> track_obj['line'][index + 1] points
    """
    track = track_obj
    if isinstance(track_obj, dict):
        track = track_obj['line']

    assert isinstance(track, np.ndarray)
    assert len(track) >= 2

    return angle_by_2_points(track[0], track[1])


def get_track_initial_position(track: TrackType) -> np.array:
    """
    Just return starting position for track object.
    """
    if isinstance(track, dict):
        return track['line'][0]
    if isinstance(track, (np.ndarray, list)):
        return track[0]

    raise ValueError('unknown track type, fix me!')


def angle_by_2_points(A: np.ndarray, B: np.ndarray) -> float:
    return angle_by_3_points(
        np.array(A) + np.array([1.0, 0.0]),
        np.array(A),
        np.array(B),
    )


def angle_by_3_points(A: np.ndarray, B: np.ndarray, C: np.ndarray) -> float:
    """
    compute angle
    :param A: np.array of shape (2, )
    :param B: np.array of shape (2, )
    :param C: np.array of shape (2, )
    :return: angle in radians between AB and BC
    """
    if A.shape != (2,) or B.shape != (2,) or C.shape != (2,):
        raise ValueError('incorrect points shape')

    def unit_vector(vector):
        return vector / np.linalg.norm(vector)

    def angle_between(v1, v2):
        v1_u = unit_vector(v1)
        v2_u = unit_vector(v2)
        return np.arctan2(np.cross(v1_u, v2_u), np.dot(v1_u, v2_u))

    return angle_between(A - B, C - B)
