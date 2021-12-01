import math

import numpy as np

from environment.env_horizon.misc_ic import map_info, CROSS_TASKS


def shift_coordination(orig_x, orig_y, coordi_shift_x, coordi_shift_y):
    '''
    :param orig_x: original x
    :param orig_y: original y
    :param coordi_shift_x: coordi_shift_x along x axis
    :param coordi_shift_y: coordi_shift_y along y axis
    :return: shifted_x, shifted_y
    '''
    shifted_x = orig_x - coordi_shift_x
    shifted_y = orig_y - coordi_shift_y
    return shifted_x, shifted_y


def rotate_coordination(orig_x, orig_y, orig_d, coordi_rotate_d):
    """
    :param orig_x: original x
    :param orig_y: original y
    :param orig_d: original degree
    :param coordi_rotate_d: coordination rotation d, positive if anti-clockwise, unit: deg
    :return:
    transformed_x, transformed_y, transformed_d(range:(-180 deg, 180 deg])
    """

    coordi_rotate_d_in_rad = coordi_rotate_d * math.pi / 180
    transformed_x = orig_x * math.cos(coordi_rotate_d_in_rad) + orig_y * math.sin(coordi_rotate_d_in_rad)
    transformed_y = -orig_x * math.sin(coordi_rotate_d_in_rad) + orig_y * math.cos(coordi_rotate_d_in_rad)
    transformed_d = orig_d - coordi_rotate_d
    if transformed_d > 180:
        while transformed_d > 180:
            transformed_d = transformed_d - 360
    elif transformed_d <= -180:
        while transformed_d <= -180:
            transformed_d = transformed_d + 360
    else:
        transformed_d = transformed_d
    return transformed_x, transformed_y, transformed_d


def shift_and_rotate_coordination(orig_x, orig_y, orig_d, coordi_shift_x, coordi_shift_y, coordi_rotate_d):
    shift_x, shift_y = shift_coordination(orig_x, orig_y, coordi_shift_x, coordi_shift_y)
    transformed_x, transformed_y, transformed_d \
        = rotate_coordination(shift_x, shift_y, orig_d, coordi_rotate_d)
    return transformed_x, transformed_y, transformed_d


def rotate_and_shift_coordination(orig_x, orig_y, orig_d, coordi_shift_x, coordi_shift_y, coordi_rotate_d):
    shift_x, shift_y, transformed_d \
        = rotate_coordination(orig_x, orig_y, orig_d, coordi_rotate_d)
    transformed_x, transformed_y = shift_coordination(shift_x, shift_y, coordi_shift_x, coordi_shift_y)

    return transformed_x, transformed_y, transformed_d


def cal_info_in_transform_coordination(filtered_objects, x, y, rotate_d):  # rotate_d is positive if anti
    results = []
    for obj in filtered_objects:
        orig_x = obj['x']
        orig_y = obj['y']
        orig_v = obj['v']
        orig_heading = obj['phi']
        width = obj['w']
        length = obj['l']
        route = obj['route']
        shifted_x, shifted_y = shift_coordination(orig_x, orig_y, x, y)
        trans_x, trans_y, trans_heading = rotate_coordination(shifted_x, shifted_y, orig_heading, rotate_d)
        trans_v = orig_v
        results.append({'x': trans_x,
                        'y': trans_y,
                        'v': trans_v,
                        'phi': trans_heading,
                        'w': width,
                        'l': length,
                        'route': route,})
    return results


def cal_ego_info_in_transform_coordination(ego_dynamics, x, y, rotate_d):
    orig_x, orig_y, orig_a, corner_points = ego_dynamics['x'], ego_dynamics['y'], ego_dynamics['phi'], ego_dynamics['Corner_point']
    shifted_x, shifted_y = shift_coordination(orig_x, orig_y, x, y)
    trans_x, trans_y, trans_a = rotate_coordination(shifted_x, shifted_y, orig_a, rotate_d)
    trans_corner_points = []
    for corner_x, corner_y in corner_points:
        shifted_x, shifted_y = shift_coordination(corner_x, corner_y, x, y)
        trans_corner_x, trans_corner_y, _ = rotate_coordination(shifted_x, shifted_y, orig_a, rotate_d)
        trans_corner_points.append((trans_corner_x, trans_corner_y))
    ego_dynamics.update(dict(x=trans_x,
                             y=trans_y,
                             phi=trans_a,
                             Corner_point=trans_corner_points))
    return ego_dynamics




def _convert_car_coord_to_sumo_coord(x_in_car_coord, y_in_car_coord, a_in_car_coord, car_length):  # a in deg
    x_in_sumo_coord = x_in_car_coord + car_length / 2 * math.cos(math.radians(a_in_car_coord))
    y_in_sumo_coord = y_in_car_coord + car_length / 2 * math.sin(math.radians(a_in_car_coord))
    a_in_sumo_coord = -a_in_car_coord + 90.
    return x_in_sumo_coord, y_in_sumo_coord, a_in_sumo_coord


def _convert_sumo_coord_to_car_coord(x_in_sumo_coord, y_in_sumo_coord, a_in_sumo_coord, car_length):
    a_in_car_coord = -a_in_sumo_coord + 90.
    x_in_car_coord = x_in_sumo_coord - (math.cos(a_in_car_coord / 180. * math.pi) * car_length / 2)
    y_in_car_coord = y_in_sumo_coord - (math.sin(a_in_car_coord / 180. * math.pi) * car_length / 2)
    return x_in_car_coord, y_in_car_coord, deal_with_phi(a_in_car_coord)


def deal_with_phi(phi):
    while phi > 180:
        phi -= 360
    while phi <= -180:
        phi += 360
    return phi


def coordi_dynamics2crossing(x, y, a, x_center, y_center, upvector):
    angle = np.arctan2(upvector[1], upvector[0]) * 180 / np.pi - 90
    xt, yt, at = shift_and_rotate_coordination(x, y, a, x_center, y_center, angle)
    return xt, yt, at


def coordi_crossing2dynamics(x, y, a, x_center, y_center, upvector):
    angle = np.arctan2(upvector[1], upvector[0]) * 180 / np.pi - 90
    xt, yt, at = rotate_and_shift_coordination(x, y, a, -x_center, -y_center, -angle)
    return xt, yt, at


def wrapped_point_crossing2dynamics(points_list, x_center, y_center, upvector):
    points_list_new = []
    for p in points_list:
        xt, yt, _ = coordi_crossing2dynamics(p[0], p[1], 0, x_center, y_center, upvector)
        points_list_new.append((xt, yt))
    return points_list_new


def wrapped_point_dynamics2crossing(points_list, x_center, y_center, upvector):
    points_list_new = []
    for p in points_list:
        xt, yt, _ = coordi_dynamics2crossing(p[0], p[1], 0, x_center, y_center, upvector)
        points_list_new.append((xt, yt))
    return points_list_new


def points2line(points, close=False):
    x_list = []
    y_list = []
    for p in points:
        x_list.append(p[0])
        y_list.append(p[1])
    if close:
        x_list.append(points[0][0])
        y_list.append(points[0][1])
    return x_list, y_list


def set_rs_params_cx(from_edge):
    x_center, y_center = map_info[CROSS_TASKS[from_edge]['main_crossing']]['position']
    tmp_edge = CROSS_TASKS[from_edge]['sub_route'][-2]
    tmp_lane = map_info[tmp_edge]['lane_list'][-1]
    tmp_lane_shape = map_info[tmp_edge][tmp_lane]['shape']
    upvector = (tmp_lane_shape[1][0] - tmp_lane_shape[0][0],
                tmp_lane_shape[1][1] - tmp_lane_shape[0][1])
    return x_center, y_center, upvector

def get_edge_info_ml(current_edge):
    # 道路中心点，道路前进方向
    mid_index = map_info[current_edge]['num_lane'] // 2
    mid_lane = map_info[current_edge]['lane_list'][mid_index]
    mid_position = map_info[current_edge][mid_lane]['shape']
    start_point, end_point = mid_position
    x_center, y_center = (start_point[0] + end_point[0]) / 2, (start_point[1] + end_point[1]) / 2
    rightarray = end_point[0] - start_point[0], end_point[1] - start_point[1]
    upvector = -rightarray[1], rightarray[0]

    right_position = end_point
    num_lane = map_info[current_edge]['num_lane']
    return (x_center, y_center, upvector), right_position, num_lane

def line2points(x_list, y_list, close=False):
    points = []
    for x, y in zip(x_list, y_list):
        points.append((x, y))

    if close:
        return points[:-1]
    else:
        return points


if __name__ == '__main__':
    pass
