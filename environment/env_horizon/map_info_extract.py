import pprint  # noqa
import os
import shutil
import sys
import json

assert "SUMO_HOME" in os.environ, "please declare environment variable 'SUMO_HOME'"
tools = os.path.join(os.environ["SUMO_HOME"], "tools")
sys.path.append(tools)
import sumolib
import traci

import numpy as np


def save_json(path, target_dict):
    with open(path, "wt") as f:
        json.dump(target_dict, f, indent=2)
    return


def start_sumo():
    SUMO_BINARY = sumolib.checkBinary("sumo")

    dirname = os.path.dirname(__file__)
    SUMOCFG_DIR = dirname + "/sumo_file/cfg/IC_whole.sumocfg"
    step_time_str = str(float(100.0) / 1000)

    try:
        traci.start(
            [
                SUMO_BINARY,
                "-c",
                SUMOCFG_DIR,
                "--step-length",
                step_time_str,
                # "--lateral-resolution", "3.5",
                "--random",
                # "--start",
                # "--quit-on-end",
                "--no-warnings",
                "--no-step-log",
                # '--seed', str(int(seed))
            ],
            numRetries=5,
        )  # '--seed', str(int(seed))
    except traci.exceptions.FatalTraCIError:
        print("Retry by other port")
        port = sumolib.miscutils.getFreeSocketPort()
        traci.start(
            [
                SUMO_BINARY,
                "-c",
                SUMOCFG_DIR,
                "--step-length",
                step_time_str,
                "--lateral-resolution",
                "3.5",
                "--random",
                # "--start",
                # "--quit-on-end",
                "--no-warnings",
                "--no-step-log",
                # '--seed', str(int(seed))
            ],
            port=port,
            numRetries=5,
        )  # '--seed', str(int(seed))


def get_lane_of_edge(edge_ID):
    """
    get lane of the given edge, and the lane should not allow vehicle class in vehicle_class_list
    """
    num_lane_of_edge = traci.edge.getLaneNumber(edge_ID)
    laneID_list_of_edge = ["{}_{}".format(edge_ID, i) for i in range(num_lane_of_edge)]
    return laneID_list_of_edge


def filting_indicator(lane):

    allowed = traci.lane.getAllowed(lane)
    disallowed = traci.lane.getDisallowed(lane)
    if len(allowed) == 1:
        if allowed[0] == "pedestrian":
            return "pedestrian"
        elif allowed[0] == "bicycle":
            return "bicycle"
        else:
            raise ValueError("Known allowed type")
    elif len(allowed) == 0:
        return "x"
    else:
        return "vehicle"


def get_edge_path(edgeID):
    output_dict = {}
    num_lane = traci.edge.getLaneNumber(edgeID)
    output_dict["type"] = "edge"
    output_dict["num_lane"] = num_lane
    laneID_list = ["{}_{}".format(edgeID, i) for i in range(num_lane)]
    # output_dict['lane_list'] = [ll for ll in laneID_list if filting_indicator(ll) == 'vehicle']
    output_dict["lane_list"] = laneID_list
    for lane in laneID_list:
        lane_dict = {}
        next_node = traci.lane.getLinks(lane)
        lane_dict["to_lane"] = [n[0] for n in next_node]
        lane_dict["to_edge"] = [traci.lane.getEdgeID(la) for la in lane_dict["to_lane"]]
        lane_dict["shape"] = traci.lane.getShape(lane)

        sp = lane_dict["shape"][0]
        ep = lane_dict["shape"][1]

        def get_vlv_position(sp, ep):
            x1, x2 = sp
            y1, y2 = ep

            dir_vec = y1 - x1, y2 - x2

            dir_vec_len = np.sqrt(np.square(dir_vec[0]) + np.square(dir_vec[1]))

            xn = y1 + dir_vec[0] / dir_vec_len * 2.5
            yn = y2 + dir_vec[1] / dir_vec_len * 2.5
            an = np.arctan2(ep[1] - sp[1], ep[0] - sp[0]) * 180 / np.pi
            return xn, yn, an

        lane_dict["vlv"] = get_vlv_position(sp, ep)
        print(sp, ep, lane_dict["vlv"])
        lane_dict["type"] = filting_indicator(lane)
        output_dict[lane] = lane_dict
        lane_dict["lane_width"] = traci.lane.getWidth(lane)
        # if filting_indicator(lane) == 'vehicle':
        #     output_dict[lane] = lane_dict
        # else:
        #     # print(filting_indicator(lane))
        #     pass
    return output_dict


def get_junction_path(junctionID, from_edge, to_edge, junction_dict):
    if "shape" not in junction_dict.keys():
        junction_dict["shape"] = traci.junction.getShape(junctionID)
    if "position" not in junction_dict.keys():
        junction_dict["position"] = traci.junction.getPosition(junctionID)
    junction_dict["type"] = "junction"
    num_lane_from_edge = traci.edge.getLaneNumber(from_edge)
    laneID_list_from_edge = ["{}_{}".format(from_edge, i) for i in range(num_lane_from_edge)]

    num_lane_to_edge = traci.edge.getLaneNumber(to_edge)
    laneID_list_to_edge = ["{}_{}".format(to_edge, i) for i in range(num_lane_to_edge)]

    for from_lane in laneID_list_from_edge:
        if filting_indicator(from_lane) != "vehicle":
            continue
        succ_nodes = traci.lane.getLinks(from_lane)
        from_lane_shape = traci.lane.getShape(from_lane)
        for n in succ_nodes:
            if n[0] in laneID_list_to_edge:
                # print(filting_indicator(n[0]), filting_indicator(from_lane))
                lane_dict = {}
                lane_dict["from_edge"] = from_edge
                lane_dict["to_edge"] = to_edge
                lane_dict["from_lane"] = from_lane
                lane_dict["to_lane"] = n[0]

                to_lane_shape = traci.lane.getShape(n[0])
                meter_pointnum_ratio = 30
                start_point = from_lane_shape[-1]
                end_point = to_lane_shape[0]
                bez_ext = (
                    np.sqrt(np.square(start_point[0] - end_point[0]) + np.square(start_point[1] - end_point[1])) / 3
                )

                def extened_certain_length(stp, enp, ext):
                    xx = enp[0] - stp[0]
                    yy = enp[1] - stp[1]

                    exx = enp[0] + xx / np.sqrt(xx * xx + yy * yy) * ext
                    eyy = enp[1] + yy / np.sqrt(xx * xx + yy * yy) * ext
                    return exx, eyy

                control_point1 = start_point
                control_point2 = extened_certain_length(from_lane_shape[0], from_lane_shape[1], bez_ext)
                control_point3 = extened_certain_length(to_lane_shape[1], to_lane_shape[0], bez_ext)
                control_point4 = end_point

                lane_dict["control_point1"] = control_point1
                lane_dict["control_point2"] = control_point2
                lane_dict["control_point3"] = control_point3
                lane_dict["control_point4"] = control_point4

                # if filting_indicator(from_lane) == 'vehicle':
                if "lane_list" in junction_dict.keys():
                    if "{}_{}".format(from_lane, n[0]) not in junction_dict["lane_list"]:
                        junction_dict["lane_list"].append("{}_{}".format(from_lane, n[0]))
                else:
                    junction_dict["lane_list"] = ["{}_{}".format(from_lane, n[0])]

                junction_dict["{}_{}".format(from_lane, n[0])] = lane_dict
    return junction_dict


def get_junction_connection(route_juunction):
    connection_dict = {}
    connection_dict_b = {}
    connection_dict_p = {}
    for edge, data in route_juunction.items():
        junction_ID, turn_dir, junction_type, in_out_edge = data
        dl, du, dr, rd, rl, ru, ur, ud, ul, lu, lr, ld = [], [], [], [], [], [], [], [], [], [], [], []
        dd, rr, uu, ll = [], [], [], []

        dl_b, du_b, dr_b, rd_b, rl_b, ru_b, ur_b, ud_b, ul_b, lu_b, lr_b, ld_b = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        dd_b, rr_b, uu_b, ll_b = [], [], [], []
        dl_p, du_p, dr_p, rd_p, rl_p, ru_p, ur_p, ud_p, ul_p, lu_p, lr_p, ld_p = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        dd_p, rr_p, uu_p, ll_p = [], [], [], []

        def deal_with_in_out(x):
            y = x["in"]
            if isinstance(y, str):
                y = [y]
            elif isinstance(y, list):
                y = [y[0]]
            elif y == None:
                y = []
            else:
                raise ValueError
            z = x["out"]
            if isinstance(z, str):
                z = [z]
            elif isinstance(z, list):
                z = [z[-1]]
            elif z == None:
                z = []
            else:
                raise ValueError
            return y, z

        d_in, d_out = deal_with_in_out(in_out_edge["d"])
        u_in, u_out = deal_with_in_out(in_out_edge["u"])
        l_in, l_out = deal_with_in_out(in_out_edge["l"])
        r_in, r_out = deal_with_in_out(in_out_edge["r"])

        all_in = d_in + l_in + u_in + r_in
        all_out = d_out + l_out + u_out + r_out

        all_in_lane = []
        all_out_lane = []
        for inedge in all_in:
            all_in_lane += get_lane_of_edge(inedge)
        for outedge in all_out:
            all_out_lane += get_lane_of_edge(outedge)

        pathList, stack = [], [(e, []) for e in all_in_lane]

        while stack:
            cur_node, path_sub = stack.pop(0)
            succ_nodes = traci.lane.getLinks(cur_node)
            for n in succ_nodes:
                if not succ_nodes or cur_node in all_out_lane or len(path_sub) > 20:
                    if cur_node not in path_sub:
                        new_path = path_sub + [cur_node]
                        if new_path not in pathList:
                            pathList.append(new_path)
                else:
                    if n[4]:
                        if n[4] not in path_sub:
                            stack.append((n[4], path_sub + [cur_node]))
                    else:
                        if n[0] not in path_sub:
                            stack.append((n[0], path_sub + [cur_node]))
        # 分类 轨迹

        for p1 in pathList:

            start_edge = traci.lane.getEdgeID(p1[0])
            end_edge = traci.lane.getEdgeID(p1[-1])
            if start_edge in d_in:
                start_direction = "d"
            elif start_edge in u_in:
                start_direction = "u"
            elif start_edge in l_in:
                start_direction = "l"
            elif start_edge in r_in:
                start_direction = "r"
            else:
                raise ValueError

            if end_edge in d_out:
                end_direction = "d"
            elif end_edge in u_out:
                end_direction = "u"
            elif end_edge in l_out:
                end_direction = "l"
            elif end_edge in r_out:
                end_direction = "r"
            else:
                end_direction = None
            # 生成轨迹
            if end_direction is not None:
                if filting_indicator(p1[0]) == "vehicle":
                    eval(start_direction + end_direction).append(p1)
                elif filting_indicator(p1[0]) == "bicycle":
                    eval(start_direction + end_direction + "_b").append(p1)
                elif filting_indicator(p1[0]) == "pedestrian":
                    eval(start_direction + end_direction + "_p").append(p1)
                else:
                    pass
            # if end_direction is not None:
            #     eval(start_direction + end_direction).extend(p1)

        # dl = list(set(dl))
        # du = list(set(du))
        # dr = list(set(dr))
        # rd = list(set(rd))
        # rl = list(set(rl))
        # ru = list(set(ru))
        # ur = list(set(ur))
        # ud = list(set(ud))
        # ul = list(set(ul))
        # lu = list(set(lu))
        # lr = list(set(lr))
        # ld = list(set(ld))
        def cat_list(l):
            ln = []
            for x in l:
                ln.extend(x)
            return ln

        dl_pad = None
        du_pad = None
        dr_pad = None
        rd_pad = None
        rl_pad = None
        ru_pad = None
        ur_pad = None
        ud_pad = None
        ul_pad = None
        lu_pad = None
        lr_pad = None
        ld_pad = None

        iter_list = [dl, du, dr, rd, rl, ru, ur, ud, ul, lu, lr, ld]
        pad_list = []

        def set_pad_position(sp, ep):
            x1, x2 = sp
            y1, y2 = ep

            dir_vec = x1 - y1, x2 - y2

            dir_vec_len = np.sqrt(np.square(dir_vec[0]) + np.square(dir_vec[1]))

            xn = y1 + dir_vec[0] / dir_vec_len * 35
            yn = y2 + dir_vec[1] / dir_vec_len * 35

            return xn, yn

        for rout in iter_list:
            if rout:

                pp = rout[-1]
                pp_end_lane = pp[0]
                pp_end_lane_shape = traci.lane.getShape(pp_end_lane)

                sp = pp_end_lane_shape[0]
                ep = pp_end_lane_shape[1]

                xpad, y_pad = set_pad_position(sp, ep)

                pp_phi = np.arctan2(ep[1] - sp[1], ep[0] - sp[0]) * 180 / np.pi

                pad_list.append(dict(x=xpad, y=y_pad, phi=pp_phi))
            else:
                pad_list.append(None)

        dl_pad = pad_list[0]
        du_pad = pad_list[1]
        dr_pad = pad_list[2]
        rd_pad = pad_list[3]
        rl_pad = pad_list[4]
        ru_pad = pad_list[5]
        ur_pad = pad_list[6]
        ud_pad = pad_list[7]
        ul_pad = pad_list[8]
        lu_pad = pad_list[9]
        lr_pad = pad_list[10]
        ld_pad = pad_list[11]

        connection_dict[edge] = dict(
            dl={"path": dl, "lanes": list(set(cat_list(dl))), "pad": dl_pad},
            du={"path": du, "lanes": list(set(cat_list(du))), "pad": du_pad},
            dr={"path": dr, "lanes": list(set(cat_list(dr))), "pad": dr_pad},
            rd={"path": rd, "lanes": list(set(cat_list(rd))), "pad": rd_pad},
            rl={"path": rl, "lanes": list(set(cat_list(rl))), "pad": rl_pad},
            ru={"path": ru, "lanes": list(set(cat_list(ru))), "pad": ru_pad},
            ur={"path": ur, "lanes": list(set(cat_list(ur))), "pad": ur_pad},
            ud={"path": ud, "lanes": list(set(cat_list(ud))), "pad": ud_pad},
            ul={"path": ul, "lanes": list(set(cat_list(ul))), "pad": ul_pad},
            lu={"path": lu, "lanes": list(set(cat_list(lu))), "pad": lu_pad},
            lr={"path": lr, "lanes": list(set(cat_list(lr))), "pad": lr_pad},
            ld={"path": ld, "lanes": list(set(cat_list(ld))), "pad": ld_pad},
        )  # 不考虑掉头

        du_b_pad = None
        dr_b_pad = None
        rl_b_pad = None
        ru_b_pad = None
        ud_b_pad = None
        ul_b_pad = None
        lr_b_pad = None
        ld_b_pad = None

        if du_b:
            pp = du_b[0]
            pp_start_lane = pp[0]
            pp_start_lane_shape = traci.lane.getShape(pp_start_lane)

            sp = pp_start_lane_shape[0]
            ep = pp_start_lane_shape[1]
            xpad, y_pad = set_pad_position(sp, ep)
            pp_phi = np.arctan2(ep[1] - sp[1], ep[0] - sp[0]) * 180 / np.pi

            du_b_pad = dict(x=xpad, y=y_pad, phi=pp_phi)

        if ud_b:
            pp = ud_b[0]
            pp_start_lane = pp[0]
            pp_start_lane_shape = traci.lane.getShape(pp_start_lane)

            sp = pp_start_lane_shape[0]
            ep = pp_start_lane_shape[1]
            xpad, y_pad = set_pad_position(sp, ep)
            pp_phi = np.arctan2(ep[1] - sp[1], ep[0] - sp[0]) * 180 / np.pi

            ud_b_pad = dict(x=xpad, y=y_pad, phi=pp_phi)

        if lr_b:
            pp = lr_b[0]
            pp_start_lane = pp[0]
            pp_start_lane_shape = traci.lane.getShape(pp_start_lane)

            sp = pp_start_lane_shape[0]
            ep = pp_start_lane_shape[1]
            xpad, y_pad = set_pad_position(sp, ep)
            pp_phi = np.arctan2(ep[1] - sp[1], ep[0] - sp[0]) * 180 / np.pi

            lr_b_pad = dict(x=xpad, y=y_pad, phi=pp_phi)

        connection_dict_b[edge] = dict(
            du_b={"path": du_b, "lanes": list(set(cat_list(du_b))), "pad": du_b_pad},
            dr_b={"path": dr_b, "lanes": list(set(cat_list(dr_b))), "pad": dr_b_pad},
            rl_b={"path": rl_b, "lanes": list(set(cat_list(rl_b))), "pad": rl_b_pad},
            ru_b={"path": ru_b, "lanes": list(set(cat_list(ru_b))), "pad": ru_b_pad},
            ud_b={"path": ud_b, "lanes": list(set(cat_list(ud_b))), "pad": ud_b_pad},
            ul_b={"path": ul_b, "lanes": list(set(cat_list(ul_b))), "pad": ul_b_pad},
            lr_b={"path": lr_b, "lanes": list(set(cat_list(lr_b))), "pad": lr_b_pad},
            ld_b={"path": ld_b, "lanes": list(set(cat_list(ld_b))), "pad": ld_b_pad},
        )  # 不考虑掉头, 左转

        du_p_pad = None
        rl_p_pad = None
        ud_p_pad = None
        lr_p_pad = None
        if du_p:
            pp = du_p[0]
            pp_start_lane = pp[0]
            pp_start_lane_shape = traci.lane.getShape(pp_start_lane)

            sp = pp_start_lane_shape[0]
            ep = pp_start_lane_shape[1]
            xpad, y_pad = set_pad_position(sp, ep)
            pp_phi = np.arctan2(ep[1] - sp[1], ep[0] - sp[0]) * 180 / np.pi

            du_p_pad = dict(x=xpad, y=y_pad, phi=pp_phi)
        if rl_p:
            pp = rl_p[0]
            pp_start_lane = pp[0]
            pp_start_lane_shape = traci.lane.getShape(pp_start_lane)

            sp = pp_start_lane_shape[0]
            ep = pp_start_lane_shape[1]
            xpad, y_pad = set_pad_position(sp, ep)
            pp_phi = np.arctan2(ep[1] - sp[1], ep[0] - sp[0]) * 180 / np.pi

            rl_p_pad = dict(x=xpad, y=y_pad, phi=pp_phi)

        if ud_p:
            pp = ud_p[0]
            pp_start_lane = pp[0]
            pp_start_lane_shape = traci.lane.getShape(pp_start_lane)

            sp = pp_start_lane_shape[0]
            ep = pp_start_lane_shape[1]
            xpad, y_pad = set_pad_position(sp, ep)
            pp_phi = np.arctan2(ep[1] - sp[1], ep[0] - sp[0]) * 180 / np.pi

            ud_p_pad = dict(x=xpad, y=y_pad, phi=pp_phi)

        if lr_p:
            pp = lr_p[0]
            pp_start_lane = pp[0]
            pp_start_lane_shape = traci.lane.getShape(pp_start_lane)

            sp = pp_start_lane_shape[0]
            ep = pp_start_lane_shape[1]
            xpad, y_pad = set_pad_position(sp, ep)
            pp_phi = np.arctan2(ep[1] - sp[1], ep[0] - sp[0]) * 180 / np.pi

            lr_p_pad = dict(x=xpad, y=y_pad, phi=pp_phi)
        connection_dict_p[edge] = dict(
            du_p={"path": du_p, "lanes": list(set(cat_list(du_p))), "pad": du_p_pad},
            rl_p={"path": rl_p, "lanes": list(set(cat_list(rl_p))), "pad": rl_p_pad},
            ud_p={"path": ud_p, "lanes": list(set(cat_list(ud_p))), "pad": ud_p_pad},
            lr_p={"path": lr_p, "lanes": list(set(cat_list(lr_p))), "pad": lr_p_pad},
        )  # 不考虑掉头， 左转右转

    return connection_dict, connection_dict_p, connection_dict_b
    # with open(os.path.join('map_info', "turning_v.json"), 'wt') as f:
    #     json.dump(connection_dict, f, indent=2)
    # with open(os.path.join('map_info', "turning_p.json"), 'wt') as f:
    #     json.dump(connection_dict_p, f, indent=2)
    # with open(os.path.join('map_info', "turning_b.json"), 'wt') as f:
    #     json.dump(connection_dict_b, f, indent=2)
    # print(connection_dict)
    # print(len(connection_dict['EE6']))
    # print(connection_dict['EE6'][1] == connection_dict['EE6'][2])


def get_ref_lanes(CROSS_TASKS, CROSS2_TASKS={}):

    total_dict = {**CROSS_TASKS, **CROSS2_TASKS}
    task2subroute = {}
    for key, value in total_dict.items():
        sub_route = value["sub_route"]
        from_edge = sub_route[0]
        to_edge = sub_route[-1]

        from_lane = get_lane_of_edge(from_edge)
        to_lane = get_lane_of_edge(to_edge)

        pathList, stack = [], [(e, []) for e in from_lane]

        while stack:
            cur_node, path_sub = stack.pop(0)
            succ_nodes = traci.lane.getLinks(cur_node)
            for n in succ_nodes:
                if not succ_nodes or cur_node in to_lane or len(path_sub) > 20:
                    if cur_node not in path_sub:
                        if cur_node in to_lane:
                            new_path = path_sub + [cur_node]
                            if new_path not in pathList and filting_indicator(new_path[0]) == "vehicle":
                                pathList.append(new_path)
                else:
                    if n[4]:
                        if n[4] not in path_sub:
                            stack.append((n[4], path_sub + [cur_node]))
                    else:
                        if n[0] not in path_sub:
                            stack.append((n[0], path_sub + [cur_node]))

        # clean inner lanes
        def del_inner(lane_list):
            return [ll for ll in lane_list if ll[0] != ":"]

        def add_edge(lane_list):
            return [(traci.lane.getEdgeID(ll), ll) for ll in lane_list]

        pathList_del = [del_inner(path1) for path1 in pathList]

        pathList_del2 = [add_edge(ll) for ll in pathList_del]
        task2subroute[key] = pathList_del2
    return task2subroute


def main_lane_shape(ego_route, route_junction):
    """
    只生成‘车’道的信息
    :param ego_route:
    :param route_junction:
    :return:
    """
    total_dict = {}
    for i, edge in list(enumerate(ego_route))[:-1]:
        edge_dict = get_edge_path(edge)
        total_dict[edge] = edge_dict
        junction_ID = route_junction[edge][0]
        if junction_ID in total_dict.keys():
            jd = total_dict[junction_ID]
        else:
            jd = {}
        junction_dict = get_junction_path(route_junction[edge][0], edge, ego_route[i + 1], jd)
        total_dict[junction_ID] = junction_dict

    for l, io in route_junction.items():
        edge1 = io[3]["u"]["in"]
        edge2 = io[3]["d"]["in"]
        edge3 = io[3]["l"]["in"]
        edge4 = io[3]["r"]["in"]
        edge5 = io[3]["u"]["out"]
        edge6 = io[3]["d"]["out"]
        edge7 = io[3]["l"]["out"]
        edge8 = io[3]["r"]["out"]
        edge_list = [edge1, edge2, edge3, edge4, edge5, edge6, edge7, edge8]
        for e in edge_list:
            if not isinstance(e, list):
                el = [e]
            else:
                el = e
            for ee in el:
                if (ee not in total_dict.keys()) and (ee is not None):
                    edge_dict = get_edge_path(ee)
                    total_dict[ee] = edge_dict

    return total_dict
    # # save results
    # save_path = './map_info'
    # if os.path.exists(save_path):
    #     shutil.copytree(save_path, save_path + '.backup', dirs_exist_ok=True)
    #     shutil.rmtree(save_path)
    #
    # os.makedirs(save_path, exist_ok=True)
    #
    # for key in total_dict.keys():
    #     with open(os.path.join(save_path, "{}.json".format(key)), 'wt') as f:
    #         json.dump(total_dict[key], f, indent=2)


def main():
    from misc_ic import EGO_ROUTE, ROUTE_JUNCTION, CROSS_TASKS

    start_sumo()
    static_path_shape_dict = main_lane_shape(EGO_ROUTE, ROUTE_JUNCTION)
    connection_dict, connection_dict_p, connection_dict_b = get_junction_connection(ROUTE_JUNCTION)
    task2subroute = get_ref_lanes(CROSS_TASKS)
    traci.close()

    # save results
    save_path = os.path.dirname(os.path.abspath(__file__)) + "/map_info"
    print(save_path)
    if os.path.exists(save_path):
        # shutil.copytree(save_path, save_path + '.backup', dirs_exist_ok=True)
        shutil.rmtree(save_path)

    os.makedirs(save_path, exist_ok=True)

    save_json(os.path.join(save_path, "static_path_shape.json"), static_path_shape_dict)
    save_json(os.path.join(save_path, "turning_v.json"), connection_dict)
    save_json(os.path.join(save_path, "turning_p.json"), connection_dict_p)
    save_json(os.path.join(save_path, "turning_b.json"), connection_dict_b)
    save_json(os.path.join(save_path, "static_path_list.json"), task2subroute)


def te():
    start_sumo()
    print(traci.lane.getShape("EE6_0"))
    print(traci.lane.getShape(":JC6_w2_0"))
    print(traci.lane.getShape(":JC6_c1_0"))
    print(traci.lane.getShape(":JC6_w1_0"))
    print(traci.lane.getShape(":JC6_c0_0"))
    print(traci.lane.getShape(":JC6_w0_0"))
    print(traci.lane.getShape("EN7_0"))

    traci.close()


"""
def get_junction_path2(from_edge, to_edge):
    num_lane_from_edge = traci.edge.getLaneNumber(from_edge)
    laneID_list_from_edge = ['{}_{}'.format(from_edge, i) for i in range(num_lane_from_edge)]

    num_lane_to_edge = traci.edge.getLaneNumber(to_edge)
    laneID_list_to_edge = ['{}_{}'.format(to_edge, i) for i in range(num_lane_to_edge)]

    connected_path_shape_dict = {}

    def generate_beizer():
        meter_pointnum_ratio = 30
        start_point = from_lane_shape[-1]
        end_point = to_lane_shape[0]
        bez_ext = np.sqrt(np.square(start_point[0] - end_point[0]) + np.square(start_point[1] - end_point[1])) / 3
        def extened_certain_length(stp, enp, ext):
            xx = enp[0] - stp[0]
            yy = enp[1] - stp[1]

            exx = enp[0] + xx / np.sqrt(xx * xx + yy * yy) * ext
            eyy = enp[1] + yy / np.sqrt(xx * xx + yy * yy) * ext
            return exx, eyy

        control_point1 = start_point
        control_point2 = extened_certain_length(from_lane_shape[0], from_lane_shape[1], bez_ext)
        control_point3 = extened_certain_length(to_lane_shape[1], to_lane_shape[0], bez_ext)
        control_point4 = end_point
        bez_node = np.asfortranarray([[control_point1[0], control_point2[0], control_point3[0], control_point4[0]],
                                  [control_point1[1], control_point2[1], control_point3[1], control_point4[1]]],
                                 dtype=np.float32)
        curve = bezier.Curve(bez_node, degree=3)
        s_vals = np.linspace(0, 1, int(bez_ext * 3 * 3.1415926 / 2 * meter_pointnum_ratio))
        trj_data = curve.evaluate_multi(s_vals)
        trj_data = trj_data.astype(np.float32)
        trj_num = trj_data.shape[1]
        return trj_data[0], trj_data[1], trj_num

    for from_lane in laneID_list_from_edge:
        succ_nodes = traci.lane.getLinks(from_lane)
        from_lane_shape = traci.lane.getShape(from_lane)
        for n in succ_nodes:
            if n[0] in laneID_list_to_edge:
                to_lane_shape = traci.lane.getShape(n[0])
                trjx, trjy, trjnum = generate_beizer()
                connected_path_shape_dict[from_lane] = (from_edge, n[0], from_lane_shape, to_lane_shape,
                                                        trjx, trjy, trjnum)
        # s_vals = np.linspace(0, 1.0, int(pi / 2 * (CROSSROAD_SIZE / 2 + LANE_WIDTH / 2)) * meter_pointnum_ratio)
    return connected_path_shape_dict
"""

# def test_get_edge_path():
#     # from pprint import pprint
#     # start_sumo()
#     # edge_dict = get_edge_path('EE1')
#     # pprint(edge_dict)
#     # edge_dict = get_junction_path('JC6', 'EE6', 'ES8', {})
#     # pprint(edge_dict)
#
#     # main()
#     import shutil
#     shutil.copytree('./map', './map.backup')

if __name__ == "__main__":
    main()
