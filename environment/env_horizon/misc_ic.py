import os
import math
from collections import OrderedDict
import json

L, W = 4.8, 2.0
L_VEHICLE, W_VEHICLE = 4.8, 2.0
L_BIKE, W_BIKE = 2.0, 0.48
GOAL_POINT_R = 3.75

EPSILON_V = 0.01
DETECT_RANGE = 40

RANDOM_V_X = 1.0
RANDOM_V_Y = 0.5
RANDOM_R = 0.1 * math.pi / 180
RANDOM_X = 0.7
RANDOM_Y = 0.7
RANDOM_PHI = 20 * math.pi / 180

ML_BIKE_NUM = 5
ML_PERSON_NUM = 5
ML_VEH_NUM = 5

dirname = os.path.dirname(__file__)
SUMOCFG_DIR = dirname + "/sumo_file/cfg/IC_whole.sumocfg"

# worker_id2env_edge = {0: "EE6",
#                       1: "EE6", 2: "EE6", 3: "EE6", 4: "EE6", 5: "EE6", 6: "EE6",
#                       7: "EE6", 8: "EE6", 9: "EE6", 10: "EE6", 11: "EE6", 12: "EE6"}
worker_id2env_edge = {0: "EE6",
                      1: "EE6", 2: "ES8", 3: "ES9", 4: "EE10", 5: "EN15", 6: "EN14",
                      7: "EN13", 8: "EW8", 9: "EW7", 10: "EW6", 11: "EW5", 12: "EW4",
                      13: "EN4", 14: "EE2", 15: "ES7", 16: "EE2"}

EGO_ROUTE = [
    "EE6",
    "ES8",
    "ES9",
    "EE10",
    "EE11",
    "EN15",
    "EN14",
    "EN13",
    "EN12",
    "EW8",
    "EW7",
    "EW6",
    "EW5",
    "EW4",
    "EN4",
    "EN3",
    "EE2",
    "ES7",
    "EW6",
]

CROSS_TASKS = dict(
    EE5={
            "from_edge": "EE5",
            "to_edge": "EE6",
            "sub_route": ["EE5", "EE6"],
            "main_crossing": "JC4",
            "tl": [0, 3],
        },
    EE6={
        "from_edge": "EE6",
        "to_edge": "ES8",
        "sub_route": ["EE6", "ES8"],
        "main_crossing": "JC6",
        "tl": [0, 1, 3, 4, 6, 7, 9, 10],
    },
    ES8={
            "from_edge": "ES8",
            "to_edge": "ES9",
            "sub_route": ["ES8", "ES9"],
            "main_crossing": "JC7",
            "tl": [0],
        },
    ES9={
        "from_edge": "ES9",
        "to_edge": "EE10",
        "sub_route": ["ES9", "EE10"],
        "main_crossing": "JC8",
        "tl": [0, 1],
    },
    EE10={
        "from_edge": "EE10",
        "to_edge": "EN15",
        "sub_route": ["EE10", "EE11", "EN15"],
        "main_crossing": "JC13",
        "tl": [4],
    },
    EN13={
        "from_edge": "EN13",
        "to_edge": "EW8",
        "sub_route": ["EN13", "EN12", "EW8"],
        "main_crossing": "JC10",
        "tl": [2],
    },
    EW8={
        "from_edge": "EW8",
        "to_edge": "EW7",
        "sub_route": ["EW8", "EW7"],
        "main_crossing": "JC9",
        "tl": [0, 1],
    },
    EW7={
        "from_edge": "EW7",
        "to_edge": "EW6",
        "sub_route": ["EW7", "EW6"],
        "main_crossing": "JC6",
        "tl": [3, 4],
    },
    EW6={
        "from_edge": "EW6",
        "to_edge": "EW5",
        "sub_route": ["EW6", "EW5"],
        "main_crossing": "JC4",
        "tl": [0],
    },
    EW5={
        "from_edge": "EW5",
        "to_edge": "EW4",
        "sub_route": ["EW5", "EW4"],
        "main_crossing": "JC3",
        "tl": [0],
    },
    EW4={
        "from_edge": "EW4",
        "to_edge": "EN4",
        "sub_route": ["EW4", "EN4"],
        "main_crossing": "JC2",
        "tl": [0],
    },
    EN4={
        "from_edge": "EN4",
        "to_edge": "EE2",
        "sub_route": ["EN4", "EN3", "EE2"],
        "main_crossing": "JC1",
        "tl": [0, 1, 2, 3, 4, 5, 6, 7],
    },
    EE2={
        "from_edge": "EE2",
        "to_edge": "ES7",
        "sub_route": ["EE2", "ES7"],
        "main_crossing": "JC5",
        "tl": [0, 1, 2, 3, 4, 5, 6, 7],
    },
    ES7={
        "from_edge": "ES7",
        "to_edge": "EW6",
        "sub_route": ["ES7", "EW6"],
        "main_crossing": "JC6",
        "tl": [0, 1],
    },
    EN15={
        "from_edge": "EN15",
        "to_edge": "EN14",
        "sub_route": ["EN15", "EN14"],
        "main_crossing": "JC12",
        "tl": [0],
    },
    EN14={
        "from_edge": "EN14",
        "to_edge": "EN13",
        "sub_route": ["EN14", "EN13"],
        "main_crossing": "JC11",
        "tl": [0],
    },
)


MLANE_TASKS = dict(
    EE6={"lane_list": ["EE6_2", "EE6_3"]},
    ES8={"lane_list": ["ES8_2", "ES8_3"]},
    ES9={"lane_list": ["ES9_2", "ES9_3"]},
    EE10={"lane_list": ["EE10_2", "EE10_3"]},
    EE11={"lane_list": ["EE11_2", "EE11_3", "EE11_4"]},
    EN15={"lane_list": ["EN15_2", "EN15_3"]},
    EN14={"lane_list": ["EN14_2", "EN14_3"]},
    EN13={"lane_list": ["EN13_2", "EN13_3"]},
    EN12={"lane_list": ["EN12_2", "EN12_3", "EN12_4"]},
    EW8={"lane_list": ["EW8_2", "EW8_3"]},
    EW7={"lane_list": ["EW7_2", "EW7_3"]},
    EW6={"lane_list": ["EW6_2", "EW6_3"]},
    EW5={"lane_list": ["EW5_2", "EW5_3"]},
    EW4={"lane_list": ["EW4_2", "EW4_3"]},
    EN4={"lane_list": ["EN4_2", "EN4_3"]},
    EN3={"lane_list": ["EN3_2"]},
    EE2={"lane_list": ["EE2_3", "EE2_4"]},
    ES7={"lane_list": ["ES7_2", "ES7_3"]},
)

ROUTE_JUNCTION = {
    "EE6": (
        "JC6",
        "right",
        "crossing",
        dict(
            u={"in": "EW7", "out": "EE7"},
            d={"in": "EE6", "out": "EW6"},
            l={"in": "ES7", "out": "EN7"},
            r={"in": "EN8", "out": "ES8"},
        ),
    ),
    "ES8": (
        "JC7",
        "straight",
        "crosswalk",
        dict(
            u={"in": "EN9", "out": "ES9"},
            d={"in": "ES8", "out": "EN8"},
            l={"in": "EP3", "out": "EP3"},
            r={"in": "EP2", "out": "EP2"},
        ),
    ),
    "ES9": (
        "JC8",
        "left",
        "crossing",
        dict(
            u={"in": "EN10", "out": "ES10"},
            d={"in": "ES9", "out": "EN9"},
            l={"in": "EW10", "out": "EE10"},
            r={"in": "EE9", "out": "EW9"},
        ),
    ),
    "EE10": (
        "JJ4",
        "straight",
        "joint",
        dict(
            u={"in": "EW11", "out": "EE11"},
            d={"in": "EE10", "out": "EW10"},
            l={"in": None, "out": None},
            r={"in": None, "out": None},
        ),
    ),
    "EE11": (
        "JC13",
        "left",
        "T-crossing",
        dict(
            u={"in": None, "out": None},
            d={"in": ["EE10", "EE11"], "out": ["EW11", "EW10"]},
            l={"in": "ES15", "out": "EN15"},
            r={"in": "EN16", "out": "ES16"},
        ),
    ),
    "EN15": (
        "JC12",
        "straight",
        "crosswalk",
        dict(
            u={"in": "ES14", "out": "EN14"},
            d={"in": "EN15", "out": "ES15"},
            l={"in": "EP6", "out": "EP6"},
            r={"in": "EP7", "out": "EP7"},
        ),
    ),
    "EN14": (
        "JC11",
        "straight",
        "crosswalk",
        dict(
            u={"in": "ES13", "out": "EN13"},
            d={"in": "EN14", "out": "ES14"},
            l={"in": "EP4", "out": "EP4"},
            r={"in": "EP5", "out": "EP5"},
        ),
    ),
    "EN13": (
        "JJ3",
        "straight",
        "joint",
        dict(
            u={"in": "ES12", "out": "EN12"},
            d={"in": "EN13", "out": "ES13"},
            l={"in": None, "out": None},
            r={"in": None, "out": None},
        ),
    ),
    "EN12": (
        "JC10",
        "left",
        "T-crossing",
        dict(
            u={"in": "ES11", "out": "EN11"},
            d={"in": ["EN13", "EN12"], "out": ["ES12", "ES13"]},
            l={"in": "EE8", "out": "EW8"},
            r={"in": None, "out": None},
        ),
    ),
    "EW8": (
        "JC9",
        "straight",
        "crossing",
        dict(
            u={"in": "EE7", "out": "EW7"},
            d={"in": "EW8", "out": "EE8"},
            l={"in": "-EO4", "out": "EO4"},
            r={"in": "EO3", "out": "-EO3"},
        ),
    ),
    "EW7": (
        "JC6",
        "straight",
        "crossing",
        dict(
            u={"in": "EE6", "out": "EW6"},
            d={"in": "EW7", "out": "EE7"},
            l={"in": "EN8", "out": "ES8"},
            r={"in": "ES7", "out": "EN7"},
        ),
    ),
    "EW6": (
        "JC4",
        "straight",
        "T-crossing",
        dict(
            u={"in": "EE5", "out": "EW5"},
            d={"in": "EW6", "out": "EE6"},
            l={"in": "-EO2", "out": "EO2"},
            r={"in": None, "out": None},
        ),
    ),
    "EW5": (
        "JC3",
        "straight",
        "crossing",
        dict(
            u={"in": "EE4", "out": "EW4"},
            d={"in": "EW5", "out": "EE5"},
            l={"in": "EP1", "out": "EP1"},
            r={"in": "EO1", "out": "-EO1"},
        ),
    ),
    "EW4": (
        "JC2",
        "right",
        "T-crossing",
        dict(
            u={"in": None, "out": None},
            d={"in": "EW4", "out": "EE4"},
            l={"in": "EN5", "out": "ES5"},
            r={"in": "ES4", "out": "EN4"},
        ),
    ),
    "EN4": (
        "JJ2",
        "straight",
        "joint",
        dict(
            u={"in": "ES3", "out": "EN3"},
            d={"in": "EN4", "out": "ES4"},
            l={"in": None, "out": None},
            r={"in": None, "out": None},
        ),
    ),
    "EN3": (
        "JC1",
        "right",
        "crossing",
        dict(
            u={"in": ["ES1", "ES2"], "out": ["EN1", "EN2"]},
            d={"in": ["EN4", "EN3"], "out": ["ES3", "ES4"]},
            l={"in": "EE1", "out": "EW1"},
            r={"in": "EW2", "out": "EE1"},
        ),
    ),
    "EE2": (
        "JC5",
        "right",
        "crossing",
        dict(
            u={"in": "EW3", "out": "EE3"},
            d={"in": "EE2", "out": "EW2"},
            l={"in": "ES6", "out": "EN6"},
            r={"in": "EN7", "out": "ES7"},
        ),
    ),
    "ES7": (
        "JC6",
        "right",
        "crossing",
        dict(
            u={"in": "EN8", "out": "ES8"},
            d={"in": "ES7", "out": "EN7"},
            l={"in": "EW7", "out": "EE7"},
            r={"in": "EE6", "out": "EW6"},
        ),
    ),
}


def load_json(path):
    with open(path, "r", encoding="UTF-8") as f:
        value = json.load(f)
    return value


try:
    map_info_folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "map_info")
    map_info = load_json(os.path.join(map_info_folder_path, "static_path_shape.json"))
    turning_v_info = load_json(os.path.join(map_info_folder_path, "turning_v.json"))
    turning_b_info = load_json(os.path.join(map_info_folder_path, "turning_b.json"))
    turning_p_info = load_json(os.path.join(map_info_folder_path, "turning_p.json"))
    task2staticpath = load_json(os.path.join(map_info_folder_path, "static_path_list.json"))
except:
    print("Fails to load map info")

VEHICLE_MODE_DICT = dict(
    left=OrderedDict(dl=2, du=2, ud=2, ul=2),
    straight=OrderedDict(dl=1, du=2, dr=1, ru=2, ur=2),
    right=OrderedDict(dr=1, ur=2, lr=2, du=2),
)
BIKE_MODE_DICT = dict(
    left=OrderedDict(ud_b=2),
    straight=OrderedDict(du_b=4),
    right=OrderedDict(du_b=2, lr_b=2, dr_b=1),
)
PERSON_MODE_DICT = dict(
    left=OrderedDict(ud_p=4),
    straight=OrderedDict(lr_p=1),  # TODO 0 leads to bugs
    right=OrderedDict(du_p=4, lr_p=0),
)


MODE2TASK = {
    "dr": "right",
    "du": "straight",
    "dl": "left",
    "rd": "left",
    "ru": "right",
    "rl": " straight",
    "ud": "straight",
    "ur": "left",
    "ul": "right",
    "ld": "right",
    "lr": "straight",
    "lu": "left",
    "ud_b": "straight",
    "du_b": "straight",
    "lr_b": "straight",
    "dr_b": "right",
    "ud_p": "straight",
    "du_p": "straight",
    "lr_p": "straight",
    "rd_p": "straight",
}


def dict2flat(inp):
    out = []
    for key, val in inp.items():
        out.extend([key] * val)
    return out


def dict2num(inp):
    out = 0
    for _, val in inp.items():
        out += val
    return out


VEH_NUM = dict(
    left=dict2num(VEHICLE_MODE_DICT["left"]),
    straight=dict2num(VEHICLE_MODE_DICT["straight"]),
    right=dict2num(VEHICLE_MODE_DICT["right"]),
)

BIKE_NUM = dict(
    left=dict2num(BIKE_MODE_DICT["left"]),
    straight=dict2num(BIKE_MODE_DICT["straight"]),
    right=dict2num(BIKE_MODE_DICT["right"]),
)

PERSON_NUM = dict(
    left=dict2num(PERSON_MODE_DICT["left"]),
    straight=dict2num(PERSON_MODE_DICT["straight"]),
    right=dict2num(PERSON_MODE_DICT["right"]),
)

VEHICLE_MODE_LIST = dict(
    left=dict2flat(VEHICLE_MODE_DICT["left"]),
    straight=dict2flat(VEHICLE_MODE_DICT["straight"]),
    right=dict2flat(VEHICLE_MODE_DICT["right"]),
)
BIKE_MODE_LIST = dict(
    left=dict2flat(BIKE_MODE_DICT["left"]),
    straight=dict2flat(BIKE_MODE_DICT["straight"]),
    right=dict2flat(BIKE_MODE_DICT["right"]),
)
PERSON_MODE_LIST = dict(
    left=dict2flat(PERSON_MODE_DICT["left"]),
    straight=dict2flat(PERSON_MODE_DICT["straight"]),
    right=dict2flat(PERSON_MODE_DICT["right"]),
)


if __name__ == "__main__":
    from pprint import pprint

    print(turning_v_info)

    print(turning_b_info)

    print(turning_p_info)

    print(task2staticpath)

    print(task2staticpath["EN13"])
