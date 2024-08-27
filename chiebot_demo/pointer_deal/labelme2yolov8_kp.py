# -*- coding: utf-8 -*-
"""
@Author: captainfffsama
@Date: 2024-08-16 14:02:33
@LastEditors: captainfffsama tuanzhangsama@outlook.com
@LastEditTime: 2024-08-23 11:24:05
@FilePath: /ultralytics/chiebot_demo/pointer_deal/labelme2yolov8_kp.py
@Description:
"""

# -*- coding: utf-8 -*-
"""
@Author: captainfffsama
@Date: 2024-07-29 16:13:55
@LastEditors: captainfffsama tuanzhangsama@outlook.com
@LastEditTime: 2024-07-29 17:14:01
@FilePath: /detectron2/labelme2yolov8_kp.py
@Description:
translate labelme keypoint data to yolov8 format,
in labelme, key point label by line or plogon,
use invisible rect to arround invisible point.
yolov8 need one object have bbox,
each key point group all use a expand external rectangle as bbox,
expand value follow the DBNet.
if all key point on one line,use 0.1*line_length as expand value.
"""

from typing import List, Dict
import os
import json
from concurrent import futures

import numpy as np
from tqdm import tqdm
import yaml


def get_all_file_path(file_dir: str, filter_=(".jpg")) -> list:
    # 遍历文件夹下所有的file
    return [
        os.path.join(maindir, filename)
        for maindir, _, file_name_list in os.walk(file_dir)
        for filename in file_name_list
        if os.path.splitext(filename)[1] in filter_
    ]


def expand_rect(tl: np.ndarray, br: np.ndarray, h: int, w: int,r:float=0.8):
    erhw_rate = (br - tl).min() / (br - tl).max()
    if erhw_rate < 0.1:
        D = int(0.05 * (br - tl).max())
    else:
        A = (br - tl).min() * (br - tl).max()
        L = 2 * ((br - tl).min() + (br - tl).max())
        D = int(A * (1 - r**2) / L)

    tl = tl - D
    br = br + D
    tl[0] = np.clip(tl[0], 0, w)
    tl[1] = np.clip(tl[1], 0, h)
    br[0] = np.clip(br[0], 0, w)
    br[1] = np.clip(br[1], 0, h)
    return tl, br


def mark_invisible(kps, invisible_rect) -> np.ndarray:
    """_summary_

    Args:
        kps (_type_): Nx2
        invisible_rect (_type_): Mx2x2,M,tlbr,xy

    Returns:
        np.ndarray: _description_
    """
    if isinstance(invisible_rect, list):
        invisible_rect = np.array(invisible_rect)
    if invisible_rect.shape[0] == 0:
        return np.ones((kps.shape[0], 1), dtype=float) * 2.0
    t = kps[:, None, None, :] - invisible_rect[None, :, :, :]

    kps_in_each_rect = np.logical_and((t[:, :, 0] > 0).all(axis=-1), (t[:, :, 1] < 0).all(axis=-1))
    kps_is_invisible = np.logical_not(kps_in_each_rect.any(axis=-1))
    kps_is_invisible = kps_is_invisible + 1
    return kps_is_invisible


def deal_one(json_path, save_dir, class_map: dict):
    with open(json_path, "r") as fr:
        json_content = json.load(fr)

    objs_info = []
    h, w = json_content["imageHeight"], json_content["imageWidth"]
    invisible_rect = []
    for shape in json_content["shapes"]:
        if "invisible" == shape["label"]:
            invisible_rect.append(np.array(shape["points"]))
    for shape in json_content["shapes"]:
        # if shape['shape_type'] in ('polygon','line'):
        if shape["shape_type"] in ("line",):
            kps = np.array(shape["points"])
            kps[:, 0] = np.clip(kps[:, 0], 0, w)
            kps[:, 1] = np.clip(kps[:, 1], 0, h)
            kps_is_invisible = mark_invisible(kps, invisible_rect)
            kps_is_invisible = kps_is_invisible.astype(float)

            ertl = kps.min(axis=0)
            erbr = kps.max(axis=0)
            ertl, erbr = expand_rect(ertl, erbr, h, w)
            erwh: np.ndarray = erbr - ertl
            cxy: np.ndarray = ertl + erwh / 2
            erwh = erwh / np.array([w, h])
            cxy = cxy / np.array([w, h])
            if shape["label"] not in class_map:
                print(f"{json_path} label {shape['label']} not in class_map,may be a wrong label")
                continue
            label = class_map[shape["label"]]
            kps = kps / np.array([w, h])
            kps_info = np.concatenate([kps, kps_is_invisible], axis=1).flatten()

            all_info = cxy.tolist() + erwh.tolist() + kps_info.tolist()
            all_info = [str(i) for i in all_info]
            all_info = [str(label)] + all_info
            objs_info.append(" ".join(all_info) + "\n")

    save_path = os.path.join(save_dir, os.path.basename(json_path).replace(".json", ".txt"))
    with open(save_path, "w") as fw:
        fw.writelines(objs_info)
    return json_path


def generate_yaml(save_dir, class_map):
    kp_num = 2
    result = {
        "path": save_dir,
        "train": "./",
        "val": "./",
        "test": "./",
        "kpt_shape": [kp_num, 3],
        "flip_idx": [x for x in range(kp_num)],
        "names": {v: k for k, v in class_map.items()},
    }

    with open(os.path.join(save_dir, "dataset.yaml"), "w") as fw:
        yaml.dump(result, fw)


def main(label_json_dir, save_dir, class_info: List[str]):
    """_summary_

    Args:
        label_json_dir (_type_): json 保存的文件夹
        save_dir (_type_): txt 保存文件夹
        class_info (List[str]): 每个实例框的类别
    """
    all_json = get_all_file_path(label_json_dir, (".json"))
    class_map = {v: idx for idx, v in enumerate(class_info)}

    for json_path in tqdm(all_json):
        try:
            deal_one(json_path, save_dir, class_map)
        except Exception as e:
            print(f"{json_path} deal error {e}")
            raise e

    generate_yaml(save_dir, class_map)


if __name__ == "__main__":
    label_json_dir = "/data/tmp/can_rm/pointer_num"
    save_dir = "/data/tmp/can_rm/pointer_num"
    class_info = [
        "black_color_pointer",
        "red_color_pointer",
        "other_color_pointer",
        "special_pointer",
    ]
    main(label_json_dir, save_dir, class_info)
