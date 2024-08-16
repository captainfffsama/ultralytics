# -*- coding: utf-8 -*-
'''
@Author: captainfffsama
@Date: 2024-08-16 14:02:41
@LastEditors: captainfffsama tuanzhangsama@outlook.com
@LastEditTime: 2024-08-16 14:02:42
@FilePath: /ultralytics/chiebot_demo/pointer_deal/split_dataset.py
@Description:
'''
# -*- coding: utf-8 -*-
"""
@Author: captainfffsama
@Date: 2024-07-30 13:50:03
@LastEditors: captainfffsama tuanzhangsama@outlook.com
@LastEditTime: 2024-07-30 14:29:00
@FilePath: /ultralytics/chiebot_test/split_dataset.py
@Description:
"""

import os
import json
import random
from collections import defaultdict
from concurrent import futures
import yaml
from pprint import pprint


def get_all_file_path(file_dir: str, filter_=(".jpg")) -> list:
    # 遍历文件夹下所有的file
    return [
        os.path.join(maindir, filename)
        for maindir, _, file_name_list in os.walk(file_dir)
        for filename in file_name_list
        if os.path.splitext(filename)[1] in filter_
    ]


def parser_one(txt_path, idx2label):
    r = defaultdict(lambda: 0)
    with open(txt_path, "r") as fr:
        for i in fr.readlines():
            i = i.strip()
            if i:
                label_idx = int(i.split(" ")[0])
                r[idx2label[label_idx]] += 1
    return txt_path, r


def parser_label(txts, idx2label):
    count_r = defaultdict(lambda: 0)

    cls_sample_set = defaultdict(set)
    all_info = {}

    # with futures.ThreadPoolExecutor(32) as exec:
    #     tasks = [exec.submit(parser_one, txt_path) for txt_path in txts]
    # for task in futures.as_completed(tasks):
    #     txt_path, r = task.result()
    for txt in txts:
        txt_path, r = parser_one(txt,idx2label)
        if r:
            all_info[txt_path] = r
            for label, count in r.items():
                count_r[label] += count
                cls_sample_set[label].add(txt_path)

    for k,v in count_r.items():
        print(f"{k} have image: {len(cls_sample_set[k])} and object: {v}")


    label_sort=sorted(count_r.keys(), key=lambda x: count_r[x])
    for i,l in enumerate(label_sort[:-1]):
        for ll in label_sort[i+1:]:
                cls_sample_set[ll]=cls_sample_set[ll] - cls_sample_set[l]

    return count_r, cls_sample_set, all_info


def main(dataset_dir, save_dir, val_rate=0.1):
    with open(os.path.join(dataset_dir, "dataset.yaml"), "r", encoding="utf-8") as f:
        idx2label = yaml.load(f, Loader=yaml.FullLoader)["names"]

    all_label_path = get_all_file_path(dataset_dir, (".txt",))
    count_r, cls_sample_set, all_info = parser_label(all_label_path, idx2label)
    print("parser label done!!!")

    val_num = [(k, int(v * val_rate)) for k, v in count_r.items()]
    val_num = sorted(val_num, key=lambda x: x[1])

    current_val_remain_class_num = {k[0]: k[1] for k in val_num}

    val_list = []

    val_sample_info = defaultdict(lambda: [0, 0])

    for i in val_num:
        while current_val_remain_class_num[i[0]] > 0:
            select = random.choice(list(cls_sample_set[i[0]]))
            val_list.append(select)
            cls_sample_set[i[0]].remove(select)
            for label, count in all_info[select].items():
                current_val_remain_class_num[label] -= count
                val_sample_info[label][1] += count
                val_sample_info[label][0] += 1

    print("val sample info is:")
    for k, v in val_sample_info.items():
        print(f"{k} have image: {v[0]} and object: {v[1]}")

    with open(os.path.join(save_dir, "val.txt"), "w") as f:
        f.writelines([x + "\n" for x in val_list])

    train_remain = set(all_label_path) - set(val_list)
    with open(os.path.join(save_dir, "train.txt"), "w") as f:
        f.writelines([x + "\n" for x in train_remain])


if __name__ == "__main__":
    dataset_dir = "/data/tmp/can_rm/pointer_num"
    save_dir = "/data/tmp/can_rm/pointer_num_f"
    val_rate = 0.1
    main(dataset_dir, save_dir, val_rate)
