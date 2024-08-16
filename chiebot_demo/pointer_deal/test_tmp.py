# -*- coding: utf-8 -*-
'''
@Author: captainfffsama
@Date: 2024-08-16 14:02:55
@LastEditors: captainfffsama tuanzhangsama@outlook.com
@LastEditTime: 2024-08-16 14:02:55
@FilePath: /ultralytics/chiebot_demo/pointer_deal/test_tmp.py
@Description:
'''
# -*- coding: utf-8 -*-
"""
@Author: captainfffsama
@Date: 2024-08-15 17:05:59
@LastEditors: captainfffsama tuanzhangsama@outlook.com
@LastEditTime: 2024-08-15 17:05:59
@FilePath: /ultralytics/chiebot_test/pose_test/test_tmp.py
@Description:
"""

import os

from tqdm import tqdm
import torch

from ultralytics import YOLO
from ultralytics.models.yolo.pose.val import PoseValidator
LABEL2IDX={
  'black_color_pointer': 0,
  'red_color_pointer': 1,
  'other_color_pointer': 2,
  'special_pointer': 3,
}
def get_all_file_path(file_dir: str, filter_=(".jpg")) -> list:
    # 遍历文件夹下所有的file
    if os.path.isdir(file_dir):
        return [
            os.path.join(maindir, filename)
            for maindir, _, file_name_list in os.walk(file_dir)
            for filename in file_name_list
            if os.path.splitext(filename)[1] in filter_
        ]
    elif os.path.isfile(file_dir) and os.path.splitext(file_dir)[1] == ".txt":
        with open(file_dir, "r") as fr:
            r = [x.strip() for x in fr.readlines()]
        return r
    else:
        return []


class ChiebotPoseValidator(PoseValidator):
    def __init__(self,box_thr:float=0.25,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.box_thr=box_thr

    def _postprocess_one(self,pred:torch.Tensor):
        remind_idx=pred[:,4]>=self.box_thr
        pred=pred[remind_idx]
        label=pred[:,5].long()
        remain_pred=[]

        black_pointer_pred_idx=label==LABEL2IDX['black_color_pointer']
        if black_pointer_pred_idx.any():
            black_pointer_pred=pred[black_pointer_pred_idx]
            if black_pointer_pred_idx.sum()>1:
                _,idx=torch.topk(black_pointer_pred[:,5],1)
                remain=black_pointer_pred[idx]
            else:
                remain=black_pointer_pred
            remain_pred.append(remain)

        special_pointer_pred_idx=label==LABEL2IDX['special_pointer']
        if special_pointer_pred_idx.any():
            special_pointer_pred=pred[special_pointer_pred_idx]
            if special_pointer_pred_idx.sum()>1:
                _,idx=torch.topk(special_pointer_pred[:,5],1)
                remain=special_pointer_pred[idx]
            else:
                remain=special_pointer_pred
            remain_pred.append(remain)

        red_pointer_pred_idx=label==LABEL2IDX['red_color_pointer']
        if red_pointer_pred_idx.any():
            red_pointer_pred=pred[red_pointer_pred_idx]
            if red_pointer_pred_idx.sum()>2:
                _,idx=torch.topk(red_pointer_pred[:,5],2)
                remain=red_pointer_pred[idx]
            else:
                remain=red_pointer_pred
            remain_pred.append(remain)

        other_pointer_pred_idx=label==LABEL2IDX['other_color_pointer']
        if other_pointer_pred_idx.any():
            other_pointer_pred=pred[other_pointer_pred_idx]
            if other_pointer_pred_idx.sum()>2:
                _,idx=torch.topk(other_pointer_pred[:,5],2)
                remain=other_pointer_pred[idx]
            else:
                remain=other_pointer_pred
            remain_pred.append(remain)

        if remain_pred:
            return torch.concatenate(remain_pred,dim=0)
        else:
            return torch.empty((0,12),dtype=torch.float64,device=pred.device)

    def postprocess(self, preds):
        preds= super().postprocess(preds)
        final_preds=[]
        for pred in preds:
            final_preds.append(self._postprocess_one(pred))
        return final_preds


# Load a model
# model=YOLO("/data/weight/2401beijing/1/best.pt")
model = YOLO("/data/tmp/pointer_weight/best.pt")
# model = YOLO("/data/tmp/can_rm/model_test/soft_two_hot.pt")  # build a new model from YAML
m=model.val(data="/data/own_dataset/pointer_num_f/dataset.yaml",validator= ChiebotPoseValidator)
print(m.box.ap50)
print(m.pose.ap50)