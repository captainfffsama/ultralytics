# -*- coding: utf-8 -*-
'''
@Author: captainfffsama
@Date: 2024-08-16 14:02:55
@LastEditors: captainfffsama tuanzhangsama@outlook.com
@LastEditTime: 2024-08-22 15:37:55
@FilePath: /ultralytics/chiebot_demo/pointer_deal/pointer_test.py
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
import argparse
from pathlib import Path

from tqdm import tqdm
import torch

from ultralytics import YOLO
from ultralytics.models.yolo.pose.val import PoseValidator
from labelme2yolov8_kp import main as l2ymain
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
        self.save_dir=kwargs['args'].get("save_dir",None)
        if self.save_dir:
            self.save_dir=Path(self.save_dir)
        super().__init__(save_dir=self.save_dir,*args,**kwargs)
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


def main(args):
    weight_path=args.model
    data_dir=args.datadir
    dataset_yaml=os.path.join(data_dir,"dataset.yaml")
    if not os.path.exists(dataset_yaml):
        print("will transform labelme to yolov8")
        class_info = [
            "black_color_pointer",
            "red_color_pointer",
            "other_color_pointer",
            "special_pointer",
        ]
        l2ymain(data_dir,data_dir,class_info)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)


    # Load a model
    # model=YOLO("/data/weight/2401beijing/1/best.pt")
    model = YOLO(weight_path)
    if args.infer_method=="chiebot":
        validator=ChiebotPoseValidator
    else:
        validator=PoseValidator
    m=model.val(data=dataset_yaml,validator=validator,save_json=True,save_dir=args.save_dir)

    use_val_idx=m.ap_class_index
    idx2cls=model.names
    results=[]
    for idx,(box_ap,pose_ap) in enumerate(zip(m.box.ap50,m.pose.ap50)):
        current_cls=idx2cls[use_val_idx[idx]]
        result=f"{current_cls} box ap50: {box_ap:^8.4f} pose ap50: {pose_ap:^8.4f}\n"
        results.append(result)
    with open(os.path.join(args.save_dir,"result.txt"),"w") as f:
        f.writelines(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m","--model",type=str, default="/data/tmp/pointer_weight/best.pt")
    parser.add_argument("-d","--datadir", type=str, default="/data/own_dataset/pointer_num_f/data_bak/data")
    parser.add_argument("-i","--infer_method", type=str, default="chiebot")
    parser.add_argument("-s","--save_dir", type=str, default="/data/tmp/pointer")
    args = parser.parse_args()

    main(args)

