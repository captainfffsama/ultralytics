###
 # @Author: captainfffsama
 # @Date: 2024-08-27 14:04:28
 # @LastEditors: captainfffsama tuanzhangsama@outlook.com
 # @LastEditTime: 2024-08-27 14:04:29
 # @FilePath: /ultralytics/chiebot_demo/pointer_deal/val.sh
 # @Description:
###
if [ "$#" -gt 0 ]
then
    model="$1"
    datadir="$2"
    /opt/conda/bin/python /root/ultralytics_cheibot/chiebot_demo/pointer_deal/pointer_test.py -m $model -d $datadir -s /root/mount/
else
    echo $#
    read -p "please input your model path:"model
    echo $#
    read -p "please input your data dir:"datadir
fi
