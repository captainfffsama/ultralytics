<div align="center">
  <p>
    <a href="https://www.ultralytics.com/events/yolovision" target="_blank">
      <img width="100%" src="https://raw.githubusercontent.com/ultralytics/assets/main/yolov8/banner-yolov8.png" alt="YOLO Vision banner"></a>
  </p>

[中文](https://docs.ultralytics.com/zh/) | [한국어](https://docs.ultralytics.com/ko/) | [日本語](https://docs.ultralytics.com/ja/) | [Русский](https://docs.ultralytics.com/ru/) | [Deutsch](https://docs.ultralytics.com/de/) | [Français](https://docs.ultralytics.com/fr/) | [Español](https://docs.ultralytics.com/es/) | [Português](https://docs.ultralytics.com/pt/) | [Türkçe](https://docs.ultralytics.com/tr/) | [Tiếng Việt](https://docs.ultralytics.com/vi/) | [العربية](https://docs.ultralytics.com/ar/) <br>

<div>
    <a href="https://github.com/ultralytics/ultralytics/actions/workflows/ci.yaml"><img src="https://github.com/ultralytics/ultralytics/actions/workflows/ci.yaml/badge.svg" alt="Ultralytics CI"></a>
    <a href="https://zenodo.org/badge/latestdoi/264818686"><img src="https://zenodo.org/badge/264818686.svg" alt="Ultralytics YOLOv8 Citation"></a>
    <a href="https://hub.docker.com/r/ultralytics/ultralytics"><img src="https://img.shields.io/docker/pulls/ultralytics/ultralytics?logo=docker" alt="Ultralytics Docker Pulls"></a>
    <a href="https://ultralytics.com/discord"><img alt="Ultralytics Discord" src="https://img.shields.io/discord/1089800235347353640?logo=discord&logoColor=white&label=Discord&color=blue"></a>
    <a href="https://community.ultralytics.com"><img alt="Ultralytics Forums" src="https://img.shields.io/discourse/users?server=https%3A%2F%2Fcommunity.ultralytics.com&logo=discourse&label=Forums&color=blue"></a>
    <br>
    <a href="https://console.paperspace.com/github/ultralytics/ultralytics"><img src="https://assets.paperspace.io/img/gradient-badge.svg" alt="Run Ultralytics on Gradient"></a>
    <a href="https://colab.research.google.com/github/ultralytics/ultralytics/blob/main/examples/tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open Ultralytics In Colab"></a>
    <a href="https://www.kaggle.com/ultralytics/yolov8"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open Ultralytics In Kaggle"></a>
</div>
<br>

[Ultralytics](https://ultralytics.com) [YOLOv8](https://github.com/ultralytics/ultralytics) 是一款前沿、最先进（SOTA）的模型，基于先前 YOLO 版本的成功，引入了新功能和改进，进一步提升性能和灵活性。YOLOv8 设计快速、准确且易于使用，使其成为各种物体检测与跟踪、实例分割、图像分类和姿态估计任务的绝佳选择。

我们希望这里的资源能帮助您充分利用 YOLOv8。请浏览 YOLOv8 <a href="https://docs.ultralytics.com/">文档</a> 了解详细信息，在 <a href="https://github.com/ultralytics/ultralytics/issues/new/choose">GitHub</a> 上提交问题以获得支持，并加入我们的 <a href="https://ultralytics.com/discord">Discord</a> 社区进行问题和讨论！

如需申请企业许可，请在 [Ultralytics Licensing](https://ultralytics.com/license) 处填写表格

<img width="100%" src="https://raw.githubusercontent.com/ultralytics/assets/main/yolov8/yolo-comparison-plots.png" alt="YOLOv8 performance plots"></a>

<div align="center">
  <a href="https://github.com/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-github.png" width="2%" alt="Ultralytics GitHub"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="2%" alt="space">
  <a href="https://www.linkedin.com/company/ultralytics/"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-linkedin.png" width="2%" alt="Ultralytics LinkedIn"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="2%" alt="space">
  <a href="https://twitter.com/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-twitter.png" width="2%" alt="Ultralytics Twitter"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="2%" alt="space">
  <a href="https://youtube.com/ultralytics?sub_confirmation=1"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-youtube.png" width="2%" alt="Ultralytics YouTube"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="2%" alt="space">
  <a href="https://www.tiktok.com/@ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-tiktok.png" width="2%" alt="Ultralytics TikTok"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="2%" alt="space">
  <a href="https://ultralytics.com/bilibili"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-bilibili.png" width="2%" alt="Ultralytics BiliBili"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="2%" alt="space">
  <a href="https://ultralytics.com/discord"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-discord.png" width="2%" alt="Ultralytics Discord"></a>
</div>
</div>

## <div align="center">文档</div>

请参阅下面的快速安装和使用示例，以及 [YOLOv8 文档](https://docs.ultralytics.com) 上有关培训、验证、预测和部署的完整文档。
<details open>
<summary>特别用法</summary>

**grpc 服务启动**
```shell
yolo grpc your_server_config_file_path
```

`your_server_config_file_path` 写法参见[docs/chiebot](docs/chiebot.md)

**yolo grpc 支持,嵌入运算啦!!!**
**yolo 现在可以对硬盘缓存压缩了!!!**

</details>

<details open>
<summary>安装</summary>

Pip install the ultralytics package including all [requirements](https://github.com/ultralytics/ultralytics/blob/main/pyproject.toml) in a [**Python>=3.8**](https://www.python.org/) environment with [**PyTorch>=1.8**](https://pytorch.org/get-started/locally/).

[![PyPI - Version](https://img.shields.io/pypi/v/ultralytics?logo=pypi&logoColor=white)](https://pypi.org/project/ultralytics/) [![Downloads](https://static.pepy.tech/badge/ultralytics)](https://pepy.tech/project/ultralytics) [![PyPI - Python Version](https://img.shields.io/pypi/pyversions/ultralytics?logo=python&logoColor=gold)](https://pypi.org/project/ultralytics/)

```bash
pip install ultralytics
```

如需使用包括[Conda](https://anaconda.org/conda-forge/ultralytics)、[Docker](https://hub.docker.com/r/ultralytics/ultralytics)和Git在内的其他安装方法，请参考[快速入门指南](https://docs.ultralytics.com/quickstart)。

[![Conda Version](https://img.shields.io/conda/vn/conda-forge/ultralytics?logo=condaforge)](https://anaconda.org/conda-forge/ultralytics) [![Docker Image Version](https://img.shields.io/docker/v/ultralytics/ultralytics?sort=semver&logo=docker)](https://hub.docker.com/r/ultralytics/ultralytics)

</details>

<details open>
<summary>Usage</summary>

### CLI

YOLOv8 可以在命令行界面（CLI）中直接使用，只需输入 `yolo` 命令：

```bash
yolo predict model=yolov8n.pt source='https://ultralytics.com/images/bus.jpg'
```

`yolo` 可用于各种任务和模式，并接受其他参数，例如 `imgsz=640`。查看 YOLOv8 [CLI 文档](https://docs.ultralytics.com/usage/cli)以获取示例。

### Python

YOLOv8 也可以在 Python 环境中直接使用，并接受与上述 CLI 示例中相同的[参数](https://docs.ultralytics.com/usage/cfg/)：

```python
from ultralytics import YOLO

# 加载模型
model = YOLO("yolov8n.yaml")  # 从头开始构建新模型
model = YOLO("yolov8n.pt")  # 加载预训练模型（建议用于训练）

# Use the model
model.train(data="coco8.yaml", epochs=3)  # train the model
metrics = model.val()  # evaluate model performance on the validation set
results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
path = model.export(format="onnx")  # export the model to ONNX format
```

See YOLOv8 [Python Docs](https://docs.ultralytics.com/usage/python) for more examples.

</details>

### Notebooks

Ultralytics provides interactive notebooks for YOLOv8, covering training, validation, tracking, and more. Each notebook is paired with a [YouTube](https://youtube.com/ultralytics?sub_confirmation=1) tutorial, making it easy to learn and implement advanced YOLOv8 features.

| Docs                                                                                                                               | Notebook                                                                                                                                                                                                                   |                                                                                                     YouTube                                                                                                     |
| ---------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| <a href="https://docs.ultralytics.com/modes/">YOLOv8 Train, Val, Predict and Export Modes</a>                                      | <a href="https://colab.research.google.com/github/ultralytics/ultralytics/blob/main/examples/tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>                  | <a href="https://youtu.be/j8uQc0qB91s"><center><img width=30% src="https://raw.githubusercontent.com/ultralytics/assets/main/social/logo-social-youtube-rect.png" alt="Ultralytics Youtube Video"></center></a> |
| <a href="https://docs.ultralytics.com/hub/quickstart/">Ultralytics HUB QuickStart</a>                                              | <a href="https://colab.research.google.com/github/ultralytics/ultralytics/blob/main/examples/hub.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>                       | <a href="https://youtu.be/lveF9iCMIzc"><center><img width=30% src="https://raw.githubusercontent.com/ultralytics/assets/main/social/logo-social-youtube-rect.png" alt="Ultralytics Youtube Video"></center></a> |
| <a href="https://docs.ultralytics.com/modes/track/">YOLOv8 Multi-Object Tracking in Videos</a>                                     | <a href="https://colab.research.google.com/github/ultralytics/ultralytics/blob/main/examples/object_tracking.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>           | <a href="https://youtu.be/hHyHmOtmEgs"><center><img width=30% src="https://raw.githubusercontent.com/ultralytics/assets/main/social/logo-social-youtube-rect.png" alt="Ultralytics Youtube Video"></center></a> |
| <a href="https://docs.ultralytics.com/guides/object-counting/">YOLOv8 Object Counting in Videos</a>                                | <a href="https://colab.research.google.com/github/ultralytics/ultralytics/blob/main/examples/object_counting.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>           | <a href="https://youtu.be/Ag2e-5_NpS0"><center><img width=30% src="https://raw.githubusercontent.com/ultralytics/assets/main/social/logo-social-youtube-rect.png" alt="Ultralytics Youtube Video"></center></a> |
| <a href="https://docs.ultralytics.com/guides/heatmaps/">YOLOv8 Heatmaps in Videos</a>                                              | <a href="https://colab.research.google.com/github/ultralytics/ultralytics/blob/main/examples/heatmaps.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>                  | <a href="https://youtu.be/4ezde5-nZZw"><center><img width=30% src="https://raw.githubusercontent.com/ultralytics/assets/main/social/logo-social-youtube-rect.png" alt="Ultralytics Youtube Video"></center></a> |
| <a href="https://docs.ultralytics.com/datasets/explorer/">Ultralytics Datasets Explorer with SQL and OpenAI Integration 🚀 New</a> | <a href="https://colab.research.google.com/github/ultralytics/ultralytics/blob/main/docs/en/datasets/explorer/explorer.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a> | <a href="https://youtu.be/3VryynorQeo"><center><img width=30% src="https://raw.githubusercontent.com/ultralytics/assets/main/social/logo-social-youtube-rect.png" alt="Ultralytics Youtube Video"></center></a> |

## <div align="center">Models</div>

在[COCO](https://docs.ultralytics.com/datasets/detect/coco)数据集上预训练的YOLOv8 [检测](https://docs.ultralytics.com/tasks/detect)，[分割](https://docs.ultralytics.com/tasks/segment)和[姿态](https://docs.ultralytics.com/tasks/pose)模型可以在这里找到，以及在[ImageNet](https://docs.ultralytics.com/datasets/classify/imagenet)数据集上预训练的YOLOv8 [分类](https://docs.ultralytics.com/tasks/classify)模型。所有的检测，分割和姿态模型都支持[追踪](https://docs.ultralytics.com/modes/track)模式。

<img width="1024" src="https://raw.githubusercontent.com/ultralytics/assets/main/im/banner-tasks.png" alt="Ultralytics YOLO supported tasks">

所有[模型](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models)在首次使用时会自动从最新的Ultralytics [发布版本](https://github.com/ultralytics/assets/releases)下载。

<details open><summary>Detection (COCO)</summary>

See [Detection Docs](https://docs.ultralytics.com/tasks/detect/) for usage examples with these models trained on [COCO](https://docs.ultralytics.com/datasets/detect/coco/), which include 80 pre-trained classes.

| Model                                                                                | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>A100 TensorRT<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| ------------------------------------------------------------------------------------ | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
| [YOLOv8n](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt) | 640                   | 37.3                 | 80.4                           | 0.99                                | 3.2                | 8.7               |
| [YOLOv8s](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s.pt) | 640                   | 44.9                 | 128.4                          | 1.20                                | 11.2               | 28.6              |
| [YOLOv8m](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m.pt) | 640                   | 50.2                 | 234.7                          | 1.83                                | 25.9               | 78.9              |
| [YOLOv8l](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8l.pt) | 640                   | 52.9                 | 375.2                          | 2.39                                | 43.7               | 165.2             |
| [YOLOv8x](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x.pt) | 640                   | 53.9                 | 479.1                          | 3.53                                | 68.2               | 257.8             |

- **mAP<sup>val</sup>** values are for single-model single-scale on [COCO val2017](https://cocodataset.org) dataset. <br>Reproduce by `yolo val detect data=coco.yaml device=0`
- **Speed** averaged over COCO val images using an [Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/) instance. <br>Reproduce by `yolo val detect data=coco.yaml batch=1 device=0|cpu`

</details>

<details><summary>Detection (Open Image V7)</summary>

See [Detection Docs](https://docs.ultralytics.com/tasks/detect/) for usage examples with these models trained on [Open Image V7](https://docs.ultralytics.com/datasets/detect/open-images-v7/), which include 600 pre-trained classes.

| Model                                                                                     | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>A100 TensorRT<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| ----------------------------------------------------------------------------------------- | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
| [YOLOv8n](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n-oiv7.pt) | 640                   | 18.4                 | 142.4                          | 1.21                                | 3.5                | 10.5              |
| [YOLOv8s](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s-oiv7.pt) | 640                   | 27.7                 | 183.1                          | 1.40                                | 11.4               | 29.7              |
| [YOLOv8m](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m-oiv7.pt) | 640                   | 33.6                 | 408.5                          | 2.26                                | 26.2               | 80.6              |
| [YOLOv8l](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8l-oiv7.pt) | 640                   | 34.9                 | 596.9                          | 2.43                                | 44.1               | 167.4             |
| [YOLOv8x](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x-oiv7.pt) | 640                   | 36.3                 | 860.6                          | 3.56                                | 68.7               | 260.6             |

- **mAP<sup>val</sup>** values are for single-model single-scale on [Open Image V7](https://docs.ultralytics.com/datasets/detect/open-images-v7/) dataset. <br>Reproduce by `yolo val detect data=open-images-v7.yaml device=0`
- **Speed** averaged over Open Image V7 val images using an [Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/) instance. <br>Reproduce by `yolo val detect data=open-images-v7.yaml batch=1 device=0|cpu`

</details>

<details><summary>Segmentation (COCO)</summary>

See [Segmentation Docs](https://docs.ultralytics.com/tasks/segment/) for usage examples with these models trained on [COCO-Seg](https://docs.ultralytics.com/datasets/segment/coco/), which include 80 pre-trained classes.

| Model                                                                                        | size<br><sup>(pixels) | mAP<sup>box<br>50-95 | mAP<sup>mask<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>A100 TensorRT<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| -------------------------------------------------------------------------------------------- | --------------------- | -------------------- | --------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
| [YOLOv8n-seg](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n-seg.pt) | 640                   | 36.7                 | 30.5                  | 96.1                           | 1.21                                | 3.4                | 12.6              |
| [YOLOv8s-seg](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s-seg.pt) | 640                   | 44.6                 | 36.8                  | 155.7                          | 1.47                                | 11.8               | 42.6              |
| [YOLOv8m-seg](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m-seg.pt) | 640                   | 49.9                 | 40.8                  | 317.0                          | 2.18                                | 27.3               | 110.2             |
| [YOLOv8l-seg](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8l-seg.pt) | 640                   | 52.3                 | 42.6                  | 572.4                          | 2.79                                | 46.0               | 220.5             |
| [YOLOv8x-seg](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x-seg.pt) | 640                   | 53.4                 | 43.4                  | 712.1                          | 4.02                                | 71.8               | 344.1             |

- **mAP<sup>val</sup>** values are for single-model single-scale on [COCO val2017](https://cocodataset.org) dataset. <br>Reproduce by `yolo val segment data=coco-seg.yaml device=0`
- **Speed** averaged over COCO val images using an [Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/) instance. <br>Reproduce by `yolo val segment data=coco-seg.yaml batch=1 device=0|cpu`

</details>

<details><summary>Pose (COCO)</summary>

See [Pose Docs](https://docs.ultralytics.com/tasks/pose/) for usage examples with these models trained on [COCO-Pose](https://docs.ultralytics.com/datasets/pose/coco/), which include 1 pre-trained class, person.

<<<<<<< HEAD
<<<<<<< HEAD
| 模型                                                                                                   | 尺寸<br><sup>(像素) | mAP<sup>pose<br>50-95 | mAP<sup>pose<br>50 | 速度<br><sup>CPU ONNX<br>(ms) | 速度<br><sup>A100 TensorRT<br>(ms) | 参数<br><sup>(M) | FLOPs<br><sup>(B) |
| ---------------------------------------------------------------------------------------------------- | --------------- | --------------------- | ------------------ | --------------------------- | -------------------------------- | -------------- | ----------------- |
| [YOLOv8n-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-pose.pt)       | 640             | 50.4                  | 80.1               | 131.8                       | 1.18                             | 3.3            | 9.2               |
| [YOLOv8s-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-pose.pt)       | 640             | 60.0                  | 86.2               | 233.2                       | 1.42                             | 11.6           | 30.2              |
| [YOLOv8m-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-pose.pt)       | 640             | 65.0                  | 88.8               | 456.3                       | 2.00                             | 26.4           | 81.0              |
| [YOLOv8l-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l-pose.pt)       | 640             | 67.6                  | 90.0               | 784.5                       | 2.59                             | 44.4           | 168.6             |
| [YOLOv8x-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-pose.pt)       | 640             | 69.2                  | 90.2               | 1607.1                      | 3.73                             | 69.4           | 263.2             |
| [YOLOv8x-pose-p6](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-pose-p6.pt) | 1280            | 71.6                  | 91.2               | 4088.7                      | 10.04                            | 99.1           | 1066.4            |
=======
| Model                                                                                                | size<br><sup>(pixels) | mAP<sup>pose<br>50-95 | mAP<sup>pose<br>50 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>A100 TensorRT<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| ---------------------------------------------------------------------------------------------------- | --------------------- | --------------------- | ------------------ | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
| [YOLOv8n-pose](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n-pose.pt)       | 640                   | 50.4                  | 80.1               | 131.8                          | 1.18                                | 3.3                | 9.2               |
| [YOLOv8s-pose](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s-pose.pt)       | 640                   | 60.0                  | 86.2               | 233.2                          | 1.42                                | 11.6               | 30.2              |
| [YOLOv8m-pose](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m-pose.pt)       | 640                   | 65.0                  | 88.8               | 456.3                          | 2.00                                | 26.4               | 81.0              |
| [YOLOv8l-pose](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8l-pose.pt)       | 640                   | 67.6                  | 90.0               | 784.5                          | 2.59                                | 44.4               | 168.6             |
| [YOLOv8x-pose](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x-pose.pt)       | 640                   | 69.2                  | 90.2               | 1607.1                         | 3.73                                | 69.4               | 263.2             |
| [YOLOv8x-pose-p6](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x-pose-p6.pt) | 1280                  | 71.6                  | 91.2               | 4088.7                         | 10.04                               | 99.1               | 1066.4            |
>>>>>>> main

- **Speed** averaged over COCO val images using an [Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/) instance. <br>Reproduce by `yolo val pose data=coco-pose.yaml batch=1 device=0|cpu`

</details>

<details><summary>OBB (DOTAv1)</summary>

See [OBB Docs](https://docs.ultralytics.com/tasks/obb/) for usage examples with these models trained on [DOTAv1](https://docs.ultralytics.com/datasets/obb/dota-v2/#dota-v10/), which include 15 pre-trained classes.

| Model                                                                                        | size<br><sup>(pixels) | mAP<sup>test<br>50 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>A100 TensorRT<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| -------------------------------------------------------------------------------------------- | --------------------- | ------------------ | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
| [YOLOv8n-obb](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n-obb.pt) | 1024                  | 78.0               | 204.77                         | 3.57                                | 3.1                | 23.3              |
| [YOLOv8s-obb](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s-obb.pt) | 1024                  | 79.5               | 424.88                         | 4.07                                | 11.4               | 76.3              |
| [YOLOv8m-obb](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m-obb.pt) | 1024                  | 80.5               | 763.48                         | 7.61                                | 26.4               | 208.6             |
| [YOLOv8l-obb](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8l-obb.pt) | 1024                  | 80.7               | 1278.42                        | 11.83                               | 44.5               | 433.8             |
| [YOLOv8x-obb](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x-obb.pt) | 1024                  | 81.36              | 1759.10                        | 13.23                               | 69.5               | 676.7             |

- **mAP<sup>test</sup>** values are for single-model multiscale on [DOTAv1](https://captain-whu.github.io/DOTA/index.html) dataset. <br>Reproduce by `yolo val obb data=DOTAv1.yaml device=0 split=test` and submit merged results to [DOTA evaluation](https://captain-whu.github.io/DOTA/evaluation.html).
- **Speed** averaged over DOTAv1 val images using an [Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/) instance. <br>Reproduce by `yolo val obb data=DOTAv1.yaml batch=1 device=0|cpu`
>>>>>>> main

</details>

<details><summary>Classification (ImageNet)</summary>

See [Classification Docs](https://docs.ultralytics.com/tasks/classify/) for usage examples with these models trained on [ImageNet](https://docs.ultralytics.com/datasets/classify/imagenet/), which include 1000 pretrained classes.

| Model                                                                                        | size<br><sup>(pixels) | acc<br><sup>top1 | acc<br><sup>top5 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>A100 TensorRT<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) at 640 |
| -------------------------------------------------------------------------------------------- | --------------------- | ---------------- | ---------------- | ------------------------------ | ----------------------------------- | ------------------ | ------------------------ |
<<<<<<< HEAD
| [YOLOv8s-cls](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s-cls.pt) | 224                   | 73.8             | 91.7             | 23.4                           | 0.35                                | 6.4                | 13.5                     |
| [YOLOv8m-cls](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8m-cls.pt) | 224                   | 76.8             | 93.5             | 85.4                           | 0.62                                | 17.0               | 42.7                     |
| [YOLOv8l-cls](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8l-cls.pt) | 224                   | 76.8             | 93.5             | 163.0                          | 0.87                                | 37.5               | 99.7                     |
| [YOLOv8x-cls](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8x-cls.pt) | 224                   | 79.0             | 94.6             | 232.0                          | 1.01                                | 57.4               | 154.8                    |
=======
| [YOLOv8n-cls](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n-cls.pt) | 224                   | 69.0             | 88.3             | 12.9                           | 0.31                                | 2.7                | 4.3                      |
| [YOLOv8s-cls](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s-cls.pt) | 224                   | 73.8             | 91.7             | 23.4                           | 0.35                                | 6.4                | 13.5                     |
| [YOLOv8m-cls](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m-cls.pt) | 224                   | 76.8             | 93.5             | 85.4                           | 0.62                                | 17.0               | 42.7                     |
| [YOLOv8l-cls](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8l-cls.pt) | 224                   | 78.3             | 94.2             | 163.0                          | 0.87                                | 37.5               | 99.7                     |
| [YOLOv8x-cls](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x-cls.pt) | 224                   | 79.0             | 94.6             | 232.0                          | 1.01                                | 57.4               | 154.8                    |
>>>>>>> main

- **acc** values are model accuracies on the [ImageNet](https://www.image-net.org/) dataset validation set. <br>Reproduce by `yolo val classify data=path/to/ImageNet device=0`
- **Speed** averaged over ImageNet val images using an [Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/) instance. <br>Reproduce by `yolo val classify data=path/to/ImageNet batch=1 device=0|cpu`

</details>

## <div align="center">集成</div>

我们与领先的AI平台的关键整合扩展了Ultralytics产品的功能，增强了数据集标签化、训练、可视化和模型管理等任务。探索Ultralytics如何与[Roboflow](https://roboflow.com/?ref=ultralytics)、ClearML、[Comet](https://bit.ly/yolov8-readme-comet)、Neural Magic以及[OpenVINO](https://docs.ultralytics.com/integrations/openvino)合作，优化您的AI工作流程。

<br>
<a href="https://ultralytics.com/hub" target="_blank">
<img width="100%" src="https://github.com/ultralytics/assets/raw/main/yolov8/banner-integrations.png" alt="Ultralytics active learning integrations"></a>
<br>
<br>

<div align="center">
  <a href="https://roboflow.com/?ref=ultralytics">
    <img src="https://github.com/ultralytics/assets/raw/main/partners/logo-roboflow.png" width="10%" alt="Roboflow logo"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="15%" height="0" alt="space">
  <a href="https://clear.ml/">
    <img src="https://github.com/ultralytics/assets/raw/main/partners/logo-clearml.png" width="10%" alt="ClearML logo"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="15%" height="0" alt="space">
  <a href="https://bit.ly/yolov8-readme-comet">
    <img src="https://github.com/ultralytics/assets/raw/main/partners/logo-comet.png" width="10%" alt="Comet ML logo"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="15%" height="0" alt="space">
  <a href="https://bit.ly/yolov5-neuralmagic">
    <img src="https://github.com/ultralytics/assets/raw/main/partners/logo-neuralmagic.png" width="10%" alt="NeuralMagic logo"></a>
</div>

|                                                           Roboflow                                                           |                                                 ClearML ⭐ NEW                                                  |                                                                       Comet ⭐ NEW                                                                        |                                          Neural Magic ⭐ NEW                                           |
| :--------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------: |
| Label and export your custom datasets directly to YOLOv8 for training with [Roboflow](https://roboflow.com/?ref=ultralytics) | Automatically track, visualize and even remotely train YOLOv8 using [ClearML](https://clear.ml/) (open-source!) | Free forever, [Comet](https://bit.ly/yolov8-readme-comet) lets you save YOLOv8 models, resume training, and interactively visualize and debug predictions | Run YOLOv8 inference up to 6x faster with [Neural Magic DeepSparse](https://bit.ly/yolov5-neuralmagic) |

## <div align="center">Ultralytics HUB</div>

<<<<<<< HEAD
体验 [Ultralytics HUB](https://bit.ly/ultralytics_hub) ⭐ 带来的无缝 AI，这是一个一体化解决方案，用于数据可视化、YOLOv5 和即将推出的 YOLOv8 🚀 模型训练和部署，无需任何编码。通过我们先进的平台和用户友好的 [Ultralytics 应用程序](https://ultralytics.com/app_install)，轻松将图像转化为可操作的见解，并实现您的 AI 愿景。现在就开始您的**免费**之旅！
=======
Experience seamless AI with [Ultralytics HUB](https://ultralytics.com/hub) ⭐, the all-in-one solution for data visualization, YOLOv5 and YOLOv8 🚀 model training and deployment, without any coding. Transform images into actionable insights and bring your AI visions to life with ease using our cutting-edge platform and user-friendly [Ultralytics App](https://ultralytics.com/app_install). Start your journey for **Free** now!
>>>>>>> main

<a href="https://ultralytics.com/hub" target="_blank">
<img width="100%" src="https://github.com/ultralytics/assets/raw/main/im/ultralytics-hub.png" alt="Ultralytics HUB preview image"></a>

## <div align="center">贡献</div>

我们喜欢您的参与！没有社区的帮助，YOLOv5 和 YOLOv8 将无法实现。请参阅我们的[贡献指南](https://docs.ultralytics.com/help/contributing)以开始使用，并填写我们的[调查问卷](https://ultralytics.com/survey?utm_source=github&utm_medium=social&utm_campaign=Survey)向我们提供您的使用体验反馈。感谢所有贡献者的支持！🙏

<!-- SVG image from https://opencollective.com/ultralytics/contributors.svg?width=990 -->

<a href="https://github.com/ultralytics/ultralytics/graphs/contributors">
<img width="100%" src="https://github.com/ultralytics/assets/raw/main/im/image-contributors.png" alt="Ultralytics open-source contributors"></a>

## <div align="center">许可证</div>

Ultralytics 提供两种许可证选项以适应各种使用场景：

- **AGPL-3.0 许可证**：这个[OSI 批准](https://opensource.org/licenses/)的开源许可证非常适合学生和爱好者，可以推动开放的协作和知识分享。请查看[LICENSE](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) 文件以了解更多细节。
- **企业许可证**：专为商业用途设计，该许可证允许将 Ultralytics 的软件和 AI 模型无缝集成到商业产品和服务中，从而绕过 AGPL-3.0 的开源要求。如果您的场景涉及将我们的解决方案嵌入到商业产品中，请通过 [Ultralytics Licensing](https://ultralytics.com/license)与我们联系。

## <div align="center">联系方式</div>

对于 Ultralytics 的错误报告和功能请求，请访问 [GitHub Issues](https://github.com/ultralytics/ultralytics/issues)，并加入我们的 [Discord](https://ultralytics.com/discord) 社区进行问题和讨论！

<br>
<div align="center">
  <a href="https://github.com/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-github.png" width="3%" alt="Ultralytics GitHub"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://www.linkedin.com/company/ultralytics/"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-linkedin.png" width="3%" alt="Ultralytics LinkedIn"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://twitter.com/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-twitter.png" width="3%" alt="Ultralytics Twitter"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://youtube.com/ultralytics?sub_confirmation=1"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-youtube.png" width="3%" alt="Ultralytics YouTube"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://www.tiktok.com/@ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-tiktok.png" width="3%" alt="Ultralytics TikTok"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://ultralytics.com/bilibili"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-bilibili.png" width="3%" alt="Ultralytics BiliBili"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://ultralytics.com/discord"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-discord.png" width="3%" alt="Ultralytics Discord"></a>
</div>
