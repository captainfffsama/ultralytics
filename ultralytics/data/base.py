# Ultralytics YOLO 🚀, AGPL-3.0 license

import glob
import math
import os
import random
from copy import deepcopy
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Optional
from packaging.version import Version


import cv2
import numpy as np
import psutil
import torch
from torch.utils.data import Dataset
import torchvision

from ultralytics.data.utils import FORMATS_HELP_MSG, HELP_URL, IMG_FORMATS
from ultralytics.utils import DEFAULT_CFG, LOCAL_RANK, LOGGER, NUM_THREADS, TQDM,remove_colorstr
from .utils import FORMATS_HELP_MSG, HELP_URL, IMG_FORMATS, load_dataset_cache_file, save_dataset_cache_file, get_hash

from ultralytics.debug_tools import timeblock, timethis


def remove_file(file):
    file = Path(file)
    if file.exists():
        os.remove(file)
    return file


def read_img(path, device="cpu"):
    if device != "cpu" and os.path.splitext(path)[-1].lower() in (".jpg", ".jpeg"):
        try:
            img = torchvision.io.read_file(path)
            img = (
                torchvision.io.decode_jpeg(img, torchvision.io.ImageReadMode.RGB, device=device)[[2, 1, 0], ...]
                .permute(1, 2, 0)
                .detach()
                .cpu()
                .numpy()
            )
        except Exception as e:
            img = cv2.imread(path)
    else:
        img = cv2.imread(path)
    return img


class BaseDataset(Dataset):
    """
    Base dataset class for loading and processing image data.

    Args:
        img_path (str): Path to the folder containing images.
        imgsz (int, optional): Image size. Defaults to 640.
        cache (bool, optional): Cache images to RAM or disk during training. Defaults to False.
        augment (bool, optional): If True, data augmentation is applied. Defaults to True.
        hyp (dict, optional): Hyperparameters to apply data augmentation. Defaults to None.
        prefix (str, optional): Prefix to print in log messages. Defaults to ''.
        rect (bool, optional): If True, rectangular training is used. Defaults to False.
        batch_size (int, optional): Size of batches. Defaults to None.
        stride (int, optional): Stride. Defaults to 32.
        pad (float, optional): Padding. Defaults to 0.0.
        single_cls (bool, optional): If True, single class training is used. Defaults to False.
        classes (list): List of included classes. Default is None.
        fraction (float): Fraction of dataset to utilize. Default is 1.0 (use all data).

    Attributes:
        im_files (list): List of image file paths.
        labels (list): List of label data dictionaries.
        ni (int): Number of images in the dataset.
        ims (list): List of loaded images.
        npy_files (list): List of numpy file paths.
        transforms (callable): Image transformation function.
    """

    def __init__(
        self,
        img_path,
        imgsz=640,
        cache=False,
        augment=True,
        hyp=DEFAULT_CFG,
        prefix="",
        rect=False,
        batch_size=16,
        stride=32,
        pad=0.5,
        single_cls=False,
        classes=None,
        fraction=1.0,
    ):
        """Initialize BaseDataset with given configuration and options."""
        super().__init__()
        self.image_decode_device = hyp.get("image_decode_device", "cpu")
        if "cuda" == self.image_decode_device:
            LOGGER.warning("WARNING ⚠️ GPU decoding is enabled. This will be skip exif.")
            if Version(torch.version.cuda) < Version("11.7"):
                LOGGER.warning("WARNING ⚠️ GPU decoding requires CUDA 11.7 or higher.")
                self.image_decode_device = "cpu"
            else:
                # FIXME: mutil gpu decode,cuda will raise error
                self.image_decode_device = f"cuda:{LOCAL_RANK}"
        self.img_path = img_path
        self.imgsz = imgsz
        self.augment = augment
        self.single_cls = single_cls
        self.prefix = prefix
        self.fraction = fraction
        self.im_files = self.get_img_files(self.img_path)
        self.labels = self.get_labels()
        self.update_labels(include_class=classes)  # single_cls and include_class
        self.ni = len(self.labels)  # number of images
        self.rect = rect
        self.batch_size = batch_size
        self.stride = stride
        self.pad = pad
        if self.rect:
            assert self.batch_size is not None
            self.set_rectangle()

        # Buffer thread for mosaic images
        self.buffer = []  # buffer size = batch size
        self.max_buffer_length = min((self.ni, self.batch_size * 8, 1000)) if self.augment else 0

        # Cache images (options are cache = True, False, None, "ram", "disk")
        self.ims, self.im_hw0, self.im_hw = [None] * self.ni, [None] * self.ni, [None] * self.ni
        self.npy_files = [Path(f).with_suffix(".npy") for f in self.im_files]

        self.cache_compress = hyp.get("cache_compress", False)
        self.npy_img_origin_hw = [None] * self.ni
        self.shape_cache_path = Path(str(Path(self.im_files[0]).parent)+f"_{remove_colorstr(self.prefix).replace(':','').strip()}.shapecache")

        self.cache = cache.lower() if isinstance(cache, str) else "ram" if cache is True else None
        if (self.cache == "ram" and self.check_cache_ram()) or self.cache == "disk":
            self.cache_images()

        # Transforms
        self.transforms = self.build_transforms(hyp=hyp)

    def clean_npy(self):
        LOGGER.info("️💡 npy file is invalid! start clean.")
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(remove_file, self.npy_files)
            pbar = TQDM(enumerate(results), total=self.ni, disable=LOCAL_RANK > 0)
            for i, x in pbar:
                pbar.desc = f"{self.prefix}Clean npy:({i}/{self.ni})"
            pbar.close()
        remove_file(self.shape_cache_path)

    def get_img_files(self, img_path):
        """Read image files."""
        try:
            f = []  # image files
            for p in img_path if isinstance(img_path, list) else [img_path]:
                p = Path(p)  # os-agnostic
                if p.is_dir():  # dir
                    f += glob.glob(str(p / "**" / "*.*"), recursive=True)
                    # F = list(p.rglob('*.*'))  # pathlib
                elif p.is_file():  # file
                    with open(p) as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep
                        f += [x.replace("./", parent) if x.startswith("./") else x for x in t]  # local to global path
                        # F += [p.parent / x.lstrip(os.sep) for x in t]  # local to global path (pathlib)
                else:
                    raise FileNotFoundError(f"{self.prefix}{p} does not exist")
            im_files = sorted(x.replace("/", os.sep) for x in f if x.split(".")[-1].lower() in IMG_FORMATS)
            # self.img_files = sorted([x for x in f if x.suffix[1:].lower() in IMG_FORMATS])  # pathlib
            assert im_files, f"{self.prefix}No images found in {img_path}. {FORMATS_HELP_MSG}"
        except Exception as e:
            raise FileNotFoundError(f"{self.prefix}Error loading data from {img_path}\n{HELP_URL}") from e
        if self.fraction < 1:
            im_files = im_files[: round(len(im_files) * self.fraction)]  # retain a fraction of the dataset
        return im_files

    def update_labels(self, include_class: Optional[list]):
        """Update labels to include only these classes (optional)."""
        include_class_array = np.array(include_class).reshape(1, -1)
        for i in range(len(self.labels)):
            if include_class is not None:
                cls = self.labels[i]["cls"]
                bboxes = self.labels[i]["bboxes"]
                segments = self.labels[i]["segments"]
                keypoints = self.labels[i]["keypoints"]
                j = (cls == include_class_array).any(1)
                self.labels[i]["cls"] = cls[j]
                self.labels[i]["bboxes"] = bboxes[j]
                if segments:
                    self.labels[i]["segments"] = [segments[si] for si, idx in enumerate(j) if idx]
                if keypoints is not None:
                    self.labels[i]["keypoints"] = keypoints[j]
            if self.single_cls:
                self.labels[i]["cls"][:, 0] = 0

    def load_image(self, i, rect_mode=True):
        """Loads 1 image from dataset index 'i', returns (im, resized hw)."""
        im, f, fn = self.ims[i], self.im_files[i], self.npy_files[i]
        if im is None:  # not cached in RAM
            if fn.exists() and "disk"==self.cache:  # load npy
                try:
                    im = np.load(fn)
                except Exception as e:
                    LOGGER.warning(f"{self.prefix}WARNING ⚠️ Removing corrupt *.npy image file {fn} due to: {e}")
                    Path(fn).unlink(missing_ok=True)
                    im = read_img(f, self.image_decode_device)  # BGR
            else:  # read image
                im = read_img(f, self.image_decode_device)  # BGR
            if im is None:
                raise FileNotFoundError(f"Image Not Found {f}")

            h0, w0 = im.shape[:2]  # orig hw
            if self.cache_compress and fn.exists() and "disk"==self.cache:
                if self.npy_img_origin_hw[i] is not None:
                    h0, w0 = self.npy_img_origin_hw[i]
                else:
                    LOGGER.warning(f"{self.prefix}WARNING ⚠️ {fn} shape is not in cache!")
            if rect_mode:  # resize long side to imgsz while maintaining aspect ratio
                if not (self.cache_compress and self.imgsz == max(h0, w0)):
                    r = self.imgsz / max(h0, w0)  # ratio
                    if r != 1:  # if sizes are not equal
                        w, h = (min(math.ceil(w0 * r), self.imgsz), min(math.ceil(h0 * r), self.imgsz))
                        im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
            elif not (h0 == w0 == self.imgsz):  # resize by stretching image to square imgsz
                im = cv2.resize(im, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)

            # Add to buffer if training with augmentations
            if self.augment:
                self.ims[i], self.im_hw0[i], self.im_hw[i] = im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized
                self.buffer.append(i)
                if 1 < len(self.buffer) >= self.max_buffer_length:  # prevent empty buffer
                    j = self.buffer.pop(0)
                    if self.cache != "ram":
                        self.ims[j], self.im_hw0[j], self.im_hw[j] = None, None, None

            return im, (h0, w0), im.shape[:2]

        return self.ims[i], self.im_hw0[i], self.im_hw[i]

    def cache_images(self):
        """Cache images to memory or disk."""
        b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
        fcn, storage = (self.cache_images_to_disk, "Disk") if self.cache == "disk" else (self.load_image, "RAM")

        if (self.cache_compress != self.shape_cache_path.exists()) and "disk" == self.cache:
            self.clean_npy()
        all_imgs_origin_hw = {}
        if "disk" == self.cache and self.cache_compress:
            if self.shape_cache_path.exists():
                all_imgs_origin_hw = load_dataset_cache_file(self.shape_cache_path)
                if all_imgs_origin_hw.get("hash", None) != get_hash(self.im_files,repr(self.imgsz)):
                    self.clean_npy()
                    all_imgs_origin_hw = {}

        origin_hw_pickle_update = False
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(fcn, range(self.ni))
            pbar = TQDM(enumerate(results), total=self.ni, disable=LOCAL_RANK > 0)
            for i, x in pbar:
                if self.cache == "disk":
                    b += self.npy_files[i].stat().st_size
                    if self.cache_compress:
                        if self.npy_img_origin_hw[i] is not None:
                            all_imgs_origin_hw[self.npy_files[i]] = self.npy_img_origin_hw[i]
                            origin_hw_pickle_update = True
                        else:
                            self.npy_img_origin_hw[i] = all_imgs_origin_hw.get(self.npy_files[i], None)

                else:  # 'ram'
                    self.ims[i], self.im_hw0[i], self.im_hw[i] = x  # im, hw_orig, hw_resized = load_image(self, i)
                    b += self.ims[i].nbytes
                pbar.desc = f"{self.prefix}Caching images ({b / gb:.1f}GB {storage})"
            pbar.close()

        if origin_hw_pickle_update:
            all_imgs_origin_hw["hash"] = get_hash(self.im_files,repr(self.imgsz))
            save_dataset_cache_file(self.prefix, self.shape_cache_path, all_imgs_origin_hw, "chiebot1.0")

    def cache_images_to_disk(self, i):
        """Saves an image as an *.npy file for faster loading."""
        f = self.npy_files[i]

        if not f.exists():
            img = read_img(self.im_files[i], self.image_decode_device)

            if self.cache_compress:
                h0, w0 = img.shape[:2]
                self.npy_img_origin_hw[i] = (h0, w0)
                r = self.imgsz / max(h0, w0)  # ratio
                if r != 1:  # if sizes are not equal
                    w, h = (min(math.ceil(w0 * r), self.imgsz), min(math.ceil(h0 * r), self.imgsz))
                    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
            np.save(f.as_posix(), img, allow_pickle=False)

    def check_cache_ram(self, safety_margin=0.5):
        """Check image caching requirements vs available memory."""
        b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
        n = min(self.ni, 30)  # extrapolate from 30 random images
        for _ in range(n):
            im = read_img(random.choice(self.im_files), self.image_decode_device)  # sample image
            ratio = self.imgsz / max(im.shape[0], im.shape[1])  # max(h, w)  # ratio
            b += im.nbytes * ratio**2
        mem_required = b * self.ni / n * (1 + safety_margin)  # GB required to cache dataset into RAM
        mem = psutil.virtual_memory()
        success = mem_required < mem.available  # to cache or not to cache, that is the question
        if not success:
            self.cache = None
            LOGGER.info(
                f"{self.prefix}{mem_required / gb:.1f}GB RAM required to cache images "
                f"with {int(safety_margin * 100)}% safety margin but only "
                f"{mem.available / gb:.1f}/{mem.total / gb:.1f}GB available, not caching images ⚠️"
            )
        return success

    def set_rectangle(self):
        """Sets the shape of bounding boxes for YOLO detections as rectangles."""
        bi = np.floor(np.arange(self.ni) / self.batch_size).astype(int)  # batch index
        nb = bi[-1] + 1  # number of batches

        s = np.array([x.pop("shape") for x in self.labels])  # hw
        ar = s[:, 0] / s[:, 1]  # aspect ratio
        irect = ar.argsort()
        self.im_files = [self.im_files[i] for i in irect]
        self.labels = [self.labels[i] for i in irect]
        ar = ar[irect]

        # Set training image shapes
        shapes = [[1, 1]] * nb
        for i in range(nb):
            ari = ar[bi == i]
            mini, maxi = ari.min(), ari.max()
            if maxi < 1:
                shapes[i] = [maxi, 1]
            elif mini > 1:
                shapes[i] = [1, 1 / mini]

        self.batch_shapes = np.ceil(np.array(shapes) * self.imgsz / self.stride + self.pad).astype(int) * self.stride
        self.batch = bi  # batch index of image

    def __getitem__(self, index):
        """Returns transformed label information for given index."""
        with timeblock("get image and label spend time:"):
            iandl = self.get_image_and_label(index)
        with timeblock("transforms spend time:"):
            r = self.transforms(iandl)
        # return self.transforms(self.get_image_and_label(index))
        return r

    def get_image_and_label(self, index):
        """Get and return label information from the dataset."""
        with timeblock("get label:"):
            label = deepcopy(
                self.labels[index]
            )  # requires deepcopy() https://github.com/ultralytics/ultralytics/pull/1948
            label.pop("shape", None)  # shape is for rect, remove it
        with timeblock("load image:"):
            label["img"], label["ori_shape"], label["resized_shape"] = self.load_image(index)
        with timeblock("update label other:"):
            label["ratio_pad"] = (
                label["resized_shape"][0] / label["ori_shape"][0],
                label["resized_shape"][1] / label["ori_shape"][1],
            )  # for evaluation
            if self.rect:
                label["rect_shape"] = self.batch_shapes[self.batch[index]]
            r = self.update_labels_info(label)
        # return self.update_labels_info(label)
        return r

    def __len__(self):
        """Returns the length of the labels list for the dataset."""
        return len(self.labels)

    def update_labels_info(self, label):
        """Custom your label format here."""
        return label

    def build_transforms(self, hyp=None):
        """
        Users can customize augmentations here.

        Example:
            ```python
            if self.augment:
                # Training transforms
                return Compose([])
            else:
                # Val transforms
                return Compose([])
            ```
        """
        raise NotImplementedError

    def get_labels(self):
        """
        Users can customize their own format here.

        Note:
            Ensure output is a dictionary with the following keys:
            ```python
            dict(
                im_file=im_file,
                shape=shape,  # format: (height, width)
                cls=cls,
                bboxes=bboxes, # xywh
                segments=segments,  # xy
                keypoints=keypoints, # xy
                normalized=True, # or False
                bbox_format="xyxy",  # or xywh, ltwh
            )
            ```
        """
        raise NotImplementedError
