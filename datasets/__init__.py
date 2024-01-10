from .samplers import *
from .transforms import *
from .torchvision_datasets import CocoDetection
from .coco import build as build_coco
from .coco_eval import *
from .coco_panoptic import *
from .data_prefetcher import *
from .panoptic_eval import *
from .pascal_voc import build as build_voc



def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, CocoDetection):
        return dataset.coco


def build_dataset(image_set, args):
    if args.dataset_file == "coco":
        return build_coco(image_set, args)
    if args.dataset_file == "voc":
        return build_voc(image_set, args)
    if args.dataset_file == "coco_panoptic":
        # to avoid making panopticapi required for coco
        from .coco_panoptic import build as build_coco_panoptic
        return build_coco_panoptic(image_set, args)
    raise ValueError(f"dataset {args.dataset_file} not supported")