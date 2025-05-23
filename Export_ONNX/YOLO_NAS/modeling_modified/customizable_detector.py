"""
A base for a detection network built according to the following scheme:
 * constructed from nested arch_params;
 * inside arch_params each nested level (module) has an explicit type and its required parameters
 * each module accepts in_channels and other parameters
 * each module defines out_channels property on construction
"""

from typing import Union, Optional, List, Callable
from functools import lru_cache

import torch
from torch import nn
from omegaconf import DictConfig

from super_gradients.common.decorators.factory_decorator import resolve_param
from super_gradients.common.factories.processing_factory import ProcessingFactory
from super_gradients.module_interfaces import SupportsReplaceNumClasses, SupportsReplaceInputChannels, HasPredict
from super_gradients.modules.head_replacement_utils import replace_num_classes_with_random_weights
from super_gradients.training.utils.utils import HpmStruct, arch_params_deprecated
from super_gradients.training.models.sg_module import SgModule
import super_gradients.common.factories.detection_modules_factory as det_factory
from super_gradients.training.utils.predict import ImagesDetectionPrediction
from super_gradients.training.pipelines.pipelines import DetectionPipeline
from super_gradients.training.processing.processing import Processing, ComposeProcessing, DetectionAutoPadding
from super_gradients.training.utils.detection_utils import DetectionPostPredictionCallback
from super_gradients.training.utils.media.image import ImageSource
from yolo_config import EXPORT_YOLO_RESIZE


class CustomizableDetector(HasPredict, SgModule):
    """
    A customizable detector with backbone -> neck -> heads
    Each submodule with its parameters must be defined explicitly.
    Modules should follow the interface of BaseDetectionModule
    """

    @arch_params_deprecated
    def __init__(
        self,
        backbone: Union[str, dict, HpmStruct, DictConfig],
        heads: Union[str, dict, HpmStruct, DictConfig],
        neck: Optional[Union[str, dict, HpmStruct, DictConfig]] = None,
        num_classes: int = None,
        bn_eps: Optional[float] = None,
        bn_momentum: Optional[float] = None,
        inplace_act: Optional[bool] = True,
        in_channels: int = 3,
    ):
        """
        :param backbone:    Backbone configuration.
        :param heads:       Head configuration.
        :param neck:        Neck configuration.
        :param num_classes: num classes to predict.
        :param bn_eps:      Epsilon for batch norm.
        :param bn_momentum: Momentum for batch norm.
        :param inplace_act: If True, do the operations operation in-place when possible.
        :param in_channels: number of input channels
        """
        super().__init__()

        self.heads_params = heads
        self.bn_eps = bn_eps
        self.bn_momentum = bn_momentum
        self.inplace_act = inplace_act
        self.in_channels = in_channels
        factory = det_factory.DetectionModulesFactory()

        # move num_classes into heads params
        if num_classes is not None:
            self.heads_params = factory.insert_module_param(self.heads_params, "num_classes", num_classes)

        self.backbone = factory.get(factory.insert_module_param(backbone, "in_channels", in_channels))
        if neck is not None:
            self.neck = factory.get(factory.insert_module_param(neck, "in_channels", self.backbone.out_channels))
            self.heads = factory.get(factory.insert_module_param(heads, "in_channels", self.neck.out_channels))
        else:
            self.neck = nn.Identity()
            self.heads = factory.get(factory.insert_module_param(heads, "in_channels", self.backbone.out_channels))

        self._initialize_weights(bn_eps, bn_momentum, inplace_act)

        # Processing params
        self._class_names: Optional[List[str]] = None
        self._image_processor: Optional[Processing] = None
        self._default_nms_iou: float = 0.7
        self._default_nms_conf: float = 0.5
        self._default_nms_top_k: int = 1024
        self._default_max_predictions = 300
        self._default_multi_label_per_box = True
        self._default_class_agnostic_nms = False
        self.inv_255 = float(1.0 / 255.0)

    def forward(self, x):
        return self.heads(self.neck(self.backbone(torch.nn.functional.interpolate(
            x.float(),
            size=EXPORT_YOLO_RESIZE,
            mode='bilinear',
            align_corners=True
        ) * self.inv_255)))

    def _initialize_weights(self, bn_eps: Optional[float] = None, bn_momentum: Optional[float] = None, inplace_act: Optional[bool] = True):
        for m in self.modules():
            t = type(m)
            if t is nn.BatchNorm2d:
                m.eps = bn_eps if bn_eps else m.eps
                m.momentum = bn_momentum if bn_momentum else m.momentum
            elif inplace_act and t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, nn.Mish]:
                m.inplace = True

    def prep_model_for_conversion(self, input_size: Optional[Union[tuple, list]] = None, **kwargs):
        for module in self.modules():
            if module != self and hasattr(module, "prep_model_for_conversion"):
                module.prep_model_for_conversion(input_size, **kwargs)

    def replace_head(self, new_num_classes: Optional[int] = None, new_head: Optional[nn.Module] = None):
        if new_num_classes is None and new_head is None:
            raise ValueError("At least one of new_num_classes, new_head must be given to replace output layer.")
        if new_head is not None:
            self.heads = new_head
        elif isinstance(self.heads, SupportsReplaceNumClasses):
            self.heads.replace_num_classes(new_num_classes, replace_num_classes_with_random_weights)
        else:
            factory = det_factory.DetectionModulesFactory()
            self.heads_params = factory.insert_module_param(self.heads_params, "num_classes", new_num_classes)
            self.heads = factory.get(factory.insert_module_param(self.heads_params, "in_channels", self.neck.out_channels))
            self._initialize_weights(self.bn_eps, self.bn_momentum, self.inplace_act)

    def replace_input_channels(self, in_channels: int, compute_new_weights_fn: Optional[Callable[[nn.Module, int], nn.Module]] = None):
        if isinstance(self.backbone, SupportsReplaceInputChannels):
            self.backbone.replace_input_channels(in_channels=in_channels, compute_new_weights_fn=compute_new_weights_fn)
            self.in_channels = self.get_input_channels()
        else:
            raise NotImplementedError(f"`{self.backbone.__class__.__name__}` does not support `replace_input_channels`")

    def get_input_channels(self) -> int:
        if isinstance(self.backbone, SupportsReplaceInputChannels):
            return self.backbone.get_input_channels()
        else:
            raise NotImplementedError(f"`{self.backbone.__class__.__name__}` does not support `replace_input_channels`")

    def get_post_prediction_callback(
        self, *, conf: float, iou: float, nms_top_k: int, max_predictions: int, multi_label_per_box: bool, class_agnostic_nms: bool
    ) -> DetectionPostPredictionCallback:
        """
        Get a post prediction callback for this model.

        :param conf:                A minimum confidence threshold for predictions to be used in post-processing.
        :param iou:                 A IoU threshold for boxes non-maximum suppression.
        :param nms_top_k:           The maximum number of detections to consider for NMS.
        :param max_predictions:     The maximum number of detections to return.
        :param multi_label_per_box: If True, each anchor can produce multiple labels of different classes.
                                    If False, each anchor can produce only one label of the class with the highest score.
        :param class_agnostic_nms:  If True, perform class-agnostic NMS (i.e IoU of boxes of different classes is checked).
                                    If False NMS is performed separately for each class.
        :return:
        """
        raise NotImplementedError

    @resolve_param("image_processor", ProcessingFactory())
    def set_dataset_processing_params(
        self,
        class_names: Optional[List[str]] = None,
        image_processor: Optional[Processing] = None,
        iou: Optional[float] = None,
        conf: Optional[float] = None,
        nms_top_k: Optional[int] = None,
        max_predictions: Optional[int] = None,
        multi_label_per_box: Optional[bool] = None,
        class_agnostic_nms: Optional[bool] = None,
    ) -> None:
        """Set the processing parameters for the dataset.

        :param class_names:         (Optional) Names of the dataset the model was trained on.
        :param image_processor:     (Optional) Image processing objects to reproduce the dataset preprocessing used for training.
        :param iou:                 (Optional) IoU threshold for the nms algorithm
        :param conf:                (Optional) Below the confidence threshold, prediction are discarded
        :param nms_top_k:           (Optional) The maximum number of detections to consider for NMS.
        :param max_predictions:     (Optional) The maximum number of detections to return.
        :param multi_label_per_box: (Optional) If True, each anchor can produce multiple labels of different classes.
                                    If False, each anchor can produce only one label of the class with the highest score.
        :param class_agnostic_nms:  (Optional) If True, perform class-agnostic NMS (i.e IoU of boxes of different classes is checked).
                                    If False NMS is performed separately for each class.
        """
        if class_names is not None:
            self._class_names = tuple(class_names)
        if image_processor is not None:
            self._image_processor = image_processor
        if iou is not None:
            self._default_nms_iou = float(iou)
        if conf is not None:
            self._default_nms_conf = float(conf)
        if nms_top_k is not None:
            self._default_nms_top_k = int(nms_top_k)
        if max_predictions is not None:
            self._default_max_predictions = int(max_predictions)
        if multi_label_per_box is not None:
            self._default_multi_label_per_box = bool(multi_label_per_box)
        if class_agnostic_nms is not None:
            self._default_class_agnostic_nms = bool(class_agnostic_nms)

    def get_processing_params(self) -> Optional[Processing]:
        return self._image_processor

    @lru_cache(maxsize=1)
    def _get_pipeline(
        self,
        *,
        iou: Optional[float] = None,
        conf: Optional[float] = None,
        fuse_model: bool = True,
        skip_image_resizing: bool = False,
        nms_top_k: Optional[int] = None,
        max_predictions: Optional[int] = None,
        multi_label_per_box: Optional[bool] = None,
        class_agnostic_nms: Optional[bool] = None,
        fp16: bool = True,
    ) -> DetectionPipeline:
        """Instantiate the prediction pipeline of this model.

        :param iou:                 (Optional) IoU threshold for the nms algorithm. If None, the default value associated to the training is used.
        :param conf:                (Optional) Below the confidence threshold, prediction are discarded.
                                    If None, the default value associated to the training is used.
        :param fuse_model:          If True, create a copy of the model, and fuse some of its layers to increase performance. This increases memory usage.
        :param skip_image_resizing: If True, the image processor will not resize the images.
        :param nms_top_k:           (Optional) The maximum number of detections to consider for NMS.
        :param max_predictions:     (Optional) The maximum number of detections to return.
        :param multi_label_per_box: (Optional) If True, each anchor can produce multiple labels of different classes.
                                    If False, each anchor can produce only one label of the class with the highest score.
        :param class_agnostic_nms:  (Optional) If True, perform class-agnostic NMS (i.e IoU of boxes of different classes is checked).
                                    If False NMS is performed separately for each class.
        :param fp16:                If True, use mixed precision for inference.
        """
        if None in (self._class_names, self._image_processor, self._default_nms_iou, self._default_nms_conf):
            raise RuntimeError(
                "You must set the dataset processing parameters before calling predict.\n" "Please call `model.set_dataset_processing_params(...)` first."
            )

        iou = self._default_nms_iou if iou is None else iou
        conf = self._default_nms_conf if conf is None else conf
        nms_top_k = self._default_nms_top_k if nms_top_k is None else nms_top_k
        max_predictions = self._default_max_predictions if max_predictions is None else max_predictions
        multi_label_per_box = self._default_multi_label_per_box if multi_label_per_box is None else multi_label_per_box
        class_agnostic_nms = self._default_class_agnostic_nms if class_agnostic_nms is None else class_agnostic_nms

        # Ensure that the image size is divisible by 32.
        if isinstance(self._image_processor, ComposeProcessing) and skip_image_resizing:
            image_processor = self._image_processor.get_equivalent_compose_without_resizing(
                auto_padding=DetectionAutoPadding(shape_multiple=(32, 32), pad_value=0)
            )
        else:
            image_processor = self._image_processor

        pipeline = DetectionPipeline(
            model=self,
            image_processor=image_processor,
            post_prediction_callback=self.get_post_prediction_callback(
                iou=iou,
                conf=conf,
                nms_top_k=nms_top_k,
                max_predictions=max_predictions,
                multi_label_per_box=multi_label_per_box,
                class_agnostic_nms=class_agnostic_nms,
            ),
            class_names=self._class_names,
            fuse_model=fuse_model,
            fp16=fp16,
        )
        return pipeline

    def predict(
        self,
        images: ImageSource,
        iou: Optional[float] = None,
        conf: Optional[float] = None,
        batch_size: int = 32,
        fuse_model: bool = True,
        skip_image_resizing: bool = False,
        nms_top_k: Optional[int] = None,
        max_predictions: Optional[int] = None,
        multi_label_per_box: Optional[bool] = None,
        class_agnostic_nms: Optional[bool] = None,
        fp16: bool = True,
    ) -> ImagesDetectionPrediction:
        """Predict an image or a list of images.

        :param images:              Images to predict.
        :param iou:                 (Optional) IoU threshold for the nms algorithm. If None, the default value associated to the training is used.
        :param conf:                (Optional) Below the confidence threshold, prediction are discarded.
                                    If None, the default value associated to the training is used.
        :param batch_size:          Maximum number of images to process at the same time.
        :param fuse_model:          If True, create a copy of the model, and fuse some of its layers to increase performance. This increases memory usage.
        :param skip_image_resizing: If True, the image processor will not resize the images.
        :param nms_top_k:           (Optional) The maximum number of detections to consider for NMS.
        :param max_predictions:     (Optional) The maximum number of detections to return.
        :param multi_label_per_box: (Optional) If True, each anchor can produce multiple labels of different classes.
                                    If False, each anchor can produce only one label of the class with the highest score.
        :param class_agnostic_nms:  (Optional) If True, perform class-agnostic NMS (i.e IoU of boxes of different classes is checked).
                                    If False NMS is performed separately for each class.
        :param fp16:                        If True, use mixed precision for inference.
        """
        pipeline = self._get_pipeline(
            iou=iou,
            conf=conf,
            fuse_model=fuse_model,
            skip_image_resizing=skip_image_resizing,
            nms_top_k=nms_top_k,
            max_predictions=max_predictions,
            multi_label_per_box=multi_label_per_box,
            class_agnostic_nms=class_agnostic_nms,
            fp16=fp16,
        )
        return pipeline(images, batch_size=batch_size)  # type: ignore

    def predict_webcam(
        self,
        iou: Optional[float] = None,
        conf: Optional[float] = None,
        fuse_model: bool = True,
        skip_image_resizing: bool = False,
        nms_top_k: Optional[int] = None,
        max_predictions: Optional[int] = None,
        multi_label_per_box: Optional[bool] = None,
        class_agnostic_nms: Optional[bool] = None,
        fp16: bool = True,
    ):
        """Predict using webcam.

        :param iou:                 (Optional) IoU threshold for the nms algorithm. If None, the default value associated to the training is used.
        :param conf:                (Optional) Below the confidence threshold, prediction are discarded.
                                    If None, the default value associated to the training is used.
        :param batch_size:          Maximum number of images to process at the same time.
        :param fuse_model:          If True, create a copy of the model, and fuse some of its layers to increase performance. This increases memory usage.
        :param skip_image_resizing: If True, the image processor will not resize the images.
        :param nms_top_k:           (Optional) The maximum number of detections to consider for NMS.
        :param max_predictions:     (Optional) The maximum number of detections to return.
        :param multi_label_per_box: (Optional) If True, each anchor can produce multiple labels of different classes.
                                    If False, each anchor can produce only one label of the class with the highest score.
        :param class_agnostic_nms:  (Optional) If True, perform class-agnostic NMS (i.e IoU of boxes of different classes is checked).
                                    If False NMS is performed separately for each class.
        :param fp16:                If True, use mixed precision for inference.
        """
        pipeline = self._get_pipeline(
            iou=iou,
            conf=conf,
            fuse_model=fuse_model,
            skip_image_resizing=skip_image_resizing,
            nms_top_k=nms_top_k,
            max_predictions=max_predictions,
            multi_label_per_box=multi_label_per_box,
            class_agnostic_nms=class_agnostic_nms,
            fp16=fp16,
        )
        pipeline.predict_webcam()

    def train(self, mode: bool = True):
        self._get_pipeline.cache_clear()
        torch.cuda.empty_cache()
        return super().train(mode)

    def get_finetune_lr_dict(self, lr: float):
        return {"heads": lr, "default": 0}
