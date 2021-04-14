from ..registry import DETECTORS
from .single_stage import SingleStageDetector
from det3d.torchie.trainer import load_checkpoint
from torch import nn, empty, cat
from copy import deepcopy 

@DETECTORS.register_module
class FMF_Concat_VN(SingleStageDetector):
    def __init__(
        self,
        reader,
        backbone,
        neck,
        bbox_head,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
    ):
        super(FMF_Concat_VN, self).__init__(
            reader, backbone, neck, bbox_head, train_cfg, test_cfg, pretrained
        )
        self.tensor = empty(4, bbox_head.in_channels, 128, 128).cuda()   # TODO don't hardcore Batch_size 
        self.shared_conv = nn.Sequential(
            nn.Conv2d(2*bbox_head.in_channels, bbox_head.in_channels,
            kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(bbox_head.in_channels),
            nn.ReLU(inplace=False).cuda()
        )

        
    def extract_feat(self, data):
        input_features = self.reader(data["features"], data["num_voxels"])
        x, voxel_feature = self.backbone(
            input_features, data["coors"], data["batch_size"], data["input_shape"]
        )
        if self.with_neck:
            x = self.neck(x)

        return x, voxel_feature

    def forward(self, example, return_loss=True, **kwargs):
        voxels = example["voxels"]
        coordinates = example["coordinates"]
        num_points_in_voxel = example["num_points"]
        num_voxels = example["num_voxels"]

        batch_size = len(num_voxels)

        data = dict(
            features=voxels,
            num_voxels=num_points_in_voxel,
            coors=coordinates,
            batch_size=batch_size,
            input_shape=example["shape"][0],
        )

        x, _ = self.extract_feat(data)

        x1 = cat((x,self.tensor),1)
        self.temp = x.detach().clone()
        x = self.shared_conv(x1)

        preds = self.bbox_head(x)

        if return_loss:
            return self.bbox_head.loss(example, preds)
        else:
            return self.bbox_head.predict(example, preds, self.test_cfg)

    def forward_two_stage(self, example, return_loss=True, **kwargs):
        voxels = example["voxels"]
        coordinates = example["coordinates"]
        num_points_in_voxel = example["num_points"]
        num_voxels = example["num_voxels"]

        batch_size = len(num_voxels)

        data = dict(
            features=voxels,
            num_voxels=num_points_in_voxel,
            coors=coordinates,
            batch_size=batch_size,
            input_shape=example["shape"][0],
        )

        x, voxel_feature = self.extract_feat(data)
        bev_feature = x 
        preds = self.bbox_head(x)

        # manual deepcopy ...
        new_preds = []
        for pred in preds:
            new_pred = {} 
            for k, v in pred.items():
                new_pred[k] = v.detach()

            new_preds.append(new_pred)

        boxes = self.bbox_head.predict(example, new_preds, self.test_cfg)

        if return_loss:
            return boxes, bev_feature, voxel_feature, self.bbox_head.loss(example, preds)
        else:
            return boxes, bev_feature, voxel_feature, None 
