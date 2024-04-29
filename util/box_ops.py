import torch, os
from torchvision.ops.boxes import box_area


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    # import pdb; pdb.set_trace()
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / (union + 1e-6)
    return iou, union


def generalized_box_iou(boxes1, boxes2, data_batch=None):
    """Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    if not (boxes1[:, 2:] >= boxes1[:, :2]).all():
        import mmcv
        import cv2
        import numpy as np
        bs = len(data_batch['img'])
        boxes_pred = boxes1.reshape(bs, 100, 4)
        for i in range(bs):
            import torch.distributed as dist
            dist.barrier()
            idx = data_batch['idx']
            img = mmcv.imdenormalize(
            img=(data_batch['img'][i].cpu().numpy()).transpose(1, 2, 0), 
            mean=np.array([123.675, 116.28, 103.53]), 
            std=np.array([58.395, 57.12, 57.375]),
            to_bgr=True).astype(np.uint8)
            img_wh = data_batch['img_shape'][i]
            lhand_bbox = data_batch['lhand_bbox'][i]
            lhand_bbox = (lhand_bbox.reshape(-1,2).cpu().numpy()*img_wh.cpu().numpy()[::-1]).reshape(-1, 4)
            rhand_bbox = data_batch['rhand_bbox'][i]
            rhand_bbox = (rhand_bbox.reshape(-1,2).cpu().numpy()*img_wh.cpu().numpy()[::-1]).reshape(-1, 4)
            face_bbox = data_batch['face_bbox'][i]
            face_bbox = (face_bbox.reshape(-1,2).cpu().numpy()*img_wh.cpu().numpy()[::-1]).reshape(-1, 4)
            body_bbox = data_batch['body_bbox'][i]
            body_bbox = (body_bbox.reshape(-1,2).cpu().numpy()*img_wh.cpu().numpy()[::-1]).reshape(-1, 4)
            img = mmcv.imshow_bboxes(img, body_bbox, show=False, colors='green')
            img = mmcv.imshow_bboxes(img, lhand_bbox, show=False, colors='blue')
            img = mmcv.imshow_bboxes(img, rhand_bbox, show=False, colors='yellow')
            img = mmcv.imshow_bboxes(img, face_bbox, show=False, colors='red')
            cv2.imwrite(f'error_gt_img_{idx[i]}.jpg',img)
            
            img = mmcv.imdenormalize(
            img=(data_batch['img'][i].cpu().numpy()).transpose(1, 2, 0), 
            mean=np.array([123.675, 116.28, 103.53]), 
            std=np.array([58.395, 57.12, 57.375]),
            to_bgr=True).astype(np.uint8)
            boxes_pred_ = (boxes_pred[i].reshape(-1,2).detach().cpu().numpy()*img_wh.cpu().numpy()[::-1]).reshape(-1, 4)
            img = mmcv.imshow_bboxes(img.copy(), boxes_pred_, show=False)
            cv2.imwrite(f'error_pred_img_{idx[i]}.jpg',img)
            
        # assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
        # assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / (area + 1e-6)


# modified from torchvision to also return the union
def box_iou_pairwise(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, :2], boxes2[:, :2])  # [N,2]
    rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])  # [N,2]

    wh = (rb - lt).clamp(min=0)  # [N,2]
    inter = wh[:, 0] * wh[:, 1]  # [N]

    union = area1 + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou_pairwise(boxes1, boxes2):
    """Generalized IoU from https://giou.stanford.edu/

    Input:
        - boxes1, boxes2: N,4
    Output:
        - giou: N, 4
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    assert boxes1.shape == boxes2.shape
    iou, union = box_iou_pairwise(boxes1, boxes2)  # N, 4

    lt = torch.min(boxes1[:, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,2]
    area = wh[:, 0] * wh[:, 1]

    return iou - (area - union) / area


def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks.

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)


if __name__ == '__main__':
    x = torch.rand(5, 4)
    y = torch.rand(3, 4)
    iou, union = box_iou(x, y)
    import pdb
    pdb.set_trace()
