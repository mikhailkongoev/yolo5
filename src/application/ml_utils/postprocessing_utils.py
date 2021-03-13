from ml_utils.bricks_utils import iou


def filter_class(labels, boxes, confs, idx_to_filter):
    cond = labels == idx_to_filter

    label_boxes = boxes[cond]
    label_confs = confs[cond]

    return label_boxes, label_confs


def postprocessing(brick_boxes, deffect_boxes, deffect_confs):
    deffect_boxes_processed = []
    for box, conf in zip(deffect_boxes, deffect_confs):
        inside_brick = False
        for brick_box in brick_boxes:
            inside_brick = inside_brick or (iou(box, brick_box) > 0)
        if inside_brick:
            deffect_boxes_processed.append((box, conf))
    deffect_boxes = [pair[0] for pair in deffect_boxes_processed]
    deffect_confs = [pair[1] for pair in deffect_boxes_processed]
    return deffect_boxes, deffect_confs
