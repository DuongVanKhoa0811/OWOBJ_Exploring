# ------------------------------------------------------------------------
# Modified from Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# -----------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
 
"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable
 
import torch
import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.open_world_eval import OWEvaluator
from datasets.panoptic_eval import PanopticEvaluator
from datasets.data_prefetcher import data_prefetcher
from util.box_ops import box_xyxy_to_cxcywh, box_cxcywh_to_xyxy
from util.plot_utils import plot_prediction,rescale_bboxes
import matplotlib.pyplot as plt
from copy import deepcopy
from torchvision.ops.boxes import batched_nms
import numpy as np


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, nc_epoch: int, max_norm: float = 0, wandb: object = None):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    prefetcher = data_prefetcher(data_loader, device, prefetch=True)
    samples, targets = prefetcher.next()

    for _ in metric_logger.log_every(range(len(data_loader)), print_freq, header):
        outputs = model(samples)
        loss_dict = criterion(outputs, targets) #criterion(samples, outputs, targets, epoch) 
        weight_dict = deepcopy(criterion.weight_dict)
        
        ## condition for starting nc loss computation after certain epoch so that the F_cls branch has the time
        ## to learn the within classes seperation.
        # if epoch < nc_epoch: 
        #     for k,v in weight_dict.items():
        #         if 'NC' in k:
        #             weight_dict[k] = 0
        # if epoch >= 0: 
        #     loss_dict.update({"loss_ce": v for k, v in loss_dict.items()  if 'loss_ce_obj'==k})
        #     for i in range(model.module.transformer.decoder.num_layers-1):
        #         loss_dict.update({"loss_ce" +f'_{i}': v for k, v in loss_dict.items()  if 'loss_ce_obj'+f'_{i}'==k})
         
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        # reduce losses over all GPUs for logging purposes

        loss_dict_reduced = utils.reduce_dict(loss_dict)
        ## Just printing NOt affectin gin loss function
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
 
        loss_value = losses_reduced_scaled.item()
 
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)
 
        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
        optimizer.step()
        
        if wandb is not None:
            wandb.log({"total_loss":loss_value})
            wandb.log(loss_dict_reduced_scaled)
            wandb.log(loss_dict_reduced_unscaled)
 
        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)
        
        samples, targets = prefetcher.next()
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

## ORIGINAL FUNCTION
@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, args):
    model.eval()
    criterion.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = OWEvaluator(base_ds, iou_types, coco=data_loader.dataset.coco,args=args)
 
    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )
    count=0
    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        if count==100:
            break
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        outputs = model(samples)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results_t= postprocessors['bbox'](outputs, orig_target_sizes)
        if isinstance(results_t,tuple):
            results=results_t[0]
            results_add=results_t[1]
            add_more=True
        else:
            results=results_t
            add_more=False
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            if not add_more: 
                coco_evaluator.update(res)
            else:
                coco_evaluator.update_more(res,results_add)
 
        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name
 
            panoptic_evaluator.update(res_pano)
        count+=1
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    # print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()
 
    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        res = coco_evaluator.summarize()
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    stats['metrics']=res
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]
    return stats, coco_evaluator
 
    
@torch.no_grad()
def get_exemplar_replay(model, exemplar_selection, device, data_loader):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = '[ExempReplay]'
    print_freq = 10
    prefetcher = data_prefetcher(data_loader, device, prefetch=True)
    samples, targets = prefetcher.next()
    image_sorted_scores_reduced={}
    model.eval()
    for _ in metric_logger.log_every(range(len(data_loader)), print_freq, header):
        outputs = model(samples)
        image_sorted_scores = exemplar_selection(samples, outputs, targets)
        for i in utils.combine_dict(image_sorted_scores):
            image_sorted_scores_reduced.update(i[0])
            
        metric_logger.update(loss=len(image_sorted_scores_reduced.keys()))
        samples, targets = prefetcher.next()
        
    print(f'found a total of {len(image_sorted_scores_reduced.keys())} images')
    return image_sorted_scores_reduced

def filter_boxes(scores, boxes, confidence=0.7, apply_nms=True, iou=0.5):
    keep = scores.max(-1).values > confidence
    scores, boxes = scores[keep], boxes[keep]
 
    if apply_nms:
        top_scores, labels = scores.max(-1)
        keep = batched_nms(boxes, top_scores, labels, iou)
        scores, boxes = scores[keep], boxes[keep]
 
    return scores, boxes

@torch.no_grad()
def viz(model, criterion, postprocessors, data_loader, base_ds, device, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    criterion.eval()
 
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
 
    for samples, targets in data_loader:
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        top_k = len(targets[0]['boxes'])
 
        outputs = model(samples)
        # results = postprocessors['bbox'](outputs, orig_target_sizes)
        # indices = outputs['pred_logits'][0].softmax(-1)[..., 1].sort(descending=True)[1][:top_k]
        # predictied_boxes = torch.stack([outputs['pred_boxes'][0][i] for i in indices]).unsqueeze(0)
        # logits = torch.stack([outputs['pred_logits'][0][i] for i in indices]).unsqueeze(0)
        fig, ax = plt.subplots(1, 3, figsize=(10,3), dpi=200)
 
        img = samples.tensors[0].cpu().permute(1,2,0).numpy()
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = (img * 255)
        img = img.astype('uint8')
        ax[0].imshow(img)
        # ax[0].set_title('Original Image')
        ax[0].grid('off')
        h, w = img.shape[:-1]
        # h, w = orig_target_sizes[0]
        out_logits, pred_obj, out_bbox = outputs['pred_logits'], outputs['pred_obj'], outputs['pred_boxes']
        out_logits[:,:, postprocessors['bbox'].invalid_cls_logits] = -10e10
        obj_prob = torch.exp(-postprocessors['bbox'].temperature*pred_obj).unsqueeze(-1)
        # probas  = (obj_prob*out_logits).softmax(-1)[0, :, :].cpu()
        probas  = out_logits.softmax(-1)[0, :, :].cpu()
        # probas = outputs['pred_logits'].softmax(-1)[0, :, :].cpu() #[100,81]
        predicted_boxes = out_bbox[0,].cpu() #[100,4]
        
        # probas = results['scores'][0, :, :].cpu() #[100,81]
        # predicted_boxes = outputs['boxes'][0,].cpu() #[100,4]
        predicted_boxes = rescale_bboxes(predicted_boxes.cpu(), [w, h])
        scores, predicted_boxes = filter_boxes(probas, predicted_boxes) #[11,81] [11,4]
        labels = scores.argmax(axis=1)#11 indices
        scores = scores.max(-1).values #11scores
 
        # Pred results
        # plot_prediction(samples.tensors[0:1], predictied_boxes, logits, ax[1], plot_prob=False)
        # Pred results
        # if not control the number of labels
        use_topk=True
        num_obj=20
        if not use_topk:
            plot_prediction(
                samples.tensors[0:1], 
                scores[-num_obj:], 
                predicted_boxes[-num_obj:], 
                labels[-num_obj:], 
                ax[1], 
                plot_prob=False,
                # dataset=dataset,
            )
        # if control the number of labels
        if use_topk:
            plot_prediction(
                samples.tensors[0:1], 
                scores[-top_k:], 
                predicted_boxes[-top_k:], 
                labels[-top_k:], 
                ax[1], 
                plot_prob=False,
                # dataset=dataset,
            )
        # ax[1].set_title('Prediction (Ours)')
 
        # GT Results
        t_bbox = rescale_bboxes(targets[0]['boxes'].cpu(), [w, h])
        # plot_prediction(samples.tensors[0:1], targets[0]['boxes'].unsqueeze(0), torch.zeros(1, targets[0]['boxes'].shape[0], 4).to(logits), ax[2], plot_prob=False)
        plot_prediction(samples.tensors[0:1], torch.ones(targets[0]['boxes'].shape[0]), t_bbox, targets[0]['labels'], ax[2], plot_prob=False)
        # [3, 967, 800] [1,4,4] [1,4,4]
        # ax[2].set_title('GT')
 
        for i in range(3):
            ax[i].set_aspect('equal')
            ax[i].set_axis_off()
 
        plt.savefig(os.path.join(output_dir, f'img_{int(targets[0]["image_id"][0])}.jpg'),pad_inches=0.2, bbox_inches='tight')