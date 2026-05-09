import numpy as np


def distance2bbox(points, distance, max_shape=None):
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = np.clip(x1, 0, max_shape[1])
        y1 = np.clip(y1, 0, max_shape[0])
        x2 = np.clip(x2, 0, max_shape[1])
        y2 = np.clip(y2, 0, max_shape[0])
    return np.stack([x1, y1, x2, y2], axis=-1)


def distance2kps(points, distance, max_shape=None):
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i%2] + distance[:, i]
        py = points[:, i%2+1] + distance[:, i+1]
        if max_shape is not None:
            px = np.clip(px, 0, max_shape[1])
            py = np.clip(py, 0, max_shape[0])
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)

class RetinaFace:
    def __init__(self, model_file=None, session=None):
        self.taskname = 'detection'
        self.center_cache = {}
        self.nms_thresh = 0.4
        self.det_thresh = 0.5
        self._init_vars()

    def _init_vars(self):
        input_shape = [1, 3, '?', '?']
        self.input_shape = input_shape
        
        if isinstance(input_shape[2], str):
            self.input_size = None
        else:
            self.input_size = tuple(input_shape[2:4][::-1])
        
        self.input_name = 'input.1'
        self.output_names = ['448', '471', '494', '451', '474', '497', '454', '477', '500']
        
        self.input_mean = 127.5
        self.input_std = 128.0
        self.use_kps = False
        self._anchor_ratio = 1.0
        self._num_anchors = 1
        
        self.fmc = 3
        self._feat_stride_fpn = [8, 16, 32]
        self._num_anchors = 2
        self.use_kps = True

    def prepare(self, ctx_id, **kwargs):
        if ctx_id < 0:
            print('Warning: CPU execution not supported in TRT version')
        nms_thresh = kwargs.get('nms_thresh', None)
        if nms_thresh is not None:
            self.nms_thresh = nms_thresh
        det_thresh = kwargs.get('det_thresh', None)
        if det_thresh is not None:
            self.det_thresh = det_thresh
        input_size = kwargs.get('input_size', None)
        if input_size is not None:
            if self.input_size is not None:
                print('warning: det_size is already set in detection model, ignore')
            else:
                self.input_size = input_size

    def detect(self, net_outs, input_size, max_num=0, metric='default'):
        det_scale = 1
        scores_list = []
        bboxes_list = []
        kpss_list = []
        
        input_height = input_size[0]
        input_width = input_size[1]
        fmc = self.fmc
        
        for idx, stride in enumerate(self._feat_stride_fpn):
            scores = net_outs[idx]
            bbox_preds = net_outs[idx+fmc]
            bbox_preds = bbox_preds * stride
            
            if self.use_kps:
                kps_preds = net_outs[idx+fmc*2] * stride
            
            height = input_height // stride
            width = input_width // stride
            K = height * width
            key = (height, width, stride)
            
            if key in self.center_cache:
                anchor_centers = self.center_cache[key]
            else:
                anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
                anchor_centers = (anchor_centers * stride).reshape((-1, 2))
                
                if self._num_anchors > 1:
                    anchor_centers = np.stack([anchor_centers]*self._num_anchors, axis=1).reshape((-1, 2))
                
                if len(self.center_cache) < 100:
                    self.center_cache[key] = anchor_centers

            pos_inds = np.where(scores >= self.det_thresh)[0]
            bboxes = distance2bbox(anchor_centers, bbox_preds)
            pos_scores = scores[pos_inds]
            pos_bboxes = bboxes[pos_inds]
            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)
            
            if self.use_kps:
                kpss = distance2kps(anchor_centers, kps_preds)
                kpss = kpss.reshape((kpss.shape[0], -1, 2))
                pos_kpss = kpss[pos_inds]
                kpss_list.append(pos_kpss)

        if len(scores_list) == 0:
            return np.zeros((0, 5)), None
            
        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]
        bboxes = np.vstack(bboxes_list) / det_scale
        
        if self.use_kps and len(kpss_list) > 0:
            kpss = np.vstack(kpss_list) / det_scale
        else:
            kpss = None
        
        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]
        keep = self.nms(pre_det)
        det = pre_det[keep, :]
        
        if self.use_kps and kpss is not None:
            kpss = kpss[order, :, :]
            kpss = kpss[keep, :, :]
        else:
            kpss = None
        
        if max_num > 0 and det.shape[0] > max_num:
            area = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
            img_center = 640 // 2, 640 // 2
            offsets = np.vstack([
                (det[:, 0] + det[:, 2]) / 2 - img_center[1],
                (det[:, 1] + det[:, 3]) / 2 - img_center[0]
            ])
            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
            
            if metric == 'max':
                values = area
            else:
                values = area - offset_dist_squared * 2.0
            
            bindex = np.argsort(values)[::-1]
            bindex = bindex[0:max_num]
            det = det[bindex, :]
            
            if kpss is not None:
                kpss = kpss[bindex, :]
        
        return det, kpss



    def detect_batch(self, net_outs_batch, input_size, max_num=0, metric='default'):
        """
        Convert batch engine outputs (3 concatenated tensors) to 
        single engine format (9 split tensors) expected by detect()
        
        Args:
            net_outs_batch: List of 3 batch outputs from TensorRT
                            [loc_batch(1,16800,4), conf_batch(1,16800,2), landmarks_batch(1,16800,10)]
        Returns:
            List of (detections, kpss) for each frame in batch
        """
        batch_size = net_outs_batch[0].shape[0]
        
        # Split indices for each feature level (stride 8, 16, 32)
        # For 640x640 input: 12800 anchors at stride 8, 3200 at stride 16, 800 at stride 32
        split_indices = [12800, 12800 + 3200]  # Cumulative: [12800, 16000]
        
        results = []
        
        for batch_idx in range(batch_size):
            # Extract single frame from batch (remove batch dimension)
            loc = net_outs_batch[0][batch_idx]      # Shape: (16800, 4)
            conf = net_outs_batch[1][batch_idx]     # Shape: (16800, 2)
            landmarks = net_outs_batch[2][batch_idx] # Shape: (16800, 10)
            
            # Split concatenated outputs into 9 separate outputs
            split_outputs = []
            
            # Split scores (conf has shape (16800, 2), we need just scores for each level)
            scores_full = conf[:, 1]  # Get face confidence (class 1)
            
            # Split by feature levels
            scores_levels = np.split(scores_full, split_indices)  # [12800, 3200, 800]
            loc_levels = np.split(loc, split_indices)             # [12800x4, 3200x4, 800x4]
            landmarks_levels = np.split(landmarks, split_indices) # [12800x10, 3200x10, 800x10]
            
            # Build the 9 outputs in the order your detect expects:
            # [scores_stride8, scores_stride16, scores_stride32,
            #  loc_stride8, loc_stride16, loc_stride32,
            #  landmarks_stride8, landmarks_stride16, landmarks_stride32]
            
            for i in range(3):
                # Add scores (reshape to (num_anchors, 1))
                split_outputs.append(scores_levels[i].reshape(-1, 1))
            
            for i in range(3):
                # Add location predictions
                split_outputs.append(loc_levels[i])
            
            for i in range(3):
                # Add landmark predictions
                split_outputs.append(landmarks_levels[i])
            
            # Now split_outputs contains 9 arrays matching your detect's expected format
            # Process using your existing detect method
            det, kpss = self.detect(split_outputs, input_size, max_num, metric)
            results.append((det, kpss))
    
        return results

    def detect_single(self, net_outs, input_size, max_num=0, metric='default'):
        """
        Original detect logic for single frame (adapted from your code)
        
        Args:
            net_outs: List of [scores, bbox_preds, kps_preds] for a single frame
            input_size: (height, width)
            max_num: maximum number of detections
            metric: metric for selection
        """
        det_scale = 1
        scores_list = []
        bboxes_list = []
        kpss_list = []
        
        input_height = input_size[0]
        input_width = input_size[1]
        fmc = self.fmc
        
        for idx, stride in enumerate(self._feat_stride_fpn):
            # Get scores and predictions for this feature level
            scores = net_outs[idx]  # Shape: (num_anchors, 2) or similar
            bbox_preds = net_outs[idx + fmc]
            bbox_preds = bbox_preds * stride
            
            if self.use_kps and len(net_outs) > idx + fmc * 2:
                kps_preds = net_outs[idx + fmc * 2] * stride
            
            height = input_height // stride
            width = input_width // stride
            K = height * width
            key = (height, width, stride)
            
            # Get or generate anchor centers
            if key in self.center_cache:
                anchor_centers = self.center_cache[key]
            else:
                anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
                anchor_centers = (anchor_centers * stride).reshape((-1, 2))
                
                if self._num_anchors > 1:
                    anchor_centers = np.stack([anchor_centers] * self._num_anchors, axis=1).reshape((-1, 2))
                
                if len(self.center_cache) < 100:
                    self.center_cache[key] = anchor_centers
            
            # Filter by detection threshold
            pos_inds = np.where(scores >= self.det_thresh)[0]
            
            if len(pos_inds) > 0:
                bboxes = self.distance2bbox(anchor_centers, bbox_preds)
                pos_scores = scores[pos_inds]
                pos_bboxes = bboxes[pos_inds]
                scores_list.append(pos_scores)
                bboxes_list.append(pos_bboxes)
                
                if self.use_kps:
                    kpss = self.distance2kps(anchor_centers, kps_preds)
                    kpss = kpss.reshape((kpss.shape[0], -1, 2))
                    pos_kpss = kpss[pos_inds]
                    kpss_list.append(pos_kpss)
        
        if len(scores_list) == 0:
            return np.zeros((0, 5)), None
        
        # Concatenate results from all feature levels
        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]
        bboxes = np.vstack(bboxes_list) / det_scale
        
        if self.use_kps and len(kpss_list) > 0:
            kpss = np.vstack(kpss_list) / det_scale
        else:
            kpss = None
        
        # Combine bboxes and scores
        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]
        
        # Apply NMS
        keep = self.nms(pre_det)
        det = pre_det[keep, :]
        
        if self.use_kps and kpss is not None:
            kpss = kpss[order, :, :]
            kpss = kpss[keep, :, :]
        else:
            kpss = None
        
        # Keep top max_num detections
        if max_num > 0 and det.shape[0] > max_num:
            area = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
            img_center = input_width // 2, input_height // 2
            offsets = np.vstack([
                (det[:, 0] + det[:, 2]) / 2 - img_center[1],
                (det[:, 1] + det[:, 3]) / 2 - img_center[0]
            ])
            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
            
            if metric == 'max':
                values = area
            else:
                values = area - offset_dist_squared * 2.0
            
            bindex = np.argsort(values)[::-1]
            bindex = bindex[0:max_num]
            det = det[bindex, :]
            
            if kpss is not None:
                kpss = kpss[bindex, :]
        
        return det, kpss






    def nms(self, dets):
        thresh = self.nms_thresh
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]

        return keep

