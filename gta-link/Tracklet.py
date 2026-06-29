class Tracklet:
    def __init__(self, track_id=None, frames=None, scores=None, bboxes=None, feats=None, class_ids=None):
        '''
        Initialize the Tracklet with IDs, times, scores, bounding boxes, and optional features.
        If parameters are not provided, initializes them to None or empty lists.

        Args:
            track_id (int, optional): Unique identifier for the track. Defaults to None.
            frames (list or int, optional): Frame numbers where the track is present. Can be a list of frames or a single frame. Defaults to None.
            scores (list or float, optional): Detection scores corresponding to frames. Can be a list of scores or a single score. Defaults to None.
            bboxes (list of lists or list, optional): Bounding boxes corresponding to each frame. Each bounding box is a list of 4 elements. Defaults to None.
            feats (list of np.array, optional): Feature vectors corresponding to frames. Each feature should be a numpy array of shape (512,). Defaults to None.
            class_ids (list or int, optional): Canonical class id (1=goalkeeper, 2=player, 3=referee, -1=unknown) per frame, aligned 1:1 with frames. Can be a list or a single id. Defaults to None.
        '''
        self.track_id = track_id
        self.parent_id = track_id
        self.scores = scores if isinstance(scores, list) else [scores] if scores is not None else []
        self.times = frames if isinstance(frames, list) else [frames] if frames is not None else []
        self.bboxes = bboxes if isinstance(bboxes, list) and bboxes and isinstance(bboxes[0], list) else [bboxes] if bboxes is not None else []
        self.features = feats if feats is not None else []
        # Per-frame canonical class id, aligned 1:1 with times/scores/bboxes/features.
        # Constructed even when not provided so older pickles / code paths that
        # build Tracklets without classes keep working; fall back to -1 (unknown)
        # so it stays length-aligned with the frames that WERE provided.
        if class_ids is None:
            self.class_ids = [-1] * len(self.times)
        elif isinstance(class_ids, list):
            self.class_ids = class_ids
        else:
            self.class_ids = [class_ids]

    def __setstate__(self, state):
        '''
        Restore from a pickle, then backfill class_ids for legacy pickles.

        Tracklets pickled before class_ids existed bypass __init__ on load, so
        their __dict__ has no class_ids and would AttributeError in append_det /
        extract. Ensure class_ids exists and is length-aligned with times,
        defaulting to -1 (unknown) when missing or SHORTER than times.

        The slim refined_tracklets.pkl (Stage 2's OPT-1 export) deliberately
        clears times/bboxes/features while KEEPING class_ids + scores (Stage 3
        reads those), so class_ids is intentionally LONGER than the emptied
        times. That is not a legacy mismatch, so it is left intact — only a
        missing or short class_ids is backfilled.
        '''
        self.__dict__.update(state)
        times = getattr(self, 'times', [])
        class_ids = getattr(self, 'class_ids', None)
        if class_ids is None or len(class_ids) < len(times):
            self.class_ids = [-1] * len(times)

    def append_det(self, frame, score, bbox, class_id=-1):
        '''
        Appends a detection to the tracklet.

        Args:
            frame (int): Frame number for the detection.
            score (float): Detection score.
            bbox (list of float): Bounding box with four elements [x, y, width, height].
            class_id (int, optional): Canonical class id for this detection. Defaults to -1 (unknown).
        '''
        self.scores.append(score)
        self.times.append(frame)
        self.bboxes.append(bbox)
        self.class_ids.append(class_id)

    def append_feat(self, feat):
        '''
        Appends a feature vector to the tracklet.

        Args:
            feat (np.array): Feature vector of shape (512,).
        '''
        self.features.append(feat)

    def extract(self, start, end):
        '''
        Extracts a subtrack from the tracklet between two indices.

        Args:
            start (int): Start index for the extraction.
            end (int): End index for the extraction.

        Returns:
            Tracklet: A new Tracklet object that is a subset of the original from start to end indices.
        '''
        subtrack = Tracklet(self.track_id, self.times[start:end + 1], self.scores[start:end + 1], self.bboxes[start:end + 1], self.features[start:end + 1] if self.features else None, self.class_ids[start:end + 1] if self.class_ids else None)
        return subtrack