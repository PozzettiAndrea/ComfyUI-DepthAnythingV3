# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Adapted from [VGGT-Long](https://github.com/DengKaiCQ/VGGT-Long)

import logging
import os
from pathlib import Path
import comfy.model_management

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

import torch
import torchvision.transforms as T
from PIL import Image
from torch import nn
from tqdm import tqdm

logger = logging.getLogger("DA3Streaming")

# SALAD VPR model - requires external salad package
# The salad model code is not included in the DA3 streaming repo.
# Users need to provide a compatible checkpoint.
_salad_helper = None
try:
    import sys
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    SALAD_ROOT = os.path.join(CURRENT_DIR, "salad")
    if os.path.exists(SALAD_ROOT) and SALAD_ROOT not in sys.path:
        sys.path.insert(0, SALAD_ROOT)
    from models import helper as _salad_helper
except ImportError:
    logger.warning("SALAD model code not found. Loop closure detection will not be available.")


class VPRModel(nn.Module):
    """This is the main model for Visual Place Recognition
    we use Pytorch Lightning for modularity purposes.

    Args:
        pl (_type_): _description_
    """

    def __init__(
        self,
        # ---- Backbone
        backbone_arch="resnet50",
        backbone_config={},
        # ---- Aggregator
        agg_arch="ConvAP",
        agg_config={},
    ):
        super().__init__()

        # Backbone
        self.encoder_arch = backbone_arch
        self.backbone_config = backbone_config

        # Aggregator
        self.agg_arch = agg_arch
        self.agg_config = agg_config

        # ----------------------------------
        # get the backbone and the aggregator
        if _salad_helper is None:
            raise ImportError("SALAD model code is required for loop closure. Place salad model code in the 'salad/' directory.")
        self.backbone = _salad_helper.get_backbone(backbone_arch, backbone_config)
        self.aggregator = _salad_helper.get_aggregator(agg_arch, agg_config)

    # the forward pass of the lightning model
    def forward(self, x):
        x = self.backbone(x)
        x = self.aggregator(x)
        return x


class LoopDetector:
    """Loop detector class for detecting loop closures in image sequences"""

    def __init__(self, image_dir, output="loop_closures.txt", config=None):
        """Initialize the loop detector

        Args:
            image_dir: Directory path containing images
            ckpt_path: Model checkpoint path
            image_size: Image resize dimensions [height width]
            batch_size: Batch size for processing
            similarity_threshold: Similarity threshold for loop closure
            top_k: Number of nearest neighbors to check for each image
            use_nms: Whether to use Non-Maximum Suppression (NMS) filtering
            nms_threshold: NMS threshold for minimum frame difference between loop pairs
            output: Output file path
        """
        if not HAS_FAISS:
            raise ImportError("faiss-gpu or faiss-cpu is required for loop closure detection")

        self.config = config
        self.image_dir = image_dir
        self.ckpt_path = self.config["Weights"]["SALAD"]
        self.image_size = self.config["Loop"]["SALAD"]["image_size"]
        self.batch_size = self.config["Loop"]["SALAD"]["batch_size"]
        self.similarity_threshold = self.config["Loop"]["SALAD"]["similarity_threshold"]
        self.top_k = self.config["Loop"]["SALAD"]["top_k"]
        self.use_nms = self.config["Loop"]["SALAD"]["use_nms"]
        self.nms_threshold = self.config["Loop"]["SALAD"]["nms_threshold"]
        self.output = output

        self.model = None
        self.device = None
        self.image_paths = None
        self.descriptors = None
        self.loop_closures = None

    def _input_transform(self, image_size=None):
        """Create image transformation function"""
        MEAN = [0.485, 0.456, 0.406]
        STD = [0.229, 0.224, 0.225]
        if image_size:
            return T.Compose(
                [
                    T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
                    T.ToTensor(),
                    T.Normalize(mean=MEAN, std=STD),
                ]
            )
        else:
            return T.Compose([T.ToTensor(), T.Normalize(mean=MEAN, std=STD)])

    def load_model(self):
        """Load model"""
        model = VPRModel(
            backbone_arch="dinov2_vitb14",
            backbone_config={
                "num_trainable_blocks": 4,
                "return_token": True,
                "norm_layer": True,
            },
            agg_arch="SALAD",
            agg_config={
                "num_channels": 768,
                "num_clusters": 64,
                "cluster_dim": 128,
                "token_dim": 256,
            },
        )

        model.load_state_dict(torch.load(self.ckpt_path, map_location="cpu", weights_only=True))
        model = model.eval()
        device = comfy.model_management.get_torch_device()
        model = model.to(device)
        logger.info(f"Model loaded: {self.ckpt_path}")

        self.model = model
        self.device = device
        return model, device

    def get_image_paths(self):
        """Get paths of all image files in directory"""
        image_extensions = [".jpg", ".jpeg", ".png"]
        image_paths = []

        for ext in image_extensions:
            image_paths.extend(list(Path(self.image_dir).glob(f"*{ext}")))
            image_paths.extend(list(Path(self.image_dir).glob(f"*{ext.upper()}")))

        image_paths = sorted(image_paths)
        self.image_paths = image_paths
        return image_paths

    def extract_descriptors(self):
        """Extract image feature descriptors"""
        if self.model is None or self.device is None:
            self.load_model()

        if self.image_paths is None:
            self.get_image_paths()

        transform = self._input_transform(self.image_size)
        descriptors = []

        for i in tqdm(
            range(0, len(self.image_paths), self.batch_size), desc="Extracting features"
        ):
            batch_paths = self.image_paths[i : i + self.batch_size]
            batch_imgs = []

            for path in batch_paths:
                try:
                    img = Image.open(path).convert("RGB")
                    img = transform(img)
                    batch_imgs.append(img)
                except Exception as e:
                    logger.warning(f"Error processing image {path}: {e}")
                    img = (
                        torch.zeros(3, 224, 224)
                        if self.image_size is None
                        else torch.zeros(3, self.image_size[0], self.image_size[1])
                    )
                    batch_imgs.append(img)

            batch_tensor = torch.stack(batch_imgs).to(self.device)

            with torch.no_grad():
                with torch.autocast(
                    device_type=comfy.model_management.get_torch_device().type, dtype=torch.float16
                ):
                    batch_descriptors = self.model(batch_tensor).cpu()

            descriptors.append(batch_descriptors)

        self.descriptors = torch.cat(descriptors)
        return self.descriptors

    def _apply_nms_filter(self, loop_closures, nms_threshold):
        """Apply Non-Maximum Suppression (NMS) filtering to loop pairs"""
        if not loop_closures or nms_threshold <= 0:
            return loop_closures

        sorted_loops = sorted(loop_closures, key=lambda x: x[2], reverse=True)
        filtered_loops = []
        suppressed = set()

        max_frame = max(max(idx1, idx2) for idx1, idx2, _ in loop_closures)

        for idx1, idx2, sim in sorted_loops:
            if idx1 in suppressed or idx2 in suppressed:
                continue

            filtered_loops.append((idx1, idx2, sim))

            suppress_range = set()

            start1 = max(0, idx1 - nms_threshold)
            end1 = min(idx1 + nms_threshold + 1, idx2)
            suppress_range.update(range(start1, end1))

            start2 = max(idx1 + 1, idx2 - nms_threshold)
            end2 = min(idx2 + nms_threshold + 1, max_frame + 1)
            suppress_range.update(range(start2, end2))

            suppressed.update(suppress_range)

        return filtered_loops

    def _ensure_decending_order(self, tuples_list):
        return [(max(a, b), min(a, b), score) for a, b, score in tuples_list]

    def find_loop_closures(self):
        """Find loop closures"""
        if self.descriptors is None:
            self.extract_descriptors()

        embed_size = self.descriptors.shape[1]
        faiss_index = faiss.IndexFlatIP(embed_size)

        normalized_descriptors = self.descriptors.numpy()
        faiss_index.add(normalized_descriptors)

        similarities, indices = faiss_index.search(
            normalized_descriptors, self.top_k + 1
        )  # +1 because self is most similar

        loop_closures = []
        for i in range(len(self.descriptors)):
            # Skip first result (self)
            for j in range(1, self.top_k + 1):
                neighbor_idx = indices[i, j]
                similarity = similarities[i, j]

                if similarity > self.similarity_threshold and abs(i - neighbor_idx) > 10:
                    if i < neighbor_idx:
                        loop_closures.append((i, neighbor_idx, similarity))
                    else:
                        loop_closures.append((neighbor_idx, i, similarity))

        loop_closures = list(set(loop_closures))
        loop_closures.sort(key=lambda x: x[2], reverse=True)

        if self.use_nms and self.nms_threshold > 0:
            loop_closures = self._apply_nms_filter(loop_closures, self.nms_threshold)

        self.loop_closures = self._ensure_decending_order(loop_closures)
        return self.loop_closures

    def save_results(self):
        """Save loop detection results to file"""
        if self.loop_closures is None:
            self.find_loop_closures()

        with open(self.output, "w") as f:
            f.write("# Loop Detection Results (index1, index2, similarity)\n")
            if self.use_nms:
                f.write(f"# NMS filtering applied, threshold: {self.nms_threshold}\n")
            f.write("\n# Loop pairs:\n")
            for i, j, sim in self.loop_closures:
                f.write(f"{i}, {j}, {sim:.4f}\n")
            f.write("\n# Image path list:\n")
            for i, path in enumerate(self.image_paths):
                f.write(f"# {i}: {path}\n")

        logger.info(f"Found {len(self.loop_closures)} loop pairs, results saved to {self.output}")
        if self.use_nms:
            logger.info(f"NMS filtering applied, threshold: {self.nms_threshold}")

        if self.loop_closures:
            logger.info("Top 10 loop pairs:")
            for i, (idx1, idx2, sim) in enumerate(self.loop_closures[:10]):
                logger.info(f"{idx1}, {idx2}, similarity: {sim:.4f}")
                if i >= 9:
                    break

    def get_loop_list(self):
        return [(idx1, idx2) for idx1, idx2, _ in self.loop_closures]

    def run(self):
        """Run complete loop detection pipeline"""
        logger.info("Loading model...")
        if self.model is None:
            self.load_model()

        self.get_image_paths()
        if not self.image_paths:
            logger.warning(f"No image files found in {self.image_dir}")
            return

        logger.info(f"Found {len(self.image_paths)} image files")

        self.extract_descriptors()

        self.find_loop_closures()

        self.save_results()

        return self.loop_closures
