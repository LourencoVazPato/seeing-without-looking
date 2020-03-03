import torch
import time
import json
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image


class Helper(object):

    def __init__(self, path_anns, img_folder=None, path_dets=None):
        self._load_annotations(path_anns)
        if path_dets:
            self._load_detections(path_dets)
        self.img_folder = img_folder
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def get_annotations(self, img_id):
        if img_id in self.annotations.keys():
            return self.annotations[img_id]
        return []

    def get_detections(self, img_id):
        if img_id in self.detections.keys():
            return self.detections[img_id]
        return []

    def get_reclassifications(self, img_id):
        if img_id in self.reclassifications.keys():
            return self.reclassifications[img_id]
        return []

    def convert_id(self, category_id):
        return self.category_index[category_id]

    def get_label(self, category_id):
        return self.categories[category_id]["name"]

    def load_image(self, image_id):
        assert image_id in self.images.keys(), "Error: Image id not found"
        filename = str(image_id).zfill(12)
        filename = os.path.join(self.img_folder, filename + ".jpg")
        im = Image.open(filename)
        if im.mode != "RGB":
            im = im.convert("RGB")
        W, H = self.images[image_id]["width"], self.images[image_id]["height"]
        return im, (W, H)

    def get_patch(self, im, bbox, pad=0, transform=None):
        x, y, w, h = bbox
        patch = im.crop((x - pad, y - pad, x + w + pad, y + h + pad))
        if transform is not None:
            return transform(patch).to(self.device)
        return patch.to(self.device)

    def plot_annotations(self, img_id, color='white'):
        im, _ = self.load_image(img_id)
        plt.figure(figsize=(10, 10))
        ax1 = plt.gca()
        ax1.imshow(im)
        for annot in self.get_annotations(img_id):
            x, y, w, h = annot["bbox"]
            label = self.get_label(annot["category_id"])
            ax1.add_patch(
                Rectangle(
                    (x, y), w - 1, h - 1, fill=None, linewidth=2, edgecolor=color
                )
            )
            plt.text(x + 2, y + 15, label, fontsize=15, color=color)
        plt.show()
        plt.tight_layout()

    def plot_detection_single(self, img_id, det, color='black'):
        im, _ = self.load_image(img_id)
        fig = plt.figure(figsize=(10, 10))
        ax2 = plt.gca()
        ax2.imshow(im)

        x, y, w, h = det["bbox"]
        label = det["category_id"]
        label = self.categories[label]["name"]

        ax2.add_patch(
            Rectangle(
                (x - 1, y - 1),
                w + 2,
                h + 2,
                fill=None,
                linewidth=6 * det["score"],
                edgecolor=color,
            )
        )
        plt.text(
            x + 40,
            y + 40,
            label + "\n" + str(round(det["score"], 2)),
            fontsize=22,
            color=color,
            # weight="bold",
        )
        
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        return fig

    def plot_detections(self, img_id, color='yellow', threshold=0.05):
        im, _ = self.load_image(img_id)
        fig = plt.figure(figsize=(9, 9))
        ax2 = plt.gca()
        ax2.imshow(im)
        for det in self.get_detections(img_id):
            if det["score"] >= threshold:
                x, y, w, h = det["bbox"]
                label = self.get_label(det["category_id"])
                ax2.add_patch(
                    Rectangle(
                        (x, y),
                        w - 1,
                        h - 1,
                        fill=None,
                        linewidth=2 * (det["score"]),
                        edgecolor=color,
                    )
                )
                if det["score"] >= 0.5:
                    plt.text(
                        x + 2,
                        y + 35,
                        label + "\n" + str(round(det["score"], 2)),
                        fontsize=18,
                        color=color,
                        weight="bold",
                    )
                else:
                    plt.text(
                        x + 2,
                        y + 35,
                        label + "\n" + str(round(det["score"], 2)),
                        fontsize=14,
                        color=color,
                    )
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        return fig

    def plot_reclassifications(self, img_id, color='yellow', threshold=0.05):
        im, _ = self.load_image(img_id)
        fig = plt.figure(figsize=(9, 9))
        ax2 = plt.gca()
        ax2.imshow(im)
        for det in self.get_reclassifications(img_id):
            if det["score"] >= threshold:
                x, y, w, h = det["bbox"]
                label = self.get_label(det["category_id"])
                ax2.add_patch(
                    Rectangle(
                        (x, y),
                        w - 1,
                        h - 1,
                        fill=None,
                        linewidth=2 * (det["score"]),
                        edgecolor=color,
                    )
                )
                if det["score"] >= 0.5:
                    plt.text(
                        x + 2,
                        y + 35,
                        label + "\n" + str(round(det["score"], 2)),
                        fontsize=18,
                        color=color,
                        weight="bold",
                    )
                else:
                    plt.text(
                        x + 2,
                        y + 30,
                        label + "\n" + str(round(det["score"], 2)),
                        fontsize=14,
                        color=color,
                    )

        plt.tight_layout()
        plt.axis('off')
        plt.show()
        return fig

    def _load_annotations(self, path_anns):
        # Load categories and make a list of category ids because mmdet uses a different id schema
        start = time.time()
        with open(path_anns) as json_file:
            data = json.load(json_file)
        categories = data["categories"]
        self.super_categories = {cat["id"]: cat["supercategory"] for cat in categories}
        self.categories = {cat["id"]: cat for cat in categories}
        self.category_index = list(self.categories.keys())

        # Load images and annotations
        imgs = data["images"]
        self.images = dict()
        for img in imgs:
            img.pop("license", None)
            img.pop("date_captured", None)
            img.pop("flickr_url", None)
            self.images[img["id"]] = img

        if "annotations" in data.keys():
            anns = data["annotations"]
            self.annotations = dict()
            for ann in anns:
                if ann["image_id"] not in self.annotations.keys():
                    self.annotations[ann["image_id"]] = list()
                ann.pop("segmentation", None)
                ann.pop("area", None)
                self.annotations[ann["image_id"]].append(ann)
            print("Loaded annotations (t=%.1fs)" % (time.time() - start))

    def _load_detections(self, path_dets):
        # Load detections from json file with COCO results format
        start = time.time()
        with open(path_dets) as json_file:
            dets = json.load(json_file)

        self.detections = dict()
        for det in dets:
            id_ = det["image_id"]
            if id_ not in self.detections.keys():
                self.detections[id_] = list()
            self.detections[id_].append(det)
        print("Loaded detections (t=%.1fs)" % (time.time() - start))

    def convert_idx2id_image(self, image_id):
        # Converts category indexes (0-79) to category id (1-90) for a single image
        ids = list(self.categories.keys())
        for det in self.detections[image_id]:
            det['category_id'] = ids[det['category_id']]

    def convert_idx2id(self):
        # Converts category indexes (0-79) to category id's (1-90) for all images
        ids = list(self.categories.keys())
        for img_id, dets in self.detections.items():
            for det in dets:
                det['category_id'] = ids[det['category_id']]
    
    def convert_id2idx(self):
        # Converts category id (1-90) to category index (0-79)
        cats = list(self.categories.keys())
        for img_id, dets in self.detections.items():
            for det in dets:
                det['category_id'] = cats.index(det['category_id']) 

    def load_reclassifications(self, path_dets):
        # Load reclassifications from COCO results format json file
        start = time.time()
        with open(path_dets) as json_file:
            dets = json.load(json_file)

        self.reclassifications = dict()
        for det in dets:
            id_ = det["image_id"]
            if id_ not in self.reclassifications.keys():
                self.reclassifications[id_] = list()
            self.reclassifications[id_].append(det)
        print("Loaded reclassifications (t=%.1fs)" % (time.time() - start))
