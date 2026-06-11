import cv2
import numpy as np
from PIL import Image
import supervision as sv
from matplotlib import pyplot as plt

from tiptop.perception.bbox_utils import bbox_to_pixel_xyxy, box_2d_to_pixel_xyxy


def match_masks_to_bboxes(masks, bboxes, img_width, img_height):
    """Match masks to bounding boxes using IoU.

    Args:
        masks: Boolean masks of shape (N, H, W)
        bboxes: List of bounding box dictionaries
        img_width: Width of the image
        img_height: Height of the image

    Returns:
        List of mask indices that best match each bounding box
    """
    # Extract 2D bounding boxes from the detection results
    bbox_coords = []
    for bbox in bboxes:
        if "box_2d" in bbox and len(bbox["box_2d"]) == 4:
            coords = box_2d_to_pixel_xyxy(bbox["box_2d"], img_width, img_height)
            if coords is None:
                bbox_coords.append([0, 0, 10, 10])
            else:
                x1, y1, x2, y2 = coords
                bbox_coords.append([x1, y1, x2, y2])
        else:
            print(f"Invalid bbox format for {bbox.get('label', 'unknown')}")
            bbox_coords.append([0, 0, 10, 10])

    # Calculate IoU between each mask and each bbox to find best matches
    best_mask_indices = []

    for bbox_idx, bbox_coord in enumerate(bbox_coords):
        x_min, y_min, x_max, y_max = bbox_coord
        # Create a binary mask for the bbox
        bbox_mask = np.zeros((masks.shape[1], masks.shape[2]), dtype=bool)
        bbox_mask[y_min:y_max, x_min:x_max] = True

        # Calculate IoU for each mask with this bbox
        best_iou = -1
        best_idx = -1

        for mask_idx, mask in enumerate(masks):
            # Skip if this mask has already been assigned
            if mask_idx in best_mask_indices:
                continue

            # Calculate intersection and union
            intersection = np.logical_and(mask, bbox_mask).sum()
            union = np.logical_or(mask, bbox_mask).sum()
            iou = intersection / union if union > 0 else 0

            # Update if better IoU is found
            if iou > best_iou:
                best_iou = iou
                best_idx = mask_idx

        if best_idx >= 0:
            best_mask_indices.append(best_idx)
        else:
            # If no good match is found (all masks already assigned),
            # use any unassigned mask
            remaining_indices = [i for i in range(len(masks)) if i not in best_mask_indices]
            if remaining_indices:
                best_mask_indices.append(remaining_indices[0])
            else:
                # No masks left
                print(f"No mask available for bbox {bbox_idx}")
                # Use first mask as fallback (might be already used, but we need something)
                best_mask_indices.append(0)

    return best_mask_indices


def visualize_detections(
    image: Image.Image, results: list[dict], output_path: str | None = None, show_plot: bool = False
) -> np.ndarray | None:
    if not results:
        print("No results to visualize")
        return None

    img_array = np.array(image).copy()
    img_height, img_width = img_array.shape[:2]
    colors = plt.cm.tab20(np.linspace(0, 1, len(results)))

    for i, item in enumerate(results):
        box = item.get("box_2d", [])
        label = item.get("label", "Unknown")
        coords = bbox_to_pixel_xyxy(item, img_width, img_height)
        if coords is None:
            if len(box) != 4:
                continue
            coords = box_2d_to_pixel_xyxy(box, img_width, img_height)
        if coords is None:
            continue
        x1, y1, x2, y2 = coords
        color = tuple(int(c * 255) for c in colors[i][:3])
        cv2.rectangle(img_array, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            img_array,
            f"{i + 1}. {label}",
            (x1, max(15, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    if output_path:
        cv2.imwrite(output_path, cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))

    if show_plot:
        plt.imshow(img_array)
        plt.axis("off")
        plt.show()

    return img_array


def visualize_masks(rgb_pil: Image.Image, masks: np.ndarray, bboxes: list[dict]) -> np.ndarray:
    """Visualize segmentation masks on an image.

    Args:
        rgb_pil: Input RGB image
        masks: Segmentation masks of shape (N, 1, H, W)
        bboxes: List of bounding box dictionaries

    Returns:
        Annotated image as numpy array
    """
    masks_sv = masks.squeeze(1).astype(bool)  # (num_masks, H, W)

    # Handle case where we have more masks than bounding boxes
    if masks_sv.shape[0] > len(bboxes):
        print(f"More masks ({masks_sv.shape[0]}) than bounding boxes ({len(bboxes)}). Using IoU matching.")
        # Match masks to bboxes using IoU
        best_mask_indices = match_masks_to_bboxes(masks_sv, bboxes, rgb_pil.width, rgb_pil.height)
        # Keep only the matched masks
        masks_sv = masks_sv[best_mask_indices]

    # Handle case where we have more bounding boxes than masks
    if len(bboxes) > masks_sv.shape[0]:
        print(
            f"More bounding boxes ({len(bboxes)}) than masks ({masks_sv.shape[0]}). Some objects won't be visualized."
        )
        # Truncate the bboxes list
        bboxes = bboxes[: masks_sv.shape[0]]

    # Extract bounding boxes from masks
    xyxy = []
    for mask in masks_sv:
        coords = np.column_stack(np.where(mask))
        if len(coords) > 0:
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            xyxy.append([x_min, y_min, x_max, y_max])
        else:
            xyxy.append([0, 0, 0, 0])
    xyxy = np.array(xyxy)

    # Now the number of masks matches the number of bboxes
    detections = sv.Detections(xyxy=xyxy, mask=masks_sv, class_id=np.arange(len(bboxes)))
    labels = [bbox["label"] for bbox in bboxes]

    mask_annotator = sv.MaskAnnotator()
    label_annotator = sv.LabelAnnotator(text_position=sv.Position.TOP_LEFT)

    rgb_np = np.array(rgb_pil)
    annotated_image = mask_annotator.annotate(scene=rgb_np.copy(), detections=detections)
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
    return annotated_image
