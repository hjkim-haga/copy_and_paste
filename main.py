import argparse
import enum
import math
import os
import pprint
import random
import time
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
from PIL import ImageOps


class BoundViolation(enum.Enum):
    LEFT = 0
    UPPER = 1
    RIGHT = 2
    LOWER = 3


class Pastable(enum.Enum):
    FALSE = -1


pp = pprint.PrettyPrinter()

dataset = 'valid'
parser = argparse.ArgumentParser(description='Copy and paste objects in a image.')
parser.add_argument('--image_dir', type=str,
                    default=f'/home/ubuntu/haga-dataset/electronics/{dataset}_original_size/jpegs',
                    help='Where JPEG images are.')
parser.add_argument('--annot_dir', type=str,
                    default=f'/home/ubuntu/haga-dataset/electronics/{dataset}_original_size/xmls',
                    help='Where PASCAL VOC XML annotations are.')
parser.add_argument('--out_dir', type=str,
                    default=f'{dataset}_batch',
                    help='Where the pasted images and annotations will be placed.')
parser.add_argument('--threshold', type=float,
                    default=0.1,
                    help='IoU threshold where 0 <= thres < 1. The smaller, the stricter.')
args = parser.parse_args()


# Completed
def open_rotated_image_annot(image_path: str, annot_path: str):
    """Open image and annotation file.
    
    **Caution: This function loads an image already applying the EXIF
    orientation. The sight that picture-taker has seen.

    Args:
        image_path: An image full path.
        annot_path: An annotation in Pascal VOC format (xml) file.

    Returns:
        Image.Image, dict: image and annotation.

    """
    # Rotate the image.
    image: Image.Image = Image.open(image_path)
    exif = dict(image.getexif().items())
    try:
        orientation = exif[274]
    except KeyError:
        orientation = 0
    rot_image = ImageOps.exif_transpose(image)
    
    # Rotate the bbox.
    annot = read_pascal_voc(annot_path)
    bbox = (annot['xmin'], annot['ymin'], annot['xmax'], annot['ymax'])
    rot_bbox = rotate_bbox(bbox, orientation, (image.width, image.height))
    
    annot['xmin'] = rot_bbox[0]
    annot['ymin'] = rot_bbox[1]
    annot['xmax'] = rot_bbox[2]
    annot['ymax'] = rot_bbox[3]
    
    return rot_image, annot


# Completed
def read_pascal_voc(path: str) -> dict:
    # Parse the xml
    tree = ET.parse(os.path.abspath(path))
    root = tree.getroot()
    size_tag = root.find('size')
    object_tag = root.find('object')
    bndbox_tag = object_tag.find('bndbox')
    
    # Get the annotation
    width = int(size_tag.find('width').text)
    height = int(size_tag.find('height').text)
    name = object_tag.find('name').text
    xmin = float(bndbox_tag.find('xmin').text)
    ymin = float(bndbox_tag.find('ymin').text)
    xmax = float(bndbox_tag.find('xmax').text)
    ymax = float(bndbox_tag.find('ymax').text)
    
    return {  # bbox not normalized
        'filename': os.path.basename(path),
        'name': name,
        'width': width,
        'height': height,
        'xmin': xmin,
        'ymin': ymin,
        'xmax': xmax,
        'ymax': ymax,
    }


# Completed
def time_counter(task):
    """It measure time in sec for the given task.

    Args:
        task: task function. It is the target to be measured for time.

    Returns:
        A wrapper function.

    Notes:
    This function has the side-effect that prints the elapsed time
    for the task.

    """
    
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = task(*args, **kwargs)
        end_time = time.time()
        print(f'The elapsed time for the task is {end_time - start_time} sec.')
        return result
    
    return wrapper


# Completed
def determine_batch_sizes(total: int, lower: int, upper: int):
    """Determine the sizes of batches in the given range.

    collection에 원소 수가 total개 있다면,
    lower <= batch_size <= upper 범위 batch들을 나열한다.

    Args:
        total: sum(batch_sizes) == total.
        lower: Shoule be batch_size >= lower.
        upper: Shoule be batch_size <= upper.

    Returns:
        list: A list with the batch sizes.

    """
    result = []
    while sum(result) < total:
        batch_size = random.randint(lower, upper)
        result.append(batch_size)
    adjusted_last_batch_size = total - sum(result[:-1])
    result[-1] = adjusted_last_batch_size
    return result


# Completed
def split_batch(sequence: list, batch_sizes: list):
    """Split files into batches.

    Args:
        sequence: A collection to be split by the batch sizes.
        batch_sizes: Each element of this variable means the batch size.

    Returns:
        list: A list of tuples. The sub-tuples are the batches.

    """
    result = []
    
    next_index = 0
    for batch_size in batch_sizes:
        batch = sequence[next_index:next_index + batch_size]
        next_index += batch_size
        result.append(batch)
    
    return result


# Completed
def safe_paste(paste_pos: tuple, image_size: tuple, box_size: tuple):
    """Calculate the box coordinate based on the boundary condition.
    
    This function ensures for the new box coordinate to be within the image.
    It first checks that the box is out of the image and cropped the box
    only if the condition is true. Otherwise, the position is as it is.
    
    Args:
        paste_pos: left upper coordinate represented by (x0, y0).
        image_size: (img_width, img_height).
        box_size: (box_width, box_height).

    Returns:
        tuple: The new box coordinate (x0, y0, x1, y1).

    """
    
    x0, y0 = paste_pos
    w, h = box_size
    result = [x0, y0, x0 + w, y0 + h]
    
    def check_boundary_conditions(boundary: tuple, coordinate: tuple):
        # Return a tuple shows Which boundary condition is broken.
        # (x0_valid, y0_valid, x1_valid, y1_valid)
        result: list = [True, True, True, True]
        x0, y0, x1, y1 = coordinate
        width, height = boundary
        
        # Define the 4-directional conditions.
        left_condition = x0 >= 0
        upper_condition = y0 >= 0
        right_condition = x1 < width
        lower_condition = y1 < height
        
        # Put them in a check loop for simplicity.
        boundary_conditions = (left_condition, upper_condition,
                               right_condition, lower_condition)
        for i, cond in enumerate(boundary_conditions):
            result[i] = cond
        
        return tuple(result)
    
    # Iterate over the boundary condition and
    for direction_index, cond_check_result in enumerate(
            check_boundary_conditions(image_size, result)):
        # calibrate the box coordinate if the condition is violated.
        if not cond_check_result:
            if direction_index == BoundViolation.LEFT.value:
                result[direction_index] = 1
            elif direction_index == BoundViolation.UPPER.value:
                result[direction_index] = 1
            elif direction_index == BoundViolation.RIGHT.value:
                result[direction_index] = image_size[0] - 1
            elif direction_index == BoundViolation.LOWER.value:
                result[direction_index] = image_size[1] - 1
    
    return result


# Completed
def append_object_in_pascal_annotation(obj_annot: dict,
                                       target_annot_tree: ET):
    """Append the object annotation to the VOC PASCAL annotation.
    
    Args:
        obj_annot: This annotation has the same format with return value of
            `read_pascal_voc`.
        target_annot_tree:

    Returns:
        xml.etree.ElementTree: The tree that the object annotation is
        appended to.
    """
    # Define elements.
    object_tag = ET.Element('object')
    name_tag = ET.Element('name')
    pose_tag = ET.Element('pose')
    truncated_tag = ET.Element('truncated')
    difficult_tag = ET.Element('difficult')
    bndbox_tag = ET.Element('bndbox')
    xmin_tag = ET.Element('xmin')
    xmax_tag = ET.Element('xmax')
    ymin_tag = ET.Element('ymin')
    ymax_tag = ET.Element('ymax')
    
    # Insert values.
    name_tag.text = obj_annot['name']
    pose_tag.text = 'Unspecified'
    truncated_tag.text = str(0)
    difficult_tag.text = str(0)
    xmin_tag.text = str(obj_annot['xmin'])
    ymin_tag.text = str(obj_annot['ymin'])
    xmax_tag.text = str(obj_annot['xmax'])
    ymax_tag.text = str(obj_annot['ymax'])
    
    # Give structure to the created elements.
    bndbox_tag.append(xmin_tag)
    bndbox_tag.append(ymin_tag)
    bndbox_tag.append(xmax_tag)
    bndbox_tag.append(ymax_tag)
    
    object_tag.append(name_tag)
    object_tag.append(pose_tag)
    object_tag.append(truncated_tag)
    object_tag.append(difficult_tag)
    object_tag.append(bndbox_tag)
    
    # Append the new object node to the annotation.
    root = target_annot_tree.getroot()
    root.append(object_tag)
    return target_annot_tree


# Completed.
def compute_iou(bbox1: tuple, bbox2: tuple):
    """Compute IoU between two boxes.
    
    Args:
        bbox1: (x0, y0, x1, y1).
        bbox2: (x0, y0, x1, y1).

    Returns:
        float: IoU value.
    """
    intersection: int
    union: int
    
    # Set positions, which makes the IoU calculation simple.
    left_box, right_box = sorted([bbox1, bbox2], key=lambda x: x[0])
    upper_box, lower_box = sorted([bbox1, bbox2], key=lambda x: x[1])
    # The area of bboxes.
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    
    # Calculate the intersection and union,
    # when there is not overlapped area between two boxes.
    if left_box[2] <= right_box[0] or upper_box[3] <= lower_box[1]:
        intersection = 0
        union = area1 + area2
    # when there is overlapped area.
    else:
        width_overlap = left_box[2] - right_box[0]
        height_overlap = upper_box[3] - lower_box[1]
        intersection = width_overlap * height_overlap
        union = area1 + area2 - intersection
    return float(intersection) / float(union)


def stride_from_center(image_size: tuple,
                       incoming_object: tuple,
                       already_pasted_objects: list,
                       stride_size: int, threshold: float):
    """Find the place to put object while striding over the image.

    The idea is from the anchor box in the object detection field.
    An anchor box strides over an image by the given step size.
    The object bounding box becomes the anchor box in the function.

    Args:
        image_size: A tuple of (width, height).
        incoming_object: a tuple of (width, height).
        already_pasted_objects: An object list whose element is a tuple
            (xmin, ymin, xmax, ymax). We assume that this list is sorted by
                    1. The x-axis: lambda x: x[0] (x0),
                    2. The y-axis: lambda x: x[1] (y0).
            for time complexity.
        stride_size: Horizontal and vertical step size.
        threshold: The pasting position is not proper if IoU(target_bbox, other_bbox)
            is over this threshold.

    Returns:
        tuple: First met placable coordinate (x0, y0) for the incoming object.

    """
    obj_width = incoming_object[0]
    obj_height = incoming_object[1]
    
    img_width = image_size[0]
    img_height = image_size[1]
    
    cen_x = img_width // 2
    cen_y = img_height // 2
    
    # Continue striding until we find the first pastable position.
    for abs_offset_x in range(0, cen_x, stride_size):
        for abs_offset_y in range(0, cen_y, stride_size):
            # I don't like it.
            for offset_x in [-1 * abs_offset_x, abs_offset_x]:
                for offset_y in [-1 * abs_offset_y, abs_offset_y]:
                    cur_x_pos = cen_x + offset_x
                    cur_y_pos = cen_y + offset_y
                    bbox_to_paste = (
                        cur_x_pos, cur_y_pos,
                        cur_x_pos + obj_width, cur_y_pos + obj_height)
                    
                    # Skip if IoU(bbox_to_paste, other_bbox) >= threshold.
                    all_clear = []  # Shoulbe be all True.
                    for other_object in already_pasted_objects:
                        # Paste if we find the pastable position.
                        if compute_iou(bbox_to_paste, other_object) <= threshold:
                            pastable = True
                        else:
                            pastable = False
                        all_clear.append(pastable)
                    if all(all_clear):
                        return cur_x_pos, cur_y_pos
    
    # Any of the position is not proper.
    return Pastable.FALSE, Pastable.FALSE


# Completed.
def stride(image_size: tuple,
           incoming_object: tuple,
           already_pasted_objects: list,
           stride_size: int, threshold: float):
    """Find the place to put object while striding over the image.
    
    The idea is from the anchor box in the object detection field.
    An anchor box strides over an image by the given step size.
    The object bounding box becomes the anchor box in the function.
    
    Args:
        image_size: A tuple of (width, height).
        incoming_object: a tuple of (width, height).
        already_pasted_objects: An object list whose element is a tuple
            (xmin, ymin, xmax, ymax). We assume that this list is sorted by
                    1. The x-axis: lambda x: x[0] (x0),
                    2. The y-axis: lambda x: x[1] (y0).
            for time complexity.
        stride_size: Horizontal and vertical step size.
        threshold: The pasting position is not proper if IoU(target_bbox, other_bbox)
            is over this threshold.

    Returns:
        tuple: First met placable coordinate (x0, y0) for the incoming object.
    
    """
    obj_width = incoming_object[0]
    obj_height = incoming_object[1]
    
    img_width = image_size[0]
    img_height = image_size[1]
    # Continue striding until we find the first pastable position.
    for cur_x_pos in range(0, img_width - obj_width, stride_size):
        for cur_y_pos in range(0, img_height - obj_height, stride_size):
            bbox_to_paste = (cur_x_pos, cur_y_pos,
                             cur_x_pos + obj_width, cur_y_pos + obj_height)
            
            # Skip if IoU(bbox_to_paste, other_bbox) >= threshold.
            all_clear = []  # Shoulbe be all True.
            for other_object in already_pasted_objects:
                # Paste if we find the pastable position.
                if compute_iou(bbox_to_paste, other_object) <= threshold:
                    pastable = True
                else:
                    pastable = False
                all_clear.append(pastable)
            if all(all_clear):
                return cur_x_pos, cur_y_pos
    
    # Any of the position is not proper.
    return Pastable.FALSE, Pastable.FALSE


def rotate_bbox(bbox: tuple, rotation: int, image_size: tuple) -> tuple:
    """

    Args:
        bbox: (xmin, ymin, xmax, ymax)
        rotation:
            1: no rotation.
            6: CCW  90 deg.
            3: CCW 180 deg.
            8: CCW 270 deg.
        image_size: (width, height)

    Returns:
        tuple: Rotated bbox coordinate.
    """
    left_upper = np.array([bbox[0], bbox[1]])
    right_lower = np.array([bbox[2], bbox[3]])
    
    img_width = image_size[0]
    img_height = image_size[1]
    
    rot_angle: int
    translation: tuple  # 회전 후 직선 이동해야 할 값.
    
    if rotation == 1:
        rot_angle = np.deg2rad(0)
        translation = (0, 0, 0, 0)
    elif rotation == 6:
        rot_angle = np.deg2rad(90)
        translation = (img_height, 0, img_height, 0)
    elif rotation == 3:
        rot_angle = np.deg2rad(180)
        translation = (img_width, img_height, img_width, img_height)
    elif rotation == 8:
        rot_angle = np.deg2rad(270)
        translation = (0, img_width, 0, img_width)
    else:
        return bbox
    
    # Rotate,
    rot_matrix = np.array([
        [np.cos(rot_angle), -np.sin(rot_angle)],
        [np.sin(rot_angle), np.cos(rot_angle)],
    ])
    rot_left_upper = np.dot(rot_matrix, np.array(left_upper))
    rot_right_lower = np.dot(rot_matrix, np.array(right_lower))
    rot_bbox = np.array([*rot_left_upper, *rot_right_lower])
    
    # then translate.
    rot_bbox += np.array(translation)
    
    # left-upper, right-lower를 찾는다.
    if rotation == 8:
        xmin, ymax, xmax, ymin = rot_bbox[0], rot_bbox[1], rot_bbox[2], rot_bbox[3]
        rot_bbox = (xmin, ymin, xmax, ymax)
    elif rotation == 6:
        xmax, ymin, xmin, ymax = rot_bbox[0], rot_bbox[1], rot_bbox[2], rot_bbox[3]
        rot_bbox = (xmin, ymin, xmax, ymax)
    elif rotation == 3:
        xmax, ymax, xmin, ymin = rot_bbox[0], rot_bbox[1], rot_bbox[2], rot_bbox[3]
        rot_bbox = (xmin, ymin, xmax, ymax)
    
    return tuple(rot_bbox)


if __name__ == '__main__':
    # 회전 없이 원본 사진에서 물체가 점하던 방향 그대로를 유지할 것.
    # arguments that determine the copy_and_paste.
    image_dir: str = args.image_dir
    annot_dir: str = args.annot_dir
    out_dir: str = args.out_dir
    threshold: float = args.threshold
    min_batch_size = 4
    max_batch_size = 10
    
    # Randomly list file names without extension.
    base_file_names = [os.path.splitext(f)[0] for f in os.listdir(annot_dir) if
                       os.path.splitext(f)[1] == '.xml']
    random.shuffle(base_file_names)
    
    # Get the batch sizes which is the number of objects in one image.
    num_files = len(base_file_names)
    batch_sizes = determine_batch_sizes(
        num_files, min_batch_size, max_batch_size)
    
    # Make the actual batches by the size given from the above.
    batches = split_batch(base_file_names, batch_sizes)
    
    # Files related to the background has the prefix 'bg_'.
    # Files related to the other objects has the prefix 'obj_'.
    for batch_idx, batch in enumerate(batches):
        # Make a list about objects in the batch. This is used in the `stride`
        # function.
        objects_in_batch = []
        
        # Decide the random background image from the batch.
        bg_index = 0  # random.randint(0, len(batch) - 1)
        bg_image_path = os.path.join(image_dir, batch[bg_index] + '.jpg')
        bg_annot_path = os.path.join(annot_dir, batch[bg_index] + '.xml')
        
        # Get the background image and annotation.
        bg_tree: ET = ET.parse(os.path.abspath(bg_annot_path))
        bg_image: Image.Image
        bg_annot: dict
        bg_image, bg_annot = open_rotated_image_annot(bg_image_path, bg_annot_path)
        bg_bbox = (bg_annot['xmin'], bg_annot['ymin'],
                   bg_annot['xmax'], bg_annot['ymax'])
        stride_size = int(math.sqrt(bg_image.width ** 2 + bg_image.height ** 2) / 10)
        
        # Include the object in the background.
        objects_in_batch.append(bg_bbox)
        
        # Get the objects from the rest part of the batch.
        batch_without_bg = batch[:bg_index] + batch[bg_index + 1:]
        
        for obj_basename in batch_without_bg:
            # Get the object image and annotation.
            obj_image_path = os.path.join(image_dir, obj_basename + '.jpg')
            obj_annot_path = os.path.join(annot_dir, obj_basename + '.xml')
            
            obj_image: Image.Image
            obj_annot: dict
            obj_image, obj_annot = open_rotated_image_annot(obj_image_path, obj_annot_path)
            obj_bbox = (obj_annot['xmin'], obj_annot['ymin'],
                        obj_annot['xmax'], obj_annot['ymax'])
            obj_width = int(obj_bbox[2]) - int(obj_bbox[0])
            obj_height = int(obj_bbox[3]) - int(obj_bbox[1])
            
            # Crop the image around the bbox.
            cropped_image: Image.Image = obj_image.crop(obj_bbox)
            
            # Resize the object image.
            # TODO (me): Decide how calculate the factor.
            scale_factor = bg_image.width / obj_image.width
            random_scale_factor = random.uniform(0.5, scale_factor)
            reduced_width, reduced_height = \
                int(obj_width * random_scale_factor), \
                int(obj_height * random_scale_factor)
            reduced_cropped_image = cropped_image.resize(
                (reduced_width, reduced_height))
            
            # Stride over the background finding the pastable position.
            new_x0, new_y0 = stride_from_center(
                (bg_image.width, bg_image.height),
                (reduced_cropped_image.width, reduced_cropped_image.height),
                objects_in_batch,
                stride_size,
                threshold)
            # Skip if not pastable.
            if new_x0 == Pastable.FALSE:
                continue
            
            # Calibrate the coordinate,
            calibrated_coordinate = safe_paste(
                (new_x0, new_y0),
                (bg_image.width, bg_image.height),
                (reduced_cropped_image.width, reduced_cropped_image.height))
            
            # and box size if needed. The below code distorts the object image.
            # new_width = calibrated_coordinate[2] - calibrated_coordinate[0]
            # new_height = calibrated_coordinate[3] - calibrated_coordinate[1]
            # reduced_cropped_image = reduced_cropped_image.resize(
            #     (new_width, new_height))
            
            # Paste the cropped image on the background.
            bg_image.paste(reduced_cropped_image, (new_x0, new_y0))
            
            # Include the object position on the background.
            objects_in_batch.append(
                (new_x0, new_y0,
                 new_x0 + reduced_cropped_image.width,
                 new_y0 + reduced_cropped_image.height)
            )
            objects_in_batch.sort(key=lambda x: (x[0], x[2]))
            
            # Append the object annotation modified based on the calibrated
            # bndbox to the annotation tree.
            obj_annot['xmin'] = calibrated_coordinate[0]
            obj_annot['ymin'] = calibrated_coordinate[1]
            obj_annot['xmax'] = calibrated_coordinate[2]
            obj_annot['ymax'] = calibrated_coordinate[3]
            bg_tree = append_object_in_pascal_annotation(obj_annot, bg_tree)
        
        # The Process ended for the each objects. Now, Save the image where
        # the object have pasted and annotation file appended the objects.
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        bg_image.save(f'{out_dir}/electronics_{batch_idx:>07}.jpg')
        # The `filename` in the annotation file is required when creating the
        # record file fed in during training.
        out_filename = f'electronics_{batch_idx:>07}.xml'
        bg_tree.getroot().find('filename').text = out_filename
        bg_tree.write(f'{out_dir}/{out_filename}')
        print(f'batch {batch_idx} in {len(batches) - 1} processed.')
