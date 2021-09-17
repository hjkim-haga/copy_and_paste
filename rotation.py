import numpy as np
from PIL import Image
from PIL import ImageOps


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
    translation: tuple  # 회전 후 이미지의 원점 좌표.
    
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
    sample_image_path = 'assets/airconditioner_0001418.jpg'
    sample_image: Image.Image = Image.open(sample_image_path)
    bbox = (351, 166, 3996, 2416)
    
    exif = sample_image.getexif()
    print(f'rotation: {exif}')
    print(f'before rotation: {bbox}')
    
    # Code snippet for rotate image according to its EXIF tag.
    rotated_image: Image.Image = ImageOps.exif_transpose(sample_image)
    rot_bbox = rotate_bbox(bbox, exif[274], (sample_image.width, sample_image.height))
    print(f'after rotation: {rot_bbox}')
    
    # rotated_image.save('rotated.jpg')
    
    # test:
    # ccw 90 (ok), ccw 0 (ok), ccw 180 (ok), ccw 270 (ok).
