import os
from PIL import Image

working_dir = '/home/ubuntu/haga-dataset/electronics/train_original_size/jpegs'
image_files = [os.path.join(working_dir, image_name) for image_name in os.listdir(working_dir)]

orientations = set()
for i, image_path in enumerate(image_files):
    image: Image.Image = Image.open(image_path)
    exif = image.getexif()
    try:
        orientations.add(exif[274])
    except:
        ""
    finally:
        image.close()
    print(f'{i} th image.')
print(orientations)