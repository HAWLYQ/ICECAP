import argparse
from PIL import Image
import sys
import os


def resize_image(image, size):
    """Resize an image to the given size."""
    return image.resize(size, Image.ANTIALIAS)


def resize_images(root_dir, output_dir, size):
    """Resize the images in 'image_dir' and save into 'output_dir'."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    images = []
    # images = os.listdir(image_dir)
    for month in os.listdir(root_dir):
        for day in os.listdir(root_dir + '/' + month):
            for image in os.listdir(root_dir + '/' + month + '/' + day + '/image'):
                # print root_path+'//'+ month + '//' + day
                images.append(root_dir + '/' + month + '/' + day + '/image' + '/' + image)
    num_images = len(images)
    print(num_images)
    truncated_writer = open('breakingnews_truncated.txt', 'w')
    truncated_num = 0
    for i, image in enumerate(images):
        with open(os.path.join(image), 'r+b') as f:
            # if not os.path.exists(os.path.join(output_dir, image)):
                try:
                    with Image.open(f) as img:
                        img = resize_image(img, size)
                        img = img.convert('RGB')
                        img.save(os.path.join(output_dir, image.split('/')[-1]), img.format)
                except IOError as e:
                    truncated_num += 1
                    truncated_writer.write(image + '\n')
        sys.stdout.write(
            '\r Processing images: %d/%d images processed...truncated images: %d' % (i, num_images, truncated_num))
        sys.stdout.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_size', type=int, default=256,
                        help='which size an image will be resized to')
    parser.add_argument('--root', type=str, default="../",
                        help='which size an image will be resized to')
    args = parser.parse_args()
    image_size = [args.img_size, args.img_size]
    root_path = '/root/BreakingNews/BreakingNewsDataset_v1.2_OriginalImages/data'
    resize_images(root_path, args.root + "breakingnews_resized/", size=image_size)
