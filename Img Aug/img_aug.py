import os
from os import listdir
import cv2
import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa
import imageio

ia.seed(1)

dir = './imgfile/'

# txt파일에서 bounding box 읽어오기
def read_annotation(txtfile):
    bounding_box = []
    r = open(txtfile, mode='r')
    for box in r.readlines():
        bounding_box.append(list(map(float, box.strip('\n').split())))
        print(box)
    r.close()
    return bounding_box

def hangulFilePathImageRead ( filePath ) :
    stream = open(filePath.encode("utf-8"), "rb")
    bytes = bytearray(stream.read())
    numpyArray = np.asarray(bytes, dtype=np.uint8)
    return cv2.imdecode(numpyArray, cv2.IMREAD_UNCHANGED)

# 바운딩 박스 잡힌것만 파악해오기
def read_dataset(dir):
    images = []
    annotations = []
    bounding_box = []
    delete_list = []
    hwc = []

    for file in listdir(dir):
        if 'jpg' in file.lower():
            if os.path.getsize(dir + file.replace(file.split('.')[-1], 'txt')) == 0:
                delete_list.extend([file, file.replace(file.split('.')[-1], 'txt')])

            else:
                filedata = hangulFilePathImageRead(dir + file)
                # image_cv = cv2.imread(dir+file)
                hwc.append(filedata.shape) #height,weight,channels
                images.append(filedata)
                bounding_box.append(read_annotation(dir + file.replace(file.split('.')[-1], 'txt')))
                annotations.append(file.replace(file.split('.')[-1], '').replace('.', ''))
        return images, bounding_box, hwc, annotations

    # 만약 빈 텍스트 파일이 있으면 (바운딩박스가 잘 안잡혔으면) 삭제
    if delete_list:
        for i in delete_list:
            os.remove(dir+i)

images, bounding_box, hwc, annotations = read_dataset(dir)


def restore(width, height, x1, y1, x2, y2):
    bbox_center_x = (x1 + x2) / 2
    bbox_center_y = (y1 + y2) / 2
    bbox_width = (bbox_center_x - x1) * 2
    bbox_height = (bbox_center_y - y1) * 2
    return round(bbox_center_x/width, 6), round(bbox_center_y/height,6), round(bbox_width/width,6), round(bbox_height/height,6)

def save_imgbox(filename, transform, bbs, image_aug, bbs_aug):
    to_txt = []
    for i in range(len(bbs.bounding_boxes)):
        before = bbs.bounding_boxes[i]
        after = bbs_aug.bounding_boxes[i]
        print(restore(width, height, before.x1, before.y1, before.x2, before.y2))
        print(restore(width, height, after.x1, after.y1, after.x2, after.y2))
        print("label : %d", int(label[i]))
        to_txt.append([int(label[i]), *restore(width, height, after.x1, after.y1, after.x2, after.y2)])

    imageio.imwrite('./imgaugfile/'+filename+transform+'.jpg',image_aug)
    with open('./imgaugfile/'+filename+transform+'.txt', 'w') as file:  # hello.txt 파일을 쓰기 모드(w)로 열기
        for i in to_txt:
            line = str(i)
            line = line[1:-1].replace(',','')
            file.writelines(line)
    print(to_txt)

bbox = []

for idx in range(len(images)):
    image = images[idx]
    ia_bounding_boxes = []
    boxes = bounding_box[idx]
    filename = annotations[idx]
    height, width, channel = hwc[idx]
    label = []

    for box in boxes:
        label.append(box[0])
        bbox_center_x = box[1] * width
        bbox_center_y = box[2]* height
        bbox_width = box[3] * width
        bbox_height = box[4] * height
        bbox_top_left_x = bbox_center_x - bbox_width/2
        bbox_top_left_y = bbox_center_y - bbox_height/2
        bbox_bottom_right_x = bbox_center_x + bbox_width/2
        bbox_bottom_right_y = bbox_center_y + bbox_height/2
        bbox.append(ia.BoundingBox(bbox_top_left_x, bbox_top_left_y, bbox_bottom_right_x, bbox_bottom_right_y))
    bbs = ia.BoundingBoxesOnImage(bbox, shape = image.shape)
    # # 만약 어떻게 바운딩박스되는지 보고싶으면
    # ia.imshow(bbs.draw_on_image(image, size=2))

    seq1 = iaa.Sequential([
        iaa.Fliplr(0.5),  # horizontal flips
        iaa.Crop(percent=(0, 0.1)),  # random crops
        # Small gaussian blur with random sigma between 0 and 0.5.
        # But we only blur about 50% of all images.
        iaa.Sometimes(
            0.5,
            iaa.GaussianBlur(sigma=(0, 0.5)))
        ])
    # Strengthen or weaken the contrast in each image.
    seq2 = iaa.LinearContrast((0.75, 1.5))
    # Add gaussian noise.
    # For 50% of all images, we sample the noise once per pixel.
    # For the other 50% of all images, we sample the noise per pixel AND
    # channel. This can change the color (not only brightness) of the
    # pixels.
    seq3 = iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5)
    # Make some images brighter and some darker.
    # In 20% of all cases, we sample the multiplier once per channel,
    # which can end up changing the color of the images.
    seq4 = iaa.Multiply((0.8, 1.2), per_channel=0.2)
        # Apply affine transformations to each image.
        # Scale/zoom them, translate/move them, rotate them and shear them.
    seq5 = iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        rotate=(-25, 25),
        shear=(-8, 8)
    )
    seq6 = iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)  # apply augmenters in random order

    seq7 = iaa.Flipud()

    seq_det = seq1.to_deterministic()
    image_aug = seq_det.augment_images([image])[0]
    bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]
    save_imgbox(filename, 'seq1', bbs, image_aug, bbs_aug)
    seq_det = seq2.to_deterministic()
    image_aug = seq_det.augment_images([image])[0]
    bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]
    save_imgbox(filename, 'seq2', bbs, image_aug, bbs_aug)
    seq_det = seq3.to_deterministic()
    image_aug = seq_det.augment_images([image])[0]
    bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]
    save_imgbox(filename, 'seq3', bbs, image_aug, bbs_aug)
    seq_det = seq4.to_deterministic()
    image_aug = seq_det.augment_images([image])[0]
    bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]
    save_imgbox(filename, 'seq4', bbs, image_aug, bbs_aug)
    seq_det = seq5.to_deterministic()
    image_aug = seq_det.augment_images([image])[0]
    bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]
    save_imgbox(filename, 'seq5', bbs, image_aug, bbs_aug)
    seq_det = seq1.to_deterministic()
    image_aug = seq_det.augment_images([image])[0]
    bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]
    save_imgbox(filename, 'seq6', bbs, image_aug, bbs_aug)
    seq_det = seq1.to_deterministic()
    image_aug = seq_det.augment_images([image])[0]
    bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]
    save_imgbox(filename, 'seq7', bbs, image_aug, bbs_aug)
