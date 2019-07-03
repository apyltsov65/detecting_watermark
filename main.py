import glob
import os
from classify import MultinomialNB
from PIL import Image

TRAINING_POSITIVE = 'training_positive/*.jpg'
TRAINING_NEGATIVE = 'training_negative/*.jpg'
TEST_POSITIVE = 'test_positive/*jpg'
TEST_NEGATIVE = 'test_negative/*jpg'
SAVE_CROP = '/home/anastasiiapyltsova/Projects/watermark_project/crop_image/'
CROP_WIDTH, CROP_HEIGHT = 100, 100
RESIZED = (16, 16)


def get_image_data(infile):
    image = Image.open(infile)
    # image = image.convert('RGBA')
    width, height = image.size
    # print(SAVE_CROP + os.path.splitext(infile)[-1])
    # left upper right lower
    left = width / 2 - CROP_WIDTH / 2
    upper = height / 2 - CROP_HEIGHT / 2
    right = width / 2 + CROP_WIDTH / 2
    lower = height / 2 + CROP_HEIGHT / 2

    box = left, upper, right, lower
    region = image.crop(box)
    # region.save(SAVE_CROP + os.path.splitext(infile)[-2].split('/')[-1] + '.jpg', 'JPEG')
    resized = region.convert('RGBA')
    data = resized.getdata()
    # Convert RGB to simple averaged value.
    data = [sum(pixel) / 3 for pixel in data]
    # Combine location and value.
    values = list()
    for location, value in enumerate(data):
        values.extend([location] * int(value))
    return values


def main():
    watermark = MultinomialNB()
#     Training
    count = 0
    for infile in glob.glob(TRAINING_POSITIVE):
        data = get_image_data(infile)
        watermark.train((data, 'positive'))
        count += 1
        print(f'Training count: {count}')
        # if count == 20:
        #     return

    for infile in glob.glob(TRAINING_NEGATIVE):
        data = get_image_data(infile)
        watermark.train((data, 'negative'))
        count += 1
        print(f'Training count: {count}')

#         Testing
    positive, total = 0, 0
    for infile in glob.glob(TEST_POSITIVE):
        data = get_image_data(infile)
        prediction = watermark.classify(data)
        if prediction.label == 'positive':
            positive += 1
        total += 1

    print(f'Positive testing {positive}/{total}')

    negative, total = 0, 0
    for infile in glob.glob(TEST_NEGATIVE):
        data = get_image_data(infile)
        prediction = watermark.classify(data)
        if prediction.label == 'negative':
            negative += 1
        total += 1

    print(f'Negative testing {negative}/{total}')


if __name__ == '__main__':
    main()
