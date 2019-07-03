import os
import requests
import pandas as pd
import time
from multiprocessing.dummy import Pool as ThreadPool


# Create folder structure if not exists


def create_file_path(path):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    return path

# Download file


def save_file(path, content):
    with open(path, 'wb') as f:
        for chunk in content.iter_content(1024):
            f.write(chunk)
        create_file_path(path)


# check if already downloaded


def file_exists(path):
    return os.path.isfile(path)


# download images


def download_images(images):
    base_dir = '/home/anastasiiapyltsova/Projects/watermark_project/'
    output_folder = 'all_learning_data'

    for img_url in images.split(','):

        path = os.path.join(base_dir, output_folder, f'img_{img_url.split("/")[-2]}.jpg')

        if not file_exists(path):
            r = requests.get(img_url)
            save_file(path, r)

        # else:
        #     print("exists: %s" % path)


if __name__ == '__main__':
    start = time.time()

    # read csv file

    path = '/home/anastasiiapyltsova/Projects/watermark_project/am_autoparts_images.csv'

    try:
        data = pd.read_csv(path, delimiter='\t')
    except FileNotFoundError:
        print('Exception: file not found at the current path')

    number_of_threads = 20
    pool = ThreadPool(number_of_threads)

    pool.map(download_images, data.images)

    end = time.time()
    print(f'Executing time: {end-start}')
