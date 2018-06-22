import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sys, os, multiprocessing, csv
from urllib import request, error
from PIL import Image
from io import BytesIO
from functools import partial


_TRAIN_0_DIR_ = "/gs/hs0/tga-nlp-titech/erick/data/exp0/train/"
_TRAIN_1_DIR_ = "/gs/hs0/tga-nlp-titech/erick/data/exp1/train/"
_DEV_0_DIR_ = "/gs/hs0/tga-nlp-titech/erick/data/exp0/dev/"
_DEV_1_DIR_ = "/gs/hs0/tga-nlp-titech/erick/data/exp1/dev/"
_VAL_DIR_ = "/gs/hs0/tga-nlp-titech/erick/data/val/"
_INDEX_DIR_ = "/gs/hs0/tga-nlp-titech/erick/data/index/"
_TEST_FILE_ = "/gs/hs0/tga-nlp-titech/erick/data/index/test.csv"


def ParseData(data_file):
    csvfile = open(data_file, 'r')
    csvreader = csv.reader(csvfile)
    key_url_list = [(line[0], line[1]) for line in csvreader]
    return key_url_list

## Download images and create the folder corresponding to the respective category 

def download_image(key_url):
    (key, url) = key_url
    out_dir = _VAL_DIR_
    
    ##create the category directory
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
        
    filename = os.path.join(out_dir, '{}.jpg'.format(key))
    if os.path.exists(filename):
        #print('Image {} already exists. Skipping download.'.format(filename))
        return
    try:
        response = request.urlopen(url)
        image_data = response.read()
    except:
        #print('Warning: Could not download image {} from {}'.format(key, url))
        print(key)
        return

    try:
        pil_image = Image.open(BytesIO(image_data))
    except:
        #print('Warning: Failed to parse image {}'.format(key))
        print(key)
        return

    try:
        pil_image_rgb = pil_image.convert('RGB')
    except:
        #print('Warning: Failed to convert image {} to RGB'.format(key))
        print(key)
        return

    try:
        pil_image_rgb.save(filename, format='JPEG', quality=90)
    except:
        #print('Warning: Failed to save image {}'.format(filename))
        print(key)
        return

def loader(data_file, cores):
    key_url_list = ParseData(data_file)
    pool = multiprocessing.Pool(processes=cores)  # Num of CPUs
    pool.map(download_image, key_url_list)
    pool.close()
    pool.terminate()
    
def main():
    loader(_INDEX_DIR_+"test.csv", 100)
    
main()