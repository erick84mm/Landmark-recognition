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

def frame_to_list(df, n):
    res = []
    for row in df.itertuples():
        res.append((n, row[1], row[3], row[2]))
    return res


def create_partitions():
    ##Load index files
    train_data = pd.read_csv('../index/train.csv')
    test_data = pd.read_csv('../index/test.csv')

    ##Divide training and test set in two groups according to the category frequency (20<=) and (20>)
    categories = train_data['landmark_id'].value_counts().to_dict()

    cat0 = []
    cat1 = []

    for cat in categories:
        if(categories[cat] > 20):
            cat0.append(cat)
        else:
            cat1.append(cat)

    exp0_data = train_data[train_data['landmark_id'].isin(cat0)]
    exp1_data = train_data[train_data['landmark_id'].isin(cat1)]

    ## splitting the data into dev and training set for each experiment

    ## Experiment 0 categories with more than 20 training examples
    train0 = exp0_data.sample(frac=0.8)
    dev0 = exp0_data.drop(train0.index)

    ## Experiment 1 categories with less than 20 training examples
    train1 = exp1_data.sample(frac=0.9)
    dev1 = exp1_data.drop(train1.index)

    ## Saving the results to the index
    train0.to_csv(_INDEX_DIR_ + "train0.csv", sep=',')
    train1.to_csv(_INDEX_DIR_ + "train1.csv", sep=',')
    dev0.to_csv(_INDEX_DIR_ + "dev0.csv", sep=',')
    dev1.to_csv(_INDEX_DIR_ + "dev1.csv", sep=',')

def ParseData(data_file, n):
    csvfile = open(data_file, 'r')
    csvreader = csv.reader(csvfile)
    key_url_list = [(n, line[1], line[3], line[2]) for line in csvreader]
    return key_url_list

## Download images and create the folder corresponding to the respective category 

def download_image(key_url):
    (n, key, cat, url) = key_url
    out_dir = ""
    if(n==0):
        out_dir = _TRAIN_0_DIR_
    elif(n==1):
        out_dir = _TRAIN_1_DIR_
    elif(n==2):
        out_dir = _DEV_0_DIR_
    elif(n==3):
        out_dir = _DEV_1_DIR_
    else:
        return
    
    out_dir = out_dir+str(cat)+"/"
    
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

def loader(data_file, n, cores):
    key_url_list = ParseData(data_file, n)
    pool = multiprocessing.Pool(processes=cores)  # Num of CPUs
    pool.map(download_image, key_url_list)
    pool.close()
    pool.terminate()
    
def main():
    if(not os.path.isfile(_INDEX_DIR_+"train0.csv")):
        create_partitions()
    loader(_INDEX_DIR_+"train0.csv",0, 28)
    print("train0")
    loader(_INDEX_DIR_+"train1.csv",1, 28)
    print("train1")
    loader(_INDEX_DIR_+"dev0.csv",2, 28)
    print("dev0")
    loader(_INDEX_DIR_+"dev1.csv",3, 28)
    print("dev1")
    
main()