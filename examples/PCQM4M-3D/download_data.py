from urllib import response
import urllib3
from tqdm import tqdm
import zipfile
import os
import io
import sys 


def download_data():
    path = "data/"
    try:
        os.mkdir(path)
    except OSError as error:
        print(error) 
    http = urllib3.PoolManager()
    url = "http://ogb-data.stanford.edu/data/lsc/pcqm4m-v2_xyz.zip"
    response = http.request('GET', url, preload_content=False)
    
    content_bytes = response.headers.get("Content-Length")
    chunk_size = 2048
    total_chunks = int(content_bytes) // chunk_size
    print("Downloading the data file")

    with open(path+"pcqm4m-v2_xyz.zip", 'wb') as out_file:
        for chunk in tqdm(response.stream(chunk_size), total=total_chunks):
            out_file.write(chunk)
    response.release_conn()

def extract_data():
    path = "data/processed"
    try:
        os.mkdir(path)
    except OSError as error:
        print(error)
    with zipfile.ZipFile("data/pcqm4m-v2_xyz.zip","r") as zip_ref:
        zip_ref.extractall("data/processed")


def check_already_exists():
    return False

def clean_up():
    os.unlink("data/data.zip")
if __name__ == '__main__':
    data_exists = check_already_exists()

    if (not data_exists):
        download_data()
        # extract_data()
        # clean_up()
