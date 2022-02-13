from urllib import response
import urllib3
from tqdm import tqdm
import zipfile
import os
import os.path as osp

def get_data():
    download_data()
    extract_data("data/pcqm4m-v2_xyz.zip", "data/processed/xyz" )
    extract_data("data/pcqm4m-v2.zip","data/processed/labels")
    clean_up()

def download_data():
    path = "data/"
    try:
        os.mkdir(path)
    except OSError as error:
        print(error) 
    
    print("Downloading the xyz data file")

    if (not check_already_exists("data/processed/xyz")):
        xyz_url = "http://ogb-data.stanford.edu/data/lsc/pcqm4m-v2_xyz.zip"
        url_downloader(xyz_url, path+"pcqm4m-v2_xyz.zip")
    else:
        print("Found non-empty data/processed/xyz directory")

    if (not check_already_exists("data/processed/labels")):
        print("Downloading the smiles data file")
        smiles_url = "https://dgl-data.s3-accelerate.amazonaws.com/dataset/OGB-LSC/pcqm4m-v2.zip"
        url_downloader(smiles_url, path+"pcqm4m-v2.zip")
    else:
        print("Found non-empty data/processed/labels directory")


def url_downloader(url, path):
    http = urllib3.PoolManager()
    response = http.request('GET', url, preload_content=False)
    content_bytes = response.headers.get("Content-Length")
    chunk_size = 8096
    total_chunks = int(content_bytes) // chunk_size
    with open(path, 'wb') as out_file:
        for chunk in tqdm(response.stream(chunk_size), total=total_chunks):
            out_file.write(chunk)
    response.release_conn()


def extract_data(f_name, path):
    try:
        os.mkdir(path)
    except OSError as error:
        print(error)
    with zipfile.ZipFile(f_name,"r") as zip_ref:
        zip_ref.extractall(path)
    
def check_already_exists(path):
    if osp.isdir(path):
        # Check if the directory not empty
        return len(os.listdir(path)) > 0
    return False

def clean_up():
    os.unlink("data/pcqm4m-v2_xyz.zip")
    os.unlink("data/pcqm4m-v2.zip")

if __name__ == '__main__':
    get_data()
