import os
import requests
from zipfile import ZipFile

# Example URLs (these should be replaced with the actual URLs from the dataset page)
# dataset_urls = [
#     'https://download.data.fid-move.de/dzsf/osdar23/1_calibration_1.1.zip',
#     'https://download.data.fid-move.de/dzsf//osdar23/2_station_berliner_tor_2.1.zip',
#     'https://download.data.fid-move.de/dzsf/osdar23/3_fire_site_3.1.zip',
#     'https://download.data.fid-move.de/dzsf/osdar23/4_station_pedestrian_bridge_4.1.zip'
# ]

dataset_urls = [
    'https://download.data.fid-move.de/dzsf/osdar23/1_calibration_1.1.zip'
]

download_dir = 'osdar23_dataset'
os.makedirs(download_dir, exist_ok=True)

def download_and_extract(url, download_dir):
    local_filename = os.path.join(download_dir, url.split('/')[-1])
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    with ZipFile(local_filename, 'r') as zip_ref:
        zip_ref.extractall(download_dir)
    os.remove(local_filename)

for url in dataset_urls:
    download_and_extract(url, download_dir)

print("Download and extraction complete.")
