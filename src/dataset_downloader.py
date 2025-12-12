import os
import pathlib
from urllib.parse import urljoin
import requests

from utils.path_solver import get_base_dir, get_absolute_path_if_not
from utils.settings import load_dataset_downloader_data, JSON_PATH

BASE_URL = "https://danbooru.donmai.us"
POST_JSON_URL = urljoin(BASE_URL, "posts.json")

def download(json_url, tags:list[str], start:int, end:int, save_dir:str):
    for i in list(range(start, end)):
        url = f"{json_url}?page={str(i)}&tags={'+'.join(tags)}"
        print(f"\n>>URL: {url}, page: {i}")

        # res = requests.get(POST_JSON_URL, params=PARAMS)
        res = requests.get(url)
        print(f"URL :{res.url} (READ)")
        try:
            res.raise_for_status()
        except:
            print(f"URL {url} is failed.")
            continue
        data = res.json()

        for d in data:
            asset = d["media_asset"]
            ext = asset["file_ext"]
            if not ext in ["jpg", "png"]:
                print(f"file ext is {ext}, jpg and png is only available.")
                continue
            try:
                variants = asset["variants"]
            except:
                print(f"Response has no variants. Skip it.")
                continue
            file_info = {}
            for v in variants:
                if v["type"] == "sample":
                    file_info = v
                    break
                if v["type"] == "original":
                    file_info = v
                    break
            
            print(file_info)
            file_url = file_info["url"]
            print(f"file url: {file_url}")

            img_res = requests.get(file_url)
            try:
                img_res.raise_for_status()
            except:
                print(f"file URL {file_url} is failed.")
                continue
            
            filename = asset["md5"] + "." + ext
            save_path = os.path.join(save_dir, filename)
            if os.path.exists((save_path)):
                print("file is already exist. Skip it.")
                
            with open(save_path, "wb") as f:
                f.write(img_res.content)
                print("Saved:", filename)

if __name__ == "__main__":
    TAGS, START_PAGE_INDEX, END_PAGE_INDEX, SAVE_DIR = load_dataset_downloader_data(JSON_PATH)
    SAVE_DIR = get_absolute_path_if_not(get_base_dir(__file__), SAVE_DIR)
        
    print("Start download")
    download(POST_JSON_URL, TAGS, START_PAGE_INDEX, END_PAGE_INDEX, SAVE_DIR)
