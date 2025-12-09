import os
from urllib.parse import urljoin
import requests

TAGS = ["hoshino_(blue_archive)", "1girl"]
START_PAGE_INDEX = 100
END_PAGE_INDEX = 101
SAVE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "dl_images")

BASE_URL = "https://danbooru.donmai.us"
POST_JSON_URL = urljoin(BASE_URL, "posts.json")


def download(tags:list[str], start:int, end:int, save_dir:str):
    for i in list(range(start, end)):
        URL = f"{POST_JSON_URL}?page={str(i)}&tags={'+'.join(tags)}"
        print(f"\n>>URL: {URL}, page: {i}")

        # res = requests.get(POST_JSON_URL, params=PARAMS)
        res = requests.get(URL)
        print(f"URL :{res.url} (READ)")
        try:
            res.raise_for_status()
        except:
            print(f"URL {URL} is failed.")
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
    print("Start download")
    download(TAGS, START_PAGE_INDEX, END_PAGE_INDEX, SAVE_DIR)
