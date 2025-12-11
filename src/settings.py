import json
import os

def load_json(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
        if data is None:
            raise Exception("json data is None.")
        return data

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
JSON_PATH = os.path.join(BASE_DIR, "settings.json")
        
def load_dataset_downloader_data(json_path):
    data = load_json(json_path)
    dataset_downloader_data = data["dataset_downloader"]
    
    tags = dataset_downloader_data["tags"]
    start_index = dataset_downloader_data["start_index"]
    end_index = dataset_downloader_data["end_index"]
    save_dir = dataset_downloader_data["save_dir"]

    return tags, start_index, end_index, save_dir

def load_detect_same_image_data(json_path):
    data = load_json(json_path)
    detect_same_image_data = data["detect_same_image"]

    dir_a = detect_same_image_data["dir_a"]
    dir_b = detect_same_image_data["dir_b"]
    result_dir = detect_same_image_data["result_dir"]

    return dir_a, dir_b, result_dir

def load_isolate_duplicated_images_data(json_path):
    data = load_json(json_path)
    isolate_duplicated_images_data = data["isolate_duplicated_images"]

    target = isolate_duplicated_images_data["target"]

    return target

def load_train_data(json_path):
    data = load_json(json_path)
    train_data = data["train"]

    dataset_dir = train_data["dataset_dir"]
    model_path_save = train_data["model_path_save"]

    return dataset_dir, model_path_save

def load_test_data(json_path):
    data = load_json(json_path)
    test_data = data["test"]
    
    dataset_dir = test_data["dataset_dir"]
    model_path = test_data["model_path"]

    return dataset_dir, model_path

def load_predict_data(json_path):
    data = load_json(json_path)
    predict_data = data["predict"]

    target = predict_data["target"]
    model_path = predict_data["model_path"]

    return target, model_path

