import torch
from torchvision import models

class ModelContainer():
    # select model.
    #
    # @params
    # str model_name : "resnet18", "resnet50" is available.
    # bool is_weights : is using weights. (default = True)
    # int? num_classes : output num of fc layer. (default = None)
    # str? model_path : model path loading from. (default = None)
    @staticmethod
    def select(model_name:str, is_weights:bool=True, num_classes:int|None=None, model_path:str|None=None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        weights = None

        # select model
        match (model_name):
            case ("resnet18"):
                if is_weights:
                    weights = models.ResNet18_Weights.DEFAULT
                model = models.resnet18(weights=weights)
                
            case ("resnet50") if num_classes:
                if is_weights:
                    weights = models.ResNet50_Weights.DEFAULT
                model = models.resnet50(weights=weights)
        
        # select fc layer
        if num_classes:
            model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        else:
            model.fc = torch.nn.Identity()
            
        if model_path:
            model.load_state_dict(torch.load(model_path, map_location=device))
        
        model = model.to(device)
        return model, device