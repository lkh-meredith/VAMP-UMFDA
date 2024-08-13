DATASETs_DOMAINs = {
    "OfficeHome": {
        "domains": ["Art", "Clipart", "Product", "Real World"],
        "domain_ids": {"Art": 0, "Clipart": 1, "Product": 2, "Real World": 3},
        "ids_domain": {0: "Art", 1: "Clipart", 2: "Product", 3: "Real World"}
    },
    "DomainNet":{
        "domains":["clipart", "painting", "real", "sketch"],
        "domain_ids": {"clipart": 0, "painting": 1, "real": 2, "sketch": 3},
        "ids_domain": {0: "clipart", 1: "painting", 2: "real", 3: "sketch"}
    }
}

# source: https://github.com/openai/CLIP/blob/main/notebooks/Prompt_Engineering_for_ImageNet.ipynb
CUSTOM_TEMPLATES = {
    "OxfordPets": "a photo of a {}, a type of pet.",
    "OxfordFlowers": "a photo of a {}, a type of flower.",
    "FGVCAircraft": "a photo of a {}, a type of aircraft.",
    "DescribableTextures": "{} texture.",
    "EuroSAT": "a centered satellite photo of {}.",
    "StanfordCars": "a photo of a {}.",
    "Food101": "a photo of {}, a type of food.",
    "SUN397": "a photo of a {}.",
    "Caltech101": "a photo of a {}.",
    "UCF101": "a photo of a person doing {}.",
    "ImageNet": "a photo of a {}.",
    "ImageNetSketch": "a photo of a {}.",
    "ImageNetV2": "a photo of a {}.",
    "ImageNetA": "a photo of a {}.",
    "ImageNetR": "a photo of a {}.",
    "PACS": "a photo of a {}.",
    "VLCS": "a photo of a {}.",
    # Our defination for VAMP
    "MS_OfficeHome": "a photo of a {}.",
    "MS_DomainNet": "a photo of a {}.",
}
