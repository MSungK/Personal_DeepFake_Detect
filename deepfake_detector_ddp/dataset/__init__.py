from .abstract_dataset import AbstractDataset
from .faceforensics import FaceForensics
from .wild_deepfake import WildDeepfake
from .celeb_df import CelebDF
from .dfdc import DFDC
from .custom_deepfake import Custom_Dataset

LOADERS = {
    "FaceForensics": FaceForensics,
    "WildDeepfake": WildDeepfake,
    "CelebDF": CelebDF,
    "DFDC": DFDC,
    "Custom_Deepfake" : Custom_Dataset
}


def load_dataset(name="FaceForensics"):
    print(f"Loading dataset: '{name}'...")
    if name == "Custom_Deepfake":
        dataset = Custom_Dataset()        
    return LOADERS[name]
