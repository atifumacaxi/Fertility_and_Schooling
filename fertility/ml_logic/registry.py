import time
import os
import glob

from colorama import Fore, Style
from fertility.params import *
from tensorflow import keras

def save_model(model: keras.Model = None) -> None:

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Save model locally
    model_path = os.path.join(LOCAL_REGISTRY_PATH, "models", f"{timestamp}.h5")
    model.save(model_path)

    print("✅ Model saved locally")

    # save the model to disk
    #filename = 'finalized_model.sav'
    #pickle.dump(model, open(filename, 'wb'))

def loads_model() -> keras.Model:

    print(Fore.BLUE + f"\nLoad latest model from local registry..." + Style.RESET_ALL)

    #Get the latest model version name by the timestamp on disk
    local_model_directory = os.path.join(LOCAL_REGISTRY_PATH, "models")
    local_model_paths = glob.glob(f"{local_model_directory}/*")

    if not local_model_paths:
        return None

    most_recent_model_path_on_disk = sorted(local_model_paths)[-1]

    print(Fore.BLUE + f"\nLoad latest model from disk..." + Style.RESET_ALL)

    latest_model = keras.models.load_model(most_recent_model_path_on_disk)

    print("✅ Model loaded from local disk")


    return latest_model

#preprocess_and_train()
