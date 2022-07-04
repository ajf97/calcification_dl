import os
import shutil

import hydra
from omegaconf import DictConfig

from src.paths import CONFIG_FOLDER_PATH


def new_name_dcm_file(file_path: str) -> str:
    # Check description of folder
    folder_description: str = file_path.split(os.sep)[-2]
    new_name: str = ""

    if "full" in folder_description:
        new_name = file_path.split(os.sep)[-4] + "_FULL" + ".dcm"
    elif any(word in folder_description for word in ["cropped", "ROI", "mask"]):
        # Check file number
        file_number = file_path.split(os.sep)[-1].replace(".dcm", "").split("-")[-1]

        if file_number == "1":
            new_name = file_path.split(os.sep)[-4] + "_CROP" + ".dcm"
        elif file_number == "2":
            new_name = file_path.split(os.sep)[-4] + "_MASK" + ".dcm"

    return new_name


def make_directories(path: str) -> None:
    # Make a directory with different lesions
    path_calc = os.path.join(path, "Calc")
    path_mass = os.path.join(path, "Mass")

    os.mkdir(path_calc)
    os.mkdir(path_mass)

    # Create train and test folders for every lesion type
    path_calc_train = os.path.join(path_calc, "Training")
    path_calc_test = os.path.join(path_calc, "Test")
    path_mass_train = os.path.join(path_mass, "Training")
    path_mass_test = os.path.join(path_mass, "Test")

    os.mkdir(path_calc_train)
    os.mkdir(path_calc_test)
    os.mkdir(path_mass_train)
    os.mkdir(path_mass_test)


def remove_empty_folders(path_abs: str) -> None:
    walk = list(os.walk(path_abs))
    for path, _, _ in walk[::-1]:
        if len(os.listdir(path)) == 0:
            shutil.rmtree(path)


def restructure_dataset(path: str) -> None:
    make_directories(path)

    for root, _, files in os.walk(path):
        for file in files:
            # Rename file
            old_name_path = os.path.join(root, file)
            new_name = new_name_dcm_file(old_name_path)
            os.replace(old_name_path, os.path.join(root, new_name))

            # Move file to folder
            lesion_type = new_name.split("_")[0].split("-")[0]
            train_or_test = new_name.split("_")[0].split("-")[1]
            shutil.move(
                os.path.join(root, new_name),
                os.path.join(path, lesion_type, train_or_test, new_name),
            )
    remove_empty_folders(path)


@hydra.main(config_path=CONFIG_FOLDER_PATH, config_name="config", version_base=None)
def app(cfg: DictConfig) -> None:
    path = cfg.dataset[1].path
    restructure_dataset(path)


if __name__ == "__main__":
    app()
