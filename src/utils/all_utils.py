import yaml
import os
import logging #bcs I am going to save all logs
from bing_image_downloader import downloader

#Function to read yaml file
# it will return a dictionary,# path to file must be a string
def read_yaml(path_to_yaml: str) -> dict: 
    with open(path_to_yaml) as yaml_file: # open that yml file
        content = yaml.safe_load(yaml_file) # to read it.

    return content

# function to create our directory
def create_directory(dirs: list):#Multiple dirs
    for dir_path in dirs:#iterate throught this list of dirs
        os.makedirs(dir_path, exist_ok=True)# if it exists it will not be created
        logging.info(f"Directory is created at {dir_path}")# info :location of the created folder 

# 
def data_download(name_of_image,limits):
    downloader.download(name_of_image, limit=limits,  output_dir='new_dataset', adult_filter_off=True, force_replace=False, timeout=60, verbose=True)
    logging.info("Data has been downloaded.")


