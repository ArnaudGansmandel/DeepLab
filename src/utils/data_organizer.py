import os
import shutil

class DataOrganizer:
    def __init__(self, config):
        """
        Initialize the DataOrganizer with the required directories.

        Parameters:
        base_dir (str): Base directory to store the organized data.
        txt_dir (str): Directory containing the train.txt, val.txt, and trainval.txt files.
        src_image_dir (str): Directory containing the source JPEG images.
        src_mask_dir (str): Directory containing the source mask images.
        """
        self.base_dir = config['base_dir']
        self.txt_dir = config['txt_dir']
        self.src_image_dir = config['src_image_dir']
        self.src_mask_dir = config['src_mask_dir']

    def create_directory_structure(self, sub_dirs):
        """
        Create directory structure for datasets.

        Parameters:
        sub_dirs (list): List of subdirectories to create within the base directory.
        """
        for sub_dir in sub_dirs:
            os.makedirs(os.path.join(self.base_dir, sub_dir, 'images'), exist_ok=True)
            os.makedirs(os.path.join(self.base_dir, sub_dir, 'masks'), exist_ok=True)

    def copy_files(self, file_list, dest_image_dir, dest_mask_dir):
        """
        Copy image and mask files to the destination directories.

        Parameters:
        file_list (list): List of file names to copy.
        dest_image_dir (str): Destination directory for images.
        dest_mask_dir (str): Destination directory for masks.
        """
        for file_name in file_list:
            image_file = f"{file_name}.jpg"
            mask_file = f"{file_name}.png"

            # Copy the image file
            shutil.copy(os.path.join(self.src_image_dir, image_file), os.path.join(dest_image_dir, image_file))
            # Copy the mask file
            shutil.copy(os.path.join(self.src_mask_dir, mask_file), os.path.join(dest_mask_dir, mask_file))

    def read_file_list(self, dataset):
        """
        Read the list of files from the corresponding .txt file.

        Parameters:
        dataset (str): The dataset type ('train', 'val', 'trainval').

        Returns:
        list: List of file names.
        """
        txt_file_path = os.path.join(self.txt_dir, f"{dataset}.txt")
        with open(txt_file_path, 'r') as file:
            file_list = [line.strip() for line in file.readlines()]
        return file_list

    def organize_data(self):
        """
        Organize data into train, val, and trainval directories with corresponding images and masks.
        """
        # Create the required directory structure
        self.create_directory_structure(['train', 'val', 'trainval'])

        # Process each dataset type
        for dataset in ['train', 'val', 'trainval']:
            file_list = self.read_file_list(dataset)

            dest_image_dir = os.path.join(self.base_dir, dataset, 'images')
            dest_mask_dir = os.path.join(self.base_dir, dataset, 'masks')

            # Copy the files to the respective directories
            self.copy_files(file_list, dest_image_dir, dest_mask_dir)


