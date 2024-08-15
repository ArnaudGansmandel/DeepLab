if __name__ == "__main__":
    import os
    from data_pipeline.sourcing.downloader import download_and_extract

    # Constants
    VOC_URL = 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar'
    DATA_DIR = 'data'
    os.makedirs(DATA_DIR, exist_ok=True)

    # Download and extract the dataset
    download_path = os.path.join(DATA_DIR, 'VOCtrainval_11-May-2012.tar')
    download_and_extract(VOC_URL, download_path, DATA_DIR)
    os.remove(download_path)

