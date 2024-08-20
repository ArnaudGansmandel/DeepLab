if __name__ == "__main__":
    import os
    from data_pipeline.sourcing.downloader import download_and_extract
    from data_pipeline.sourcing.data_organizer import DataOrganizer

    config = {
    'base_dir' : 'data',
    'src_image_dir': 'data/VOCdevkit/VOC2012/JPEGImages',
    'src_mask_dir': 'data/VOCdevkit/VOC2012/SegmentationClass',
    'txt_dir': 'data/VOCdevkit/VOC2012/ImageSets/Segmentation',
    'voc_url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar'
    }

    # Download and extract the dataset if it doesn't exist
    if not os.path.exists(config['base_dir']):
        os.makedirs(config['base_dir'], exist_ok=True)

        
        download_path = os.path.join(config['base_dir'], 'VOCtrainval_11-May-2012.tar')
        download_and_extract(config['voc_url'], download_path, config['base_dir'])
        os.remove(download_path)

    # Organize the dataset
    if not os.path.exists(config['base_dir']+'/train') and not os.path.exists(config['base_dir']+'/val') and not os.path.exists(config['base_dir']+'/trainval'):
        organizer = DataOrganizer(config)
        organizer.organize_data()
