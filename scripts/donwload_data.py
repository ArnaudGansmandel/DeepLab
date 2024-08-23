if __name__ == "__main__":
    import os
    from utils.data_organizer import DataOrganizer
    from data_pipeline.downloader import Download


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
        downloader = Download(config['voc_url'], download_path, config['base_dir'])
        downloader.download_file() 
        downloader.extract_file()    
        os.remove(download_path)

    # Organize the dataset
    if not os.path.exists(config['base_dir']+'/train') and not os.path.exists(config['base_dir']+'/val') and not os.path.exists(config['base_dir']+'/trainval'):
        organizer = DataOrganizer(config)
        organizer.organize_data()
