import requests  
import tarfile
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_and_extract(url, download_path, extract_path):
    
    try:
        # Download the file
        response = requests.get(url, stream=True)
        with open(download_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    file.write(chunk)
        
        # Extract the file
        with tarfile.open(download_path, 'r') as tar:
            tar.extractall(path=extract_path)

        logger.info("Extraction du fichier...")
        with tarfile.open(download_path, "r") as tar:
            tar.extractall(path=extract_path)
        
    except requests.RequestException as e:
        logger.error(f"Error downloading the file: {e}")
        raise
    except tarfile.TarError as e:
        logger.error(f"Error extracting the tar file: {e}")
        raise
    except OSError as e:
        logger.error(f"File system error: {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        raise