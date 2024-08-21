import requests
import tarfile
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Download:
    def __init__(self, url, download_path, extract_path):
        self.url = url
        self.download_path = download_path
        self.extract_path = extract_path

    def download_file(self):
        """
        Downloads the file from the specified URL and saves it to the download path.
        """
        try:
            logger.info(f"Downloading the file from {self.url}...")
            response = requests.get(self.url, stream=True)
            with open(self.download_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        file.write(chunk)
            logger.info(f"File downloaded successfully and saved to {self.download_path}")
        except requests.RequestException as e:
            logger.error(f"Error downloading the file: {e}")
            raise

    def extract_file(self):
        """
        Extracts the downloaded tar file to the specified extract path.
        """
        try:
            logger.info(f"Extracting the file to {self.extract_path}...")
            with tarfile.open(self.download_path, 'r') as tar:
                tar.extractall(path=self.extract_path)
            logger.info("File extracted successfully.")
        except tarfile.TarError as e:
            logger.error(f"Error extracting the tar file: {e}")
            raise
        except OSError as e:
            logger.error(f"File system error: {e}")
            raise
        