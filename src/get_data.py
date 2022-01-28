"""
Download and extract data.
"""
import urllib.request
import zipfile

URL = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip'
# URL = 'https://surfdrive.surf.nl/files/index.php/s/WCPP8WJPrtCbUO5/download'
EXTRACT_DIR = "dataset"

zip_path, _ = urllib.request.urlretrieve(URL)
with zipfile.ZipFile(zip_path, "r") as f:
    f.extractall(EXTRACT_DIR)
