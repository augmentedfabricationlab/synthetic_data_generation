import blenderproc as bproc 
import os
import zipfile
import requests
from io import BytesIO
from blenderproc.python.loader.CCMaterialLoader import load_ccmaterials

# List of asset names to download from AmbientCG (you can extend this list)
ASSETS_TO_DOWNLOAD = [
    "Bricks024", "PavingStones131", "Wood049"
]

# AmbientCG base URL for downloads
BASE_URL = "https://ambientcg.com/get"

# Target folder
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
TARGET_FOLDER = os.path.join(base_dir, 'resources', 'ccmaterials')


def download_and_extract_asset(asset_name: str, target_folder: str):
    print(f"Downloading {asset_name}...")
    # Construct URL: e.g., https://ambientcg.com/get?file=Bricks024_2K-JPG.zip
    url = f"{BASE_URL}?file={asset_name}_2K-JPG.zip"
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to download {asset_name} from {url}")
        return

    with zipfile.ZipFile(BytesIO(response.content)) as z:
        z.extractall(os.path.join(target_folder, asset_name))
    print(f"✓ {asset_name} downloaded and extracted.")


def rename_for_blenderproc(asset_folder: str):
    # ambientCG names maps e.g. Bricks024_2K-JPG_Color.jpg / _NormalGL.jpg, but
    # BlenderProc 2.5.0's load_ccmaterials only finds Bricks024_2K_Color.jpg /
    # _Normal.jpg and silently skips the asset otherwise
    for f in os.listdir(asset_folder):
        new = f.replace("_2K-JPG_", "_2K_").replace("_NormalGL.", "_Normal.")
        if new != f:
            os.rename(os.path.join(asset_folder, f), os.path.join(asset_folder, new))


def main():
    os.makedirs(TARGET_FOLDER, exist_ok=True)

    for asset in ASSETS_TO_DOWNLOAD:
        asset_folder = os.path.join(TARGET_FOLDER, asset)
        if not os.path.exists(asset_folder):
            download_and_extract_asset(asset, TARGET_FOLDER)
        else:
            print(f"{asset} already exists. Skipping download.")
        if os.path.exists(asset_folder):
            rename_for_blenderproc(asset_folder)

    print("Loading materials into Blender...")
    materials = load_ccmaterials(folder_path=TARGET_FOLDER)
    print(f"Loaded {len(materials)} materials.")


if __name__ == "__main__":
    main()
