import os
import shutil
import kagglehub

def download_and_setup():
    # Make sure token is in environment just in case
    os.environ['KAGGLE_API_TOKEN'] = 'KGAT_a675eca91ae70e3383b10b40687224f4'
    
    competition_name = 'ieee-fraud-detection'
    dest_dir = os.path.abspath('Datasets/ieee-fraud-detection')
    
    print(f"Starting download for competition: {competition_name}...")
    try:
        # Download files via kagglehub
        path = kagglehub.competition_download(competition_name)
        print(f"Successfully downloaded to cache: {path}")
        
        # Create destination directory in workspace
        os.makedirs(dest_dir, exist_ok=True)
        
        # Copy files to Datasets/ieee-fraud-detection
        print(f"Copying files to local directory: {dest_dir}...")
        for item in os.listdir(path):
            s = os.path.join(path, item)
            d = os.path.join(dest_dir, item)
            if os.path.isdir(s):
                if os.path.exists(d):
                    shutil.rmtree(d)
                shutil.copytree(s, d)
            else:
                shutil.copy2(s, d)
        print("Setup complete! Dataset is ready at Datasets/ieee-fraud-detection.")
        
    except Exception as e:
        print("\n--- ERROR DURING DOWNLOAD ---")
        print(str(e))
        print("\nIf you see a '403 Client Error: Forbidden', it usually means you need to accept the rules of the competition.")
        print("Please follow these steps:")
        print("1. Log in to your Kaggle account in your web browser.")
        print("2. Navigate to: https://www.kaggle.com/competitions/ieee-fraud-detection/rules")
        print("3. Read the rules and click the 'I Understand and Accept' or 'Join Competition' button.")
        print("4. Re-run this script after accepting.")

if __name__ == '__main__':
    download_and_setup()
