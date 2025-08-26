import os
import re

def rename_rain_files_to_3digits(directory_path):
    """
    rain-11.png, rain-3.png 같은 파일을 rain-011.png, rain-003.png로 변경
    """
    if not os.path.exists(directory_path):
        print(f"Directory {directory_path} does not exist!")
        return
    
    files = os.listdir(directory_path)
    rain_files = [f for f in files if f.startswith('rain-') and f.endswith('.png')]
    
    print(f"Found {len(rain_files)} rain files in {directory_path}")
    
    renamed_count = 0
    for filename in rain_files:
        # rain-숫자.png 패턴 매칭
        match = re.match(r'rain-(\d+)\.png', filename)
        if match:
            number = int(match.group(1))
            new_filename = f"rain-{number:03d}.png"  # 3자리 숫자로 포맷
            
            if filename != new_filename:
                old_path = os.path.join(directory_path, filename)
                new_path = os.path.join(directory_path, new_filename)
                
                print(f"Renaming: {filename} -> {new_filename}")
                os.rename(old_path, new_path)
                renamed_count += 1
            else:
                print(f"Already correct format: {filename}")
    
    print(f"Renamed {renamed_count} files")

def rename_norain_files_to_3digits(directory_path):
    """
    norain-11.png, norain-3.png 같은 파일을 norain-011.png, norain-003.png로 변경
    """
    if not os.path.exists(directory_path):
        print(f"Directory {directory_path} does not exist!")
        return
    
    files = os.listdir(directory_path)
    norain_files = [f for f in files if f.startswith('norain-') and f.endswith('.png')]
    
    print(f"Found {len(norain_files)} norain files in {directory_path}")
    
    renamed_count = 0
    for filename in norain_files:
        # norain-숫자.png 패턴 매칭
        match = re.match(r'norain-(\d+)\.png', filename)
        if match:
            number = int(match.group(1))
            new_filename = f"norain-{number:03d}.png"  # 3자리 숫자로 포맷
            
            if filename != new_filename:
                old_path = os.path.join(directory_path, filename)
                new_path = os.path.join(directory_path, new_filename)
                
                print(f"Renaming: {filename} -> {new_filename}")
                os.rename(old_path, new_path)
                renamed_count += 1
            else:
                print(f"Already correct format: {filename}")
    
    print(f"Renamed {renamed_count} files")

if __name__ == "__main__":
    # Rain 데이터셋 경로들
    base_path = "/home/joon/ImageRestoration-AllInOne/Combined_Method/aoiir/datasets/dataset"
    
    # Rain100L rainy 폴더
    rainy_dir = os.path.join(base_path, "Rain100L", "rainy")
    if os.path.exists(rainy_dir):
        print(f"Processing rainy directory: {rainy_dir}")
        rename_rain_files_to_3digits(rainy_dir)
    else:
        print(f"Rainy directory not found: {rainy_dir}")
    
    # Rain100L no 폴더 (clean/ground truth)
    no_dir = os.path.join(base_path, "Rain100L", "no")
    if os.path.exists(no_dir):
        print(f"\nProcessing no directory: {no_dir}")
        rename_norain_files_to_3digits(no_dir)
    else:
        print(f"No directory not found: {no_dir}")
    
    print("\nRenaming completed!")
















