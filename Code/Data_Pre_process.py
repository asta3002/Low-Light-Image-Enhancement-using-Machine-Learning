from PIL import Image
import os
import cv2
import numpy as np

def Reshape_Images(root_directory,target_size):


    # Iterate through folders
    for folder_name in range(1, 361):
        folder_path = os.path.join(root_directory, str(folder_name))

        # Check if the folder exists
        if os.path.exists(folder_path):
            # Load image from each folder
            image_path = os.path.join(folder_path, '1.jpg')
            
            # Check if the image file exists
            if os.path.exists(image_path):
                # Open the image using PIL
                original_image = Image.open(image_path)

                # Resize the image
                resized_image = original_image.resize(target_size)

                # Save the resized image (overwrite the original image)
                resized_image.save(image_path)

                print(f"Resized image in folder {folder_name}")
            else:
                print(f"Image not found in folder {folder_name}")
        else:
            print(f"Folder not found: {folder_name}")

def Preprocess(root_directory,data_folder,patch_size):
    width, height = 600,900
    for folder_name in range(1, 361):
        folder_path = os.path.join(root_directory, str(folder_name))
        if os.path.exists(folder_path):
            image_path = os.path.join(folder_path, '1.jpg')
            if os.path.exists(image_path):
                original_image =  cv2.imread(image_path,0)
                normalized_image = cv2.normalize(original_image, None, 0, 1.0,cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                start_x = np.random.randint(0, width - patch_size[0] + 1)
                start_y = np.random.randint(0, height - patch_size[1] + 1)
                patch = normalized_image[start_x:start_x + patch_size[0],start_y:start_y + patch_size[1] ]
                patch_image = Image.fromarray((patch * 255).astype(np.uint8))

                # Save the patch to the output folder
                
                patch_image.save(os.path.join(data_folder, str(folder_name)+'.png'))

                print("Saved to", folder_name)
                
        
        




