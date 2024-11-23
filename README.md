![wofyemf/dpsqleldk.png](https://github.com/parkgwanseok/Nvidia-AI-Specialist-Certification/blob/733305409633b12547864cc0097c1a149e5022e6/wofyemf/dpsqleldk.png)

## Nvidia-AI-Specialist-Certification

&nbsp;

### Automotive recognition system using images learned with YOLOv5

---

###  OverView of the Project

    - Opening background information

    - General description of the current project

    - Proposed idea for enhancements to the project

    - Value and significance of this project

    - Current limitations

    - Literature review

&nbsp;

## Opening background information

- My eyesight is not good and I can't see dark places well. That's why I chose self-driving, which is a recent trend.
Autonomous driving technology utilizes artificial intelligence-aware driving technology using LiDAR and cameras
Recognizing your surroundings in real time helps you operate efficiently in a variety of environments.

---

&nbsp;

## General description of the current project

- LiDAR and cameras allow you to recognize your surroundings in real time and drive efficiently in a variety of environments
It can be used in autonomous driving systems combined with vehicles and cameras by experimenting with recognition with vehicle license plates of various types of vehicles, and furthermore, it can also be used in ships and airplanes

---

&nbsp;

## Proposed idea for enhancements to the project

- In order for autonomous driving technology to develop and utilize,
high-resolution images, object recognition, and distance are required, and data can be processed in real time based on high accuracy and reliability to obtain information and respond in real time for stable operation

---

&nbsp;

## Value and significance of this project

- Artificial intelligence analyzes data collected through LiDAR and Cameras, identifies objects, and supports road driving with various license plate recognition technologies to enable safe and efficient autonomous driving

---

&nbsp;

## Current limitations

- It shows inaccurate accuracy when objects similar to fast speeds are present. It is also difficult to accurately recognize when the environment is bad or too dark

---

&nbsp;

## Literature review

- It is necessary to check the limitations of LiDAR and cameras, background knowledge of object recognition technology using YOLOv5, situation data in various environments, and improve accuracy

---

&nbsp;

## Image Acquisition Path

    - I used the black box of a car running on the road, and I used the image of the license plate of the car that was shared on the Internet

[driving video] https://drive.google.com/file/d/1UK55tGkfZkpgTms5ZBL82xMa6pkb9Pki/view?usp=drive_link

[verification  file] https://drive.google.com/drive/folders/1-1b3j8pdLvy9TnfkCcut5nrK5Nip7-Co?usp=drive_link

    - I adjusted it to a size of 640X640 for training data extraction

[Video Size Edit Site] https://www.adobe.com/kr/express/feature/video/resize

<img src="https://github.com/parkgwanseok/Nvidia-AI-Specialist-Certification/blob/a5a21c4ddd5ba003099a6d01a89175f304b16190/wofyemf/adv.PNG" width="70%" height="50%"><img src="https://github.com/parkgwanseok/Nvidia-AI-Specialist-Certification/blob/a5a21c4ddd5ba003099a6d01a89175f304b16190/wofyemf/adp.PNG" width="30%" height="50%">

&nbsp;

- DarkLabel

[DarkLabel Link] https://drive.google.com/drive/folders/1L7xtncy1xvE5L9mdQ9PvInY75AWFL8Dg?usp=drive_link

    - Use a dark label

![wofyemf/dlsc.PNG](https://github.com/parkgwanseok/Nvidia-AI-Specialist-Certification/blob/a5a21c4ddd5ba003099a6d01a89175f304b16190/wofyemf/dlsc.PNG)

&nbsp;

    - First, add a class through darklabel.yml

    - Add the license plate as a class name to the yam file

![wofyemf/dlcc.PNG](https://github.com/parkgwanseok/Nvidia-AI-Specialist-Certification/blob/a5a21c4ddd5ba003099a6d01a89175f304b16190/wofyemf/dlcc.PNG)

    - Enter the Licence-Plate class in classes_set for class verification

![wofyemf/dlyc.PNG](https://github.com/parkgwanseok/Nvidia-AI-Specialist-Certification/blob/ccdd8f225c479ba673e35dda4c967893c10f943b/wofyemf/dlyc.PNG)

    - You can see that the license plate has been added to number 10

![wofyemf/dl10.png](https://github.com/parkgwanseok/Nvidia-AI-Specialist-Certification/blob/ccdd8f225c479ba673e35dda4c967893c10f943b/wofyemf/dl10.png)
                                                                                                                         
<table style="border-collapse: collapse; width: 100%; border: none;">
  <tr style="border: none;">
    <td style="border: none; vertical-align: top; width: 50%;">
      <img src="https://github.com/parkgwanseok/Nvidia-AI-Specialist-Certification/blob/ccdd8f225c479ba673e35dda4c967893c10f943b/wofyemf/dlc.PNG" style="width: 100%;">
    </td>
    <td style="border: none; vertical-align: top; padding-left: 10px;">
      <p>Open Video : Bringing up a video or photo of your choice</p>
        <p>Open Image Folder : Import folders containing pictures</p>
    </td>
  </tr>
</table>

    - In the DarkLabel program, you can convert video into images frame by frame
    
    - Import video converted to 640x640 size
    
    - Get a video or image and label the license plate of the vehicle

![wofyemf/dllpc.PNG](https://github.com/parkgwanseok/Nvidia-AI-Specialist-Certification/blob/da0516d8505fbcf8598925a5ad33d6fcf335ff94/wofyemf/dllpc.PNG)

    - convert it into an image through as images and save it in the folder you want

    - labeled value is saved through GT save

<img src="https://github.com/parkgwanseok/Nvidia-AI-Specialist-Certification/blob/da0516d8505fbcf8598925a5ad33d6fcf335ff94/wofyemf/dlic.PNG" width="55%" height="50%"><img src="https://github.com/parkgwanseok/Nvidia-AI-Specialist-Certification/blob/da0516d8505fbcf8598925a5ad33d6fcf335ff94/wofyemf/dllc.PNG" width="45%" height="50%">

    - You can see that labeled text documents and image files are in the labels folder and the images folder, respectively
    
---

&nbsp;

## NVIDIA Jetson Nano Learning

    - First, connect to Google Drive using Google Colaboratory

    from google.colab import drive
    drive.flush_and_unmount()
    drive.mount('/content/drive')
    
&nbsp;

    - Install YOLOv5, Replicate the repository and install the package specified in the path
        
    !git clone https://github.com/ultralytics/yolov5
    %cd yolov5
    %pip install -qr requirements.txt
    
&nbsp;

    - Insert images and labeled values ​​into the images and labels folder in the Train folder to be trained

<img src="https://github.com/parkgwanseok/Nvidia-AI-Specialist-Certification/blob/da0516d8505fbcf8598925a5ad33d6fcf335ff94/wofyemf/gdi.PNG" width="50%" height="50%"><img src="https://github.com/parkgwanseok/Nvidia-AI-Specialist-Certification/blob/da0516d8505fbcf8598925a5ad33d6fcf335ff94/wofyemf/gdl.PNG" width="50%" height="50%">

    - Copy some of the Val image and labeling file data from the Train folder to create validation data

    import os
    import shutil
    from sklearn.model_selection import train_test_split

    def create_validation_set(train_path, val_path, split_ratio=0.3):

        os.makedirs(os.path.join(val_path, 'images'), exist_ok=True)
        os.makedirs(os.path.join(val_path, 'labels'), exist_ok=True)
    
        train_images = os.listdir(os.path.join(train_path, 'images'))
        train_images = [f for f in train_images if f.endswith(('.jpg', '.jpeg', '.png'))]
    
        _, val_images = train_test_split(train_images,
                                       test_size=split_ratio,
                                       random_state=42)

        for image_file in val_images:
        
            src_image = os.path.join(train_path, 'images', image_file)
            dst_image = os.path.join(val_path, 'images', image_file)
            shutil.copy2(src_image, dst_image)

            label_file = os.path.splitext(image_file)[0] + '.txt'
            src_label = os.path.join(train_path, 'labels', label_file)
            dst_label = os.path.join(val_path, 'labels', label_file)
            if os.path.exists(src_label):
                shutil.copy2(src_label, dst_label)

        print(f"Created validation set with {len(val_images)} images")

    train_path = '/content/drive/MyDrive/yolov5/yolov5/Train'
    val_path = '/content/drive/MyDrive/yolov5/yolov5/Val'

    create_validation_set(train_path, val_path)
        
&nbsp;

    - Adjust to center-based size while maintaining image ratios

    - Find all images, preprocess them, and save them in a numpy array

        import numpy as np
    import tensorflow as tf
    import os
    from PIL import Image
    from tensorflow.python.eager.context import eager_mode

    def _preproc(image, output_height=512, output_width=512, resize_side=512):
        ''' imagenet-standard: aspect-preserving resize to 256px smaller-side, then central-crop to 224px'''
        with eager_mode():
            h, w = image.shape[0], image.shape[1]
            scale = tf.cond(tf.less(h, w), lambda: resize_side / h, lambda: resize_side / w)
            resized_image = tf.compat.v1.image.resize_bilinear(tf.expand_dims(image, 0), [int(h*scale), int(w*scale)])
            cropped_image = tf.compat.v1.image.resize_with_crop_or_pad(resized_image, output_height, output_width)
            return tf.squeeze(cropped_image)

    def Create_npy(imagespath, imgsize, ext) :
        images_list = [img_name for img_name in os.listdir(imagespath) if
                    os.path.splitext(img_name)[1].lower() == '.'+ext.lower()]
        calib_dataset = np.zeros((len(images_list), imgsize, imgsize, 3), dtype=np.float32)

        for idx, img_name in enumerate(sorted(images_list)):
            img_path = os.path.join(imagespath, img_name)
            try:
                
                if os.path.getsize(img_path) == 0:
                    print(f"Error: {img_path} is empty.")
                    continue

                img = Image.open(img_path)
                img = img.convert("RGB")  
                img_np = np.array(img)

                img_preproc = _preproc(img_np, imgsize, imgsize, imgsize)
                calib_dataset[idx,:,:,:] = img_preproc.numpy().astype(np.uint8)
                print(f"Processed image {img_path}")

            except Exception as e:
                print(f"Error processing image {img_path}: {e}")

        np.save('calib_set.npy', calib_dataset)
            
&nbsp;

    - Edit for a class Save copies to the specified path

![wofyemf/dyc.PNG](https://github.com/parkgwanseok/Nvidia-AI-Specialist-Certification/blob/da0516d8505fbcf8598925a5ad33d6fcf335ff94/wofyemf/dyc.PNG)

    - Learn from specified data

    !python  /content/drive/MyDrive/yolov5/yolov5/train.py  --img 512 --batch 16 --epochs 300 --data /content/drive/MyDrive/yolov5/yolov5/data.yaml --weights yolov5n.pt --cache

    --img 512 : Set image size to 512X512

    --batch 16 : Set batch size to represent the number of images processed at a time

    --epochs 300 : sets the number of training epochs to 300

    --data /content/drive/MyDrive/yolov5/yolov5/data.yaml : Point to the data.yaml file that contains the configuration for the dataset

    --weights yolov5n.pt : Specify the path of a pre-trained weight file. Here we are using the yolov5n.pt file

    --cache : By caching images, you can speed up training if you have a large dataset

---
                
&nbsp;

## learning outcome

    - F1_curve                                                          - PR_curve
    
<img src="https://github.com/parkgwanseok/Nvidia-AI-Specialist-Certification/blob/da0516d8505fbcf8598925a5ad33d6fcf335ff94/wofyemf/F1_curve.png" width="50%" height="50%"><img src="https://github.com/parkgwanseok/Nvidia-AI-Specialist-Certification/blob/da0516d8505fbcf8598925a5ad33d6fcf335ff94/wofyemf/PR_curve.png" width="50%" height="50%">

    - P_curve                                                           - R_curve

<img src="https://github.com/parkgwanseok/Nvidia-AI-Specialist-Certification/blob/da0516d8505fbcf8598925a5ad33d6fcf335ff94/wofyemf/P_curve.png" width="50%" height="50%"><img src="https://github.com/parkgwanseok/Nvidia-AI-Specialist-Certification/blob/da0516d8505fbcf8598925a5ad33d6fcf335ff94/wofyemf/R_curve.png" width="50%" height="50%">
   
    - confusion_matrix
    
![wofyemf/confusion_matrix.png](https://github.com/parkgwanseok/Nvidia-AI-Specialist-Certification/blob/da0516d8505fbcf8598925a5ad33d6fcf335ff94/wofyemf/confusion_matrix.png)

    - labels                                                            - labels_correlogram

<img src="https://github.com/parkgwanseok/Nvidia-AI-Specialist-Certification/blob/da0516d8505fbcf8598925a5ad33d6fcf335ff94/wofyemf/labels.jpg" width="50%" height="50%"><img src="https://github.com/parkgwanseok/Nvidia-AI-Specialist-Certification/blob/da0516d8505fbcf8598925a5ad33d6fcf335ff94/wofyemf/labels_correlogram.jpg" width="50%" height="50%">

    - results

![wofyemf/results.png](https://github.com/parkgwanseok/Nvidia-AI-Specialist-Certification/blob/da0516d8505fbcf8598925a5ad33d6fcf335ff94/wofyemf/results.png)

    - val_batch1                                                        - val_batch2

<img src="https://github.com/parkgwanseok/Nvidia-AI-Specialist-Certification/blob/da0516d8505fbcf8598925a5ad33d6fcf335ff94/wofyemf/val_batch1_labels.jpg" width="50%" height="50%"><img src="https://github.com/parkgwanseok/Nvidia-AI-Specialist-Certification/blob/da0516d8505fbcf8598925a5ad33d6fcf335ff94/wofyemf/val_batch2_labels.jpg" width="50%" height="50%">

---
                
&nbsp;

## Verification of learning results

    - After the study is completed, the results are verified based on the image used on detect.py

    !python  /content/drive/MyDrive/yolov5/yolov5/detect.py --weights /content/drive/MyDrive/yolov5/yolov5/runs/train/exp5/weights/best.pt --img 512 --conf 0.1 --source /content/drive/MyDrive/rufrhk.mp4
          
&nbsp;

    !python  /content/drive/MyDrive/yolov5/yolov5/detect.py : Python interpreter to execute the detect.py script, Specify detect.py file path

    --weights /content/drive/MyDrive/yolov5/yolov5/runs/train/exp5/weights/best.pt : This weight is obtained by learning the model, which is a lane detection file that collects only the best lanes, Specify best.pt file path

    --img 512 : Set image size to 512X512

    --conf 0.1 : This sets the confidence threshold for object detection

    --source /content/drive/MyDrive/rufrhk.mp4 : File path to validate results

    - Images generated as a result of detection

![wofyemf/rjatk.png](https://github.com/parkgwanseok/Nvidia-AI-Specialist-Certification/blob/99e50edb47c1264361daa4cda49f793e8c7e5e27/wofyemf/rjatk.png)

![wofyemf/rjatk1.png](https://github.com/parkgwanseok/Nvidia-AI-Specialist-Certification/blob/99e50edb47c1264361daa4cda49f793e8c7e5e27/wofyemf/rjatk1.png)

![wofyemf/rjatk2.png](https://github.com/parkgwanseok/Nvidia-AI-Specialist-Certification/blob/99e50edb47c1264361daa4cda49f793e8c7e5e27/wofyemf/rjatk2.png)

    - Video produced through detect results

https://drive.google.com/file/d/1ByDt2fPHd5JuDLL6vktm53K7Q3iIuf3e/view?usp=drive_link

https://drive.google.com/file/d/1XBiIQVP-IgoGLv9eK-Brs089oBeabRSS/view?usp=drive_link

    - Jetson Nano Results Videos

https://drive.google.com/file/d/1Xz5ZCFCFptU1eAidHBsI7a468Jv_4XyI/view?usp=drive_link

https://drive.google.com/file/d/1OPz-v1Yz5FJE4rsQIs1NfS1nzsZB7zHs/view?usp=drive_link

---
          
&nbsp;

## conclusion

- The values learned using the license plate showed the graph while maintaining a high value.
  
  However, it shows inaccurate accuracy when there are objects that are similar to high speeds.
  
  It is also difficult to accurately recognize cases where the environment is bad or too dark.
  
  That is why we need to collect more data and many angles, multiple environments, and situation data that change in real time.
  
  I hope that we can make further progress and come out with safer and more efficient results for autonomous driving systems.

  ---
      
&nbsp;
