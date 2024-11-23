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

&nbsp;

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

    - 

![wofyemf/dlc.PNG](https://github.com/parkgwanseok/Nvidia-AI-Specialist-Certification/blob/ccdd8f225c479ba673e35dda4c967893c10f943b/wofyemf/dlc.PNG) Open Video : Bringing up a video or photo of your choice<p>
                                                                                                                                                     Open Image Folder : Import folders containing pictures
                                                                                                                                                     
<table>
  <tr>
    <td>
      <img src="https://github.com/parkgwanseok/Nvidia-AI-Specialist-Certification/blob/ccdd8f225c479ba673e35dda4c967893c10f943b/wofyemf/dlc.PNG"/>
    </td>
    <td>
      <p>Open Video : Bringing up a video or photo of your choice</p>
      <p>Open Image Folder : Import folders containing pictures</p>
    </td>
  </tr>
</table>



<div style="display: flex; align-items: flex-start; width: 100%;">
  <img src="https://github.com/parkgwanseok/Nvidia-AI-Specialist-Certification/blob/ccdd8f225c479ba673e35dda4c967893c10f943b/wofyemf/dlc.PNG"" alt="설명" width="300" style="margin-right: 20px;">
  <div>
    <p>Open Video : Bringing up a video or photo of your choice</p>
    <p>Open Image Folder : Import folders containing pictures</p>
  </div>
</div>







