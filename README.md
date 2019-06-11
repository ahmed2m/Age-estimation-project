# Age Estimation project !

This is our project for Pattern Recognition CS-342 course.
The project contributors:
- **Ahmed Mohamadeen** [@ahmeed2m](https://git.io/ahmed)
- **Ali elrafei** [@alielrafeiFCIH](https://github.com/alielrafeiFCIH)
- **Ali Khaled** [@ali-khaled-elsayed](https://github.com/ali-khaled-elsayed)
- **Aly Moataz**  [@Aly-Moataz-Shorosh](https://github.com/Aly-Moataz-Shorosh)
- **Omar Farouk** [@Onsymers](https://github.com/Onsymers)
- **Marwan Bedeir** [@marwanBedeir](https://github.com/marwanBedeir)


## Brief
The problem is Age Estimation for the popular **[UTKFace](https://susanqq.github.io/UTKFace/)** data-set.<br>

The project has a gui interface for camera capturing and take the picture localize a face and give that face for out three implemented methods of estimating *KNN, SVM and CNN* trained from the previously mentioned data-set.<br>
For now the project only runs on linux due to the pickle files of the models only readable from a linux enviroment.

### Requirements
Temporary : Linux machine with Python 3 or later<br>
python libraries:
```
tensorflow
opencv-contrib-python or opencv
Pillow
```
### How to run
Clone the project
>git clone https://github.com/Ahmeed2m/Age-estimation-project.git

Download the models pickles and tensorflow session from this link and put it in the matching folders <br>
<link will be added>
Open terminal/cmd in repo path and run

>python ./main.py

### Project preview

![preview_GIF](https://media.giphy.com/media/Qx5dC1X50WfAI2ETOl/giphy.gif)

### Accuracy details


|                |ACCURACY       | 
|----------------|---------------|
|KNN             |38%            | 
|SVM             |50%            | 
|CNN             |60%            | 

