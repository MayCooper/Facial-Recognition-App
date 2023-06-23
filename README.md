# Facial-Recognition-App
AI-based Face Recognition Web Application with Flask &amp; Deployment: A facial recognition deep dive into greyscale, image processing with OpenCV, Eigen images &amp; theory. Utilizing Python, classification with SVMs, Flask (Jinja Template, HTML, CSS, HTTP Methods), pipeline model, Heroku &amp; more. 

The model development process involves creating a streamlined pipeline for data preprocessing, analysis, model training, and parameter tuning. 

Once developed, the face recognition model is integrated into a Flask application and then deployed to Heroku. The entire project is structured to ensure an end-to-end understanding of developing and deploying a machine learning-based web application, starting from scratch.

# A snapshot of the process 

#### Downloading dependencies: 
![image](https://github.com/MayCooper/Facial-Recognition-App/assets/82129870/cd147e4f-c137-4efe-98c8-3a26c29488e7)

# EDA
#### Mapping out the image sizes
![image](https://github.com/MayCooper/Facial-Recognition-App/assets/82129870/fcba1ab8-b09a-41da-a2f6-d6866d470bb5)

#### Gender distribution
![image](https://github.com/MayCooper/Facial-Recognition-App/assets/82129870/ae2a9b00-02d0-4cce-b653-909abe8ec680)

#### Exploring Grayscale for Facial Recognition
![image](https://github.com/MayCooper/Facial-Recognition-App/assets/82129870/f4595748-fd0d-426e-9a34-ced5db6c7e0a)

#### Creating a grayscale color map
![image](https://github.com/MayCooper/Facial-Recognition-App/assets/82129870/0e3dd4fa-95fa-46c5-a20a-2615f7366696)

![image](https://github.com/MayCooper/Facial-Recognition-App/assets/82129870/a53d1c32-c3ad-4671-bf20-73c766366f80)

####  Getting OpenCV, one of the most facial recognition modules
![image](https://github.com/MayCooper/Facial-Recognition-App/assets/82129870/67d8d20f-2d29-4108-949f-c4dbacffdd94)

#### Testing face detection
![image](https://github.com/MayCooper/Facial-Recognition-App/assets/82129870/43e80bea-6641-4554-a39c-ce6dd9c5d4fa)
![image](https://github.com/MayCooper/Facial-Recognition-App/assets/82129870/23389ac1-7470-4269-8495-19f15405308c)
![image](https://github.com/MayCooper/Facial-Recognition-App/assets/82129870/03acad35-25f0-484d-bdbb-521665619c70)

#### We will now be extracting female and male faces and putting them into their respective categories

![image](https://github.com/MayCooper/Facial-Recognition-App/assets/82129870/c2967730-3417-4a63-9f48-857c020780c7)

#### Cropping images
![image](https://github.com/MayCooper/Facial-Recognition-App/assets/82129870/9ad1960f-dda3-4837-b09f-2d58a3a50494)
![image](https://github.com/MayCooper/Facial-Recognition-App/assets/82129870/4264b43b-3528-48df-8649-8bf652a7f124)

# Using PCA & Eigen Images

![image](https://github.com/MayCooper/Facial-Recognition-App/assets/82129870/4e3aa470-8426-43fd-bc3d-9a36b4cf9538)

#### Using GridSearchCV for looping through params for the ML model
![image](https://github.com/MayCooper/Facial-Recognition-App/assets/82129870/e2a8ca66-f643-48d3-8744-39f686ad319e)
![image](https://github.com/MayCooper/Facial-Recognition-App/assets/82129870/bece8307-46c1-4cf1-a7c2-1c108065e158)

#### Checking ROC Curve for Model
![image](https://github.com/MayCooper/Facial-Recognition-App/assets/82129870/6b67e188-f23d-4307-85d0-e0b0bf130e1d)

#### Creating dataframe to represent the ML 
![image](https://github.com/MayCooper/Facial-Recognition-App/assets/82129870/d098e928-0461-4bab-9094-eba12078a4bc)

#### Flask app

![image](https://github.com/MayCooper/Facial-Recognition-App/assets/82129870/50dbc46a-f252-4755-8052-44a13931efbe)

# Unleashing Facial Recognition on Video

![image](https://github.com/MayCooper/Facial-Recognition-App/assets/82129870/e7781f76-f7a3-465c-ae5f-4cd284caed05)

# Known issues 
- Some issues that can be addressed in future projects:
  - Bright spots are considered faces sometimes
  - Some side faces not being picked up as a face 
  - Not every frame is picked up as a face (but this could depend on frame rate and tuning)
  - Some faces with strange masks on (i.e, Captain America mask) is not being detected as a possible face
  - A very-much covered face is undetectable, is it possible to create face detection with question marks for a human to review

![image](https://github.com/MayCooper/Facial-Recognition-App/assets/82129870/9508162f-9581-418e-84b4-3a1ada196a16)

#### Not all side-faces go un-detected

![image](https://github.com/MayCooper/Facial-Recognition-App/assets/82129870/4dc852f0-65f6-4100-9a49-a3fbb5c77e1f)


#### Example of masked face going un-detected
![image](https://github.com/MayCooper/Facial-Recognition-App/assets/82129870/44937e0a-b35a-4c8d-87ba-a94a15b4914d)

#### Creating the Flask App base template: 

![image](https://github.com/MayCooper/Facial-Recognition-App/assets/82129870/a9ed8e5a-4d59-47f5-974a-0000ad644179)

#### Gender classification code for the Flask App: 

![image](https://github.com/MayCooper/Facial-Recognition-App/assets/82129870/dcac4c9b-48f2-4a4e-956d-37f01871a020)


#### Homepage code

![image](https://github.com/MayCooper/Facial-Recognition-App/assets/82129870/6e7f7d85-b755-4e27-8e88-94b11d9966bd)


#### Home page, on the Flask app

![image](https://github.com/MayCooper/Facial-Recognition-App/assets/82129870/1492ebfe-1cbf-4325-983e-8606fb873a1a)


#### After some basic styling, Flask app look & feel:

![image](https://github.com/MayCooper/Facial-Recognition-App/assets/82129870/8336e3cc-e464-4056-9469-c82a7824e58d)


![image](https://github.com/MayCooper/Facial-Recognition-App/assets/82129870/721f53a2-b0f2-40fa-851d-d1f3a4a84010)


#### Testing the front-end gender classification model
#### Choosing file:
![image](https://github.com/MayCooper/Facial-Recognition-App/assets/82129870/7bab8f87-47e0-469d-bc3d-a4ab2d7166a9)


#### Execution & Result:

![image](https://github.com/MayCooper/Facial-Recognition-App/assets/82129870/2227ca63-da0f-4b20-a59b-b86e6eddc18c)


#### Example of more results for females:

![image](https://github.com/MayCooper/Facial-Recognition-App/assets/82129870/ea7a434e-98c3-4dca-97d9-a12dd9ba6d89)

![image](https://github.com/MayCooper/Facial-Recognition-App/assets/82129870/bfcb6390-1cc3-4b41-a339-02263bce54ad)


#### Gender Classification for males:

![image](https://github.com/MayCooper/Facial-Recognition-App/assets/82129870/852cc3d6-7293-4dd7-bc35-fbbfc85b842d)

![image](https://github.com/MayCooper/Facial-Recognition-App/assets/82129870/f2b3e900-19c4-48b4-bcb2-bfb1eedcb1cd)

![image](https://github.com/MayCooper/Facial-Recognition-App/assets/82129870/47bfad70-f6fe-487e-b475-1bb146194b62)


