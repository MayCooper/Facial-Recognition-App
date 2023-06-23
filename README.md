# Facial-Recognition-App
AI-based Face Recognition Web Application with Flask &amp; Deployment: A facial recognition deep dive into greyscale, image processing with OpenCV, Eigen images &amp; theory. Utilizing Python, classification with SVMs, Flask (Jinja Template, HTML, CSS, HTTP Methods), pipeline model, Heroku &amp; more. 

The model development process involves creating a streamlined pipeline for data preprocessing, analysis, model training, and parameter tuning. 

Once developed, the face recognition model is integrated into a Flask application and then deployed to Heroku. The entire project is structured to ensure an end-to-end understanding of developing and deploying a machine learning-based web application, starting from scratch.

# A snapshot of the process 

#### Downloading dependencies: 
![image](https://github.com/MayCooper/Facial-Recognition-App/assets/82129870/eb16b50b-33c3-449d-b6a8-9c7774201970)

# EDA
#### Mapping out the image sizes
![image](https://github.com/MayCooper/Facial-Recognition-App/assets/82129870/2fc278cc-d892-4154-82e8-6e5bdb6c16c5)

#### Gender distribution
![image](https://github.com/MayCooper/Facial-Recognition-App/assets/82129870/f8523f3b-bb22-4008-a5b6-471d409110b8)

#### Exploring Grayscale for Facial Recognition
![image](https://github.com/MayCooper/Facial-Recognition-App/assets/82129870/285e212e-016c-46a4-956d-57b6ab3ea19a)

#### Creating a grayscale color map
![image](https://github.com/MayCooper/Facial-Recognition-App/assets/82129870/f86f49d5-2d13-4da6-92c1-b022ac8d5f45)

![image](https://github.com/MayCooper/Facial-Recognition-App/assets/82129870/3f446513-32a3-43e4-9a7a-d9753ab5892e)

####  Getting OpenCV, one of the most facial recognition modules
![image](https://github.com/MayCooper/Facial-Recognition-App/assets/82129870/24aeb5be-a6df-4ade-89d1-740eaa53edbf)

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
![image](https://github.com/MayCooper/Facial-Recognition-App/assets/82129870/2a0e5bb1-aef2-468d-8a70-924220d5b2ea)

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

![image](https://github.com/MayCooper/Facial-Recognition-App/assets/82129870/9256edfc-c0b9-4116-ad97-67a39c54167e)

![image](https://github.com/MayCooper/Facial-Recognition-App/assets/82129870/183c764d-5141-4cec-88bc-59aa8f410cbe)

#### Not all side-faces go un-detected

![image](https://github.com/MayCooper/Facial-Recognition-App/assets/82129870/0876000b-4421-4a66-abf7-61782f4f6b77)

#### Example of masked face going un-detected
![image](https://github.com/MayCooper/Facial-Recognition-App/assets/82129870/8f9daa4a-2a2d-44db-b4d9-eef5a9b4b48b)

#### Creating the Flask App base template: 

![image](https://github.com/MayCooper/Facial-Recognition-App/assets/82129870/a6ddbd63-e6a5-44d9-b562-95d7f011c597)

#### Gender classification code for the Flask App: 

![image](https://github.com/MayCooper/Facial-Recognition-App/assets/82129870/8dff94cd-6513-4f2e-b1f0-6fe9cd7d87ab)

#### Homepage code

![image](https://github.com/MayCooper/Facial-Recognition-App/assets/82129870/1917fa07-2b9c-468f-ac1d-1ee486031e87)

#### Home page, on the Flask app

![image](https://github.com/MayCooper/Facial-Recognition-App/assets/82129870/a915e883-3723-43a9-8d85-e92a3e112335)

#### After some basic styling, Flask app results:

![image](https://github.com/MayCooper/Facial-Recognition-App/assets/82129870/1bb2c0d3-37e1-490a-ad2b-75db13f787c0)

![image](https://github.com/MayCooper/Facial-Recognition-App/assets/82129870/9fa50027-3f6b-4318-91a5-7591ca66ace3)

#### Testing the front-end gender classification model
#### Choosing file:
![image](https://github.com/MayCooper/Facial-Recognition-App/assets/82129870/256df335-8e82-4d24-ad02-3e82455a4ce9)

#### Execution & Result:

![image](https://github.com/MayCooper/Facial-Recognition-App/assets/82129870/02bacdd5-f834-4f88-9016-1d6c6f9c59a1)

#### Example of more results for females:
![image](https://github.com/MayCooper/Facial-Recognition-App/assets/82129870/98da7ee6-d30d-44b1-95d2-3d390f190ef3)
![image](https://github.com/MayCooper/Facial-Recognition-App/assets/82129870/ea7948be-bede-49e2-b3b0-a084e61ddee1)
![image](https://github.com/MayCooper/Facial-Recognition-App/assets/82129870/acc84165-0088-4db9-977f-b06b6ca0cf85)

#### Gender Classification for males:

![image](https://github.com/MayCooper/Facial-Recognition-App/assets/82129870/55a6e793-4abb-417e-8a48-64c5d78df6ed)
![image](https://github.com/MayCooper/Facial-Recognition-App/assets/82129870/dae202d7-2e95-4129-a8c0-79f76b94a8bb)
![image](https://github.com/MayCooper/Facial-Recognition-App/assets/82129870/82bc1982-f348-4199-b374-ac8de778f83b)




