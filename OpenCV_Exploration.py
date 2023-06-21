#!/usr/bin/env python
# coding: utf-8

# # Context
# 
# In this exploration, we'll be leveraging three robust libraries: Python Imaging Library (PIL), Matplotlib, and OpenCV, each of which provides us with unique capabilities.
# 
# PIL and Matplotlib's image function read the image in RGB format, while OpenCV reads in BGR format. Therefore, a conversion from BGR to RGB is necessary for OpenCV's image to ensure consistency.
# 
# Next, we go a layer deeper into our RGB image, uncovering the separate Red, Green, and Blue channels. 
# 
# Finally, the image is converted from color to grayscale using the OpenCV's cvtColor function. The pixel values of the grayscale image are then printed to provide a 2D array of pixel intensities, enabling further image processing and analysis. 

# In[33]:


# Importing the necessary libraries
import numpy as np  # Used for mathematical operations
import matplotlib.pyplot as plt  # Used for plotting graphs and images
from PIL import Image  # Python Imaging Library, used for opening, manipulating, and saving many different image file formats
import cv2  # OpenCV library, used for computer vision tasks
import matplotlib.image as mpimg  # Matplotlib's image reading function


# In[9]:


# Reading an image using Python Imaging Library (PIL)
img = Image.open('W:/MayCooperStation/New Documents/Data Science and ML/FacialRecognition/data/test.jpg')


# In[10]:


# Displaying the PIL image object
img


# In[13]:


# Reading the image using matplotlib's image function, this returns a numpy array
img_mat = mpimg.imread("W:/MayCooperStation/New Documents/Data Science and ML/FacialRecognition/data/test.jpg")

# Displaying the image using matplotlib's imshow function
plt.imshow(img_mat)
plt.show()


# In[19]:


# read image using opencv : BGR (Blue Green and Red)
# Reading the image using OpenCV's imread function
# Note: OpenCV reads images in BGR format
img_cv = cv2.imread('W:/MayCooperStation/New Documents/Data Science and ML/FacialRecognition/data/test.jpg')
img_cv


# In[15]:


# displaying images
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[34]:


# Converting the BGR image to RGB format
img_cv_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)


# In[35]:


plt.imshow(img_mat) # rgb


# In[20]:


img_cv = cv2.imread("W:/MayCooperStation/New Documents/Data Science and ML/FacialRecognition/data/test.jpg")

# Convert from BGR to RGB
img_cv_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

# Displaying the RGB image
plt.imshow(img_cv_rgb)
plt.show()


# In[21]:


# Printing the shape of the RGB image, this should give us (height, width, num_channels)
img_mat.shape


# In[22]:


# Splitting the RGB image into its 3 channels
r = img_mat[:,:,0] # red array
g = img_mat[:,:,1] # green array
b = img_mat[:,:,2] # blue array


# In[23]:


# Printing the shape of each channel
r.shape,g.shape,b.shape


# In[24]:


# Displaying each channel separately. We use a grayscale colormap as each channel is a 2D array
plt.figure(figsize=(10,6))
plt.subplot(1,3,1)
plt.title('Red region')
plt.imshow(r,cmap='gray')

plt.subplot(1,3,2)
plt.title('Green region')
plt.imshow(g,cmap='gray')

plt.subplot(1,3,3)
plt.title('Blue region')
plt.imshow(b,cmap='gray')
plt.show()


# In[27]:


# Converting the RGB images to grayscale using OpenCV's cvtColor function
gray_MAT = cv2.cvtColor(img_mat,cv2.COLOR_RGB2GRAY)
gray_CV = cv2.cvtColor(img_cv,cv2.COLOR_BGR2GRAY)


# In[28]:


# Printing the shape of the grayscale images
gray_MAT.shape, gray_CV.shape


# In[37]:


# Displaying the grayscale images
plt.imshow(gray_MAT,cmap='gray')


# In[32]:


# Printing the pixel values of the grayscale image. This prints a 2D array of pixel intensities
gray_CV


# In[ ]:




