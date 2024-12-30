#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
img = cv2.imread('liberty.png')
Height, width, channels = img.shape[:3]
gray_mat= np.zeros((Height, width, 1), np.uint8)

ReLU_img= np.zeros((Height, width, 1), np.uint8)
kernel= np.array([(-1,-1,-1),(-1,8,-1),(-1,-1,-1)])

maxpooling_img=np.zeros(((Height)//2, (width)//2, 1), np.uint8)


# In[2]:


#grayscale image
for i in range(0,img.shape[0]):
        for j in range(0,img.shape[1]):
            r=img[i,j,2]
            g=img[i,j,1]
            b=img[i,j,0]
            gray_mat[i,j,0]=r*0.299+g*0.587+b*0.114
cv2.imwrite('gray.jpg', gray_mat) 


# In[3]:


def convolution(x,y,img):
    result=0
    for i in range(0,3):
        for j in range(0,3):
            result=result+img[x+i,y+j,0]*kernel[i][j]
    return result

for i in range(0,img.shape[0]-2):
    for j in range(0,img.shape[1]-2):
        temp=convolution(i,j,gray_mat)
        if(temp)<0 :
            temp=0
        if(temp)>255 :
            temp=255
        ReLU_img[i,j,0]=temp
cv2.imwrite('Reluimg.jpg', ReLU_img)  


# In[4]:


def maxpooling(x,y,img):
    max_value=0
    for i in range(0,2):
        for j in range(0,2):
            if(img[x+i][y+j]>max_value):
                max_value=img[x+i][y+j]

    return max_value


for i in range(0,img.shape[0]-1,2):
    for j in range(0,img.shape[1]-1,2):
        maxpooling_img[i//2,j//2,0]=maxpooling(i,j,ReLU_img)
cv2.imwrite('maxpooling.jpg',maxpooling_img)


# In[5]:


#binarization operation 
threshold = 128
for i in range(0,(img.shape[0])//2):
    for j in range(0,(img.shape[1])//2):
        if(maxpooling_img[i][j]>=threshold):
            maxpooling_img[i][j]=255
        else:
            maxpooling_img[i][j]=0
binaryimg=maxpooling_img
cv2.imwrite('binary.jpg',binaryimg)


# In[ ]:




