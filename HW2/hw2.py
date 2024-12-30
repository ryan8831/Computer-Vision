#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread('noise_image.png')
Height, width, channels = img.shape[:3]
Mean_img= np.zeros((Height-2, width-2, 1), np.uint8)
Median_img= np.zeros((Height-2, width-2, 1), np.uint8)
noise_image_his=np.zeros(shape=256)
output1_his=np.zeros(shape=256)
output2_his=np.zeros(shape=256)
a= np.zeros((3,3))


# In[2]:


# mean filter 
def Meanfilter(x,y,noiseimg):
    mean_value=0
    for i in range(0,3):
        for j in range(0,3):
            mean_value=mean_value+noiseimg[x+i,y+j,0]
    mean_value=mean_value//9
    return mean_value

for i in range(0,Height-2):
        for j in range(0,width-2):
            Mean_img[i,j,0]=Meanfilter(i,j,img)
cv2.imwrite('output1.png',Mean_img)   



# In[3]:


# median filter
def Medianfilter(x,y,sourceimg):
    
    for i in range(0,3):
            for j in range(0,3):
                a[i,j]=sourceimg[x+i,y+j,0]                
                
    for n in range(0,8):
        for i in range(0,3):
            for j in range(0,3):
                nc=i
                nr=j+1
                if(j==2):
                    nc=(i+1)
                    nr=0
                    if(nc==3):
                        break
                if(a[i,j]>a[nc,nr]):
                    temp=a[i,j]
                    a[i,j]=a[nc,nr]
                    a[nc,nr]=temp
    return(a[1,1])
for i in range(0,Height-2):
        for j in range(0,width-2):
            Median_img[i,j,0]=Medianfilter(i,j,img)
cv2.imwrite('output2.png',Median_img)


# In[4]:


for i in range(0,Height):
            for j in range(0,width):
                a=img[i,j,0]
                noise_image_his[a]=noise_image_his[a]+1
for i in range(0,Height-2):
            for j in range(0,width-2):
                out1=Mean_img[i,j,0]
                out2=Median_img[i,j,0]
                output1_his[out1]=output1_his[out1]+1
                output2_his[out2]=output2_his[out2]+1
num_list = [i for i in range(0, 256)]

plt.plot(num_list,noise_image_his)
plt.savefig("noise_image_his.png") 
plt.close()
plt.plot(num_list,output1_his)
plt.savefig("output1_his.png")
plt.close()
plt.plot(num_list,output2_his)
plt.savefig("output2_his.png")
plt.close()
                              


# In[5]:


plt.subplot(2,2,1)
plt.plot(num_list,noise_image_his)
plt.subplot(2,2,2)
plt.plot(num_list,output1_his)
plt.subplot(2,2,3)
plt.plot(num_list,output2_his)


# In[ ]:





# In[ ]:




