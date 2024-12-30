#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cv2 
from matplotlib import pyplot as plt
from scipy.linalg import inv
from scipy.interpolate import RectBivariateSpline
from skimage.util import img_as_float


# In[2]:


def getCircleContour(centre=(0, 0), r=(1, 1), N=200):
    #x = 500 + 500*np.cos(t)
    #y = 500 + 500*np.sin(t)
    #shape=(400, 2)
    t = np.linspace(0, 2 * np.pi, N) # 參數t, [0,2π]
    x = centre[0] + r[0] * np.cos(t)
    y = centre[1] + r[1] * np.sin(t)
    return np.array([x, y]).T


# In[3]:


def sobel(Gaussian_img):
    dx = cv2.Sobel(Gaussian_img, cv2.CV_16S, 1, 0)  # X方向上取一階導數，16位有符號數，卷積核3x3
    dy = cv2.Sobel(Gaussian_img, cv2.CV_16S, 0, 1)
    E = dx**2 + dy**2
    T = np.max([abs(dx), abs(dy)])
    G=E/T
    return G


# In[4]:


def snake_shape_matrix(n,alpha,beta):
    #計算5對角循環矩陣
    a = np.roll(np.eye(n), -1, axis=0)+np.roll(np.eye(n), -1, axis=1) -2*np.eye(n)
    b = np.roll(np.eye(n), -2, axis=0)+np.roll(np.eye(n), -2, axis=1)-4*np.roll(np.eye(n), -1, axis=0) -         4*np.roll(np.eye(n), -1, axis=1)+6*np.eye(n)
    M=-alpha*a + beta*b
    return M


# In[5]:


def writevideo(num):
    h=360
    w=360
    #fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    videowrite = cv2.VideoWriter(r'D:/python/computer_vison/computer_vison/HW4/result.mp4',-1,20,(int(h),int(w)))#-1讓它自己選編號器
    img_array=[None]
    for filename in [r'D:/python/computer_vison/computer_vison/HW4/{0}.jpg'.format(i) for i in range(num)]:
        pltimg = cv2.imread(filename)
        if pltimg is None:
            print(filename + " is error!")
            continue
        img_array.append(pltimg)
    for i in range(num):
        videowrite.write(img_array[i])
    videowrite.release()
    cv2.destroyAllWindows()
    print('successful')


# In[12]:


def active_contour(image, snake, alpha=0.01, beta=0.1,
                   w_line=0, w_edge=1, gamma=0.01,
                   max_px_move=1.0,
                   max_iterations=2500, convergence=0.1):
    initial=snake.copy()
    max_iterations = int(max_iterations)
    convergence_order = 10
    img = img_as_float(image)
    height = img.shape[0]
    width = img.shape[1]
    #找出邊緣
    edge=sobel(img)
    img = w_line*img + w_edge*edge    
 
    # 為平滑度進行插值：
    intp = RectBivariateSpline(np.arange(img.shape[1]), #intp=rect(height,width,)
                               np.arange(img.shape[0]), #x軸和y軸對img.t插值
                               img.T, kx=2, ky=2, s=0)
 
    x, y = snake[:, 0].astype(float), snake[:, 1].astype(float) #(400,2)
    xsave = np.empty((convergence_order, len(x))) #convergence_order=10
    ysave = np.empty((convergence_order, len(x))) #np.empty=(10,n)隨機矩陣
    n = len(x) #n=400
    A = snake_shape_matrix(n,alpha,beta)
    inv_m =inv(A+gamma*np.eye(n))
    num=0
    for i in range(max_iterations):#圖像能量最小化
        if(i%5==0):
            num_img=str(num)+".jpg"
            arr=np.array([x, y]).T
            fig=plt.figure(figsize=(5, 5))
            plt.imshow(image,cmap="gray")
            plt.plot(arr[:, 0], arr[:, 1],'-g', lw=3)
            plt.plot(initial[:, 0], initial[:, 1], '--b', lw=3)
            plt.savefig(num_img)
            num+=1
            plt.close(fig)
        fx = intp(x, y, dx=1, grid=False) 
        fy = intp(x, y, dy=1, grid=False)
        xn = np.dot(inv_m, gamma*x + fx)    
        yn = np.dot(inv_m, gamma*y + fy)
        dx = max_px_move*np.tanh(xn-x)   
        dy = max_px_move*np.tanh(yn-y)   #tanh激活函數
        x += dx
        y += dy
        x[x<0] = 0 
        y[y<0] = 0 
        x[x>(height-1)] = height - 1
        y[y>(width-1)] = width - 1
        j = i % (convergence_order+1)   
        if j < convergence_order:       
            xsave[j, :] = x
            ysave[j, :] = y
        else:#每10次檢查一下收斂了沒                          
            dist = np.min(np.max(np.abs(xsave-x) +
                                 np.abs(ysave-y), 1))   #np.max(,1)每列最大值
                                                        #np.min()找出得到的列中最小值
            if dist < convergence: #每列最大值中的最小值<0.1收斂
                writevideo(num)
                break
 
    return np.array([x, y]).T


# In[14]:


src = cv2.imread("pic2.jpg", 0)
img = cv2.GaussianBlur(src, (3, 3), 5)
init = getCircleContour((500, 500), (500,500), N=400)
snake = active_contour(img, snake=init, alpha=0.2, beta=0.1, gamma=0.01, w_line=1, w_edge=10)
fig=plt.figure(figsize=(5, 5))
plt.imshow(img,cmap="gray")
plt.plot(snake[:, 0], snake[:, 1],'-g', lw=3)
plt.plot(init[:, 0], init[:, 1], '--b', lw=3)
plt.savefig("result.jpg")
plt.show()


# In[ ]:




