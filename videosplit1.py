
# coding: utf-8

# In[23]:


import skvideo.io
import numpy as np
import os
import time
import shutil
import scipy.misc


# In[24]:


# path = 'C:\\Users\\Archit\\Desktop\\car6.mp4'
# videodata = skvideo.io.vreader(path) # to load video frame by frame.
# metadata = skvideo.io.ffprobe(path) #to find video metadata.
# metadata.keys()


# In[25]:


# metadata = metadata['video']


# In[26]:


# metadata


# In[27]:


# try:
#     if os.path.exists('data1'):
#         shutil.rmtree('data1', ignore_errors=True)
# except OSError:
#     print("error")


# In[28]:


def VideoToFrame(video):
    starttime = time.time()
    videodata = skvideo.io.vreader(video)
    metadata = skvideo.io.ffprobe(video)
    # print(metadata.keys())
    metadata = metadata['video']
    sizeOfFrame = (metadata['@width'], metadata['@height'])
    fps = metadata['@avg_frame_rate']
    totalFrames = int(metadata['@nb_frames'])
    videoLength = metadata['@duration']
    print('TotalFramesInVideo: ', totalFrames)
    # now we'll try to create a folder named data.
    # If folder already exist, delete that and create new one.
    # we'll be using shutil library to do above task.
    try:
        if os.path.exists('data'):
            shutil.rmtree('data', ignore_errors=True)
        os.makedirs('data')
    except OSError:
        print('Error while creating data')
    
    framesWeNeed = 10 # we'll be creating 10 frames at different intervals.
    interval = round(totalFrames / framesWeNeed)
    count = 0
    frame_no = 0
    for frame in videodata:
        if(count % interval == 0):
            frame_no += 1
            name = './data/frame' + str(frame_no) + '.jpg'
            scipy.misc.imsave(name, frame) # used to save numpy array as an image.
        count+=1
        
    print('TotalFrameGenerated: ', frame_no)
    
    endtime = time.time()
    print('TimeTaken = ', endtime - starttime)
    
    return ((float(videoLength)), str(totalFrames))


# In[29]:


# path = 'C:\\Users\\Archit\\Desktop\\car6.mp4'


# In[30]:


# VideoToFrame(path)

