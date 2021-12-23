#!/usr/bin/env python
# coding: utf-8

# In[23]:


import cv2
import numpy as np
import time
import mediapipe as mp
from flask import Flask,render_template,Response


# In[ ]:





# In[2]:


app=Flask(__name__,template_folder='index.html')


# In[3]:


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


# In[4]:


#cap = cv2.VideoCapture('D:/AI Trainer/pexels-mart-production-8837221.mp4')


# In[ ]:





# In[5]:


cap = cv2.VideoCapture(0)


# In[ ]:





# In[6]:


def calculate_angle(a,b,c):
    a=np.array(a)#first
    b=np.array(b)#mid
    c=np.array(c)#end
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    
    if angle >180.0:
        angle = 360-angle
        
    return angle 
    


# In[ ]:





# In[ ]:





# In[7]:



def generate_frames():
        
    #Squat counter
    counter=0
    stage=None


    #Set up mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while True:
            success,frame = cap.read()
            #frame= cv2.resize(frame,(1200,1000))
            
            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            
            # Make detection
            results = pose.process(image)
            
            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            
            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark
                
                #Get coordinates
                hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                
                #calculate angle
                angle = calculate_angle(hip,knee,ankle)
                
                # Visualize angle
                cv2.putText(image, str(angle), 
                            tuple(np.multiply(knee, [1288, 700]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
                
                #Squat counter logic
                if angle<130:
                    stage= "Down"
                if angle>175 and stage=="Down":
                    stage="Up"
                    counter+=1
                    print(counter)
                
            except:
                pass
            
            
            #Render curl counter
            #setup status box
            cv2.rectangle(image,(0,0),(225,73),(245,117,16),-1)
            
            #Rep data
            cv2.putText(image, 'REPS', (15,12), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter), (10,60), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,0), 1, cv2.LINE_AA)
            
            #stage data
            cv2.putText(image, 'STAGE', (65,12), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, stage, (60,60), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,0), 1, cv2.LINE_AA)
            
            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) )
            
            if not success:
                break
            else:
                ret,buffer=cv2.imencode('.jpg',image)
                image = buffer.tobytes()
                
            yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')                        
            
            if cv2.waitKey(1) == ord('q'):
                break



     


# In[8]:


@app.route('/')
def index():
    return render_template('D:/AI Trainer/index.html')


# In[9]:


@app.route('/video')
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')


# In[10]:


if __name__=="__main__":
    app.run(debug=True,use_reloader=False)


# In[11]:


cap.release()
cv2.destroyAllWindows() 


# In[13]:





# In[ ]:




