#Import for streamlit
import streamlit as st
import av
from tensorflow.keras.models import load_model
import numpy as np
from streamlit_webrtc import (
    RTCConfiguration,
    VideoProcessorBase,
    WebRtcMode,
    webrtc_streamer,
    VideoTransformerBase
)

import cv2
from PIL import Image,ImageEnhance
import numpy as np
import os

import threading
from typing import Union


#Import for handling image
import cv2
from cvzone.FaceDetectionModule import FaceDetector


RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

#Create a dict for classes
emotion = emotions={
          0:'Angry',
          1:'Disgust',
          2:'Fear',
          3:'Happy',
          4:'Sad',
          5:'Surprise'}


#download model
@st.cache(allow_output_mutation=True)
def retrieve_model():

    model = load_model("caltech.h5")
    return model
#Main inelligence of the file, class to launch a webcam, detect faces, then detect emotion and output probability for each emotion

def app_emotion_detection():
    class EmotionPredictor(VideoProcessorBase):

        def __init__(self) -> None:
            # Sign detector
            self.face_detector = FaceDetector(    )
            self.model = retrieve_model()
            self.queueprediction = []
            self.frame_lock = threading.Lock()
            self.in_image = None
            self.out_image = None


        def img_convert(self,image):
            print(image.shape)
            image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            return image


        def predict(self,image,shape,reshape):

            img_resized = cv2.resize(image, shape).reshape(reshape)
            pred = self.model.predict(img_resized/255.)[0]
            return emotion[np.argmax(pred)], np.max(pred)

        def find_faces(self, image):
            image2 = image.copy()
            image_face, faces = self.face_detector.findFaces(image)



            # loop over all faces and print them on the video + apply prediction
            for face in faces:
                if face['score'][0] < 0.9:
                    continue

                SHAPE = (48, 48)
                RESHAPE = (1,48,48,1)

                xmin = int(face['bbox'][0])
                ymin = int(face['bbox'][1])
                deltax = int(face['bbox'][2])
                deltay = int(face['bbox'][3])

                start_point = (max(0,int(xmin - 0.3*deltax)),max(0,int(ymin - 0.3*deltay)))

                end_point = (min(image2.shape[1],int(xmin + 1.3*deltax)), min(image2.shape[0],int(ymin + 1.3*deltay)))

                im2crop = image2
                im2crop = im2crop[start_point[1]:end_point[1],start_point[0]:end_point[0]]
                im2crop = self.img_convert(im2crop)
                from PIL import Image
                im = Image.fromarray(im2crop)
                im.save("your_file.jpeg")


                prediction,score = self.predict(im2crop,SHAPE,RESHAPE)
                print(prediction, score)
                self.queueprediction.append((prediction,score))

                if len(self.queueprediction)>10:
                    self.queueprediction = self.queueprediction[-10:]
                    print(self.queueprediction)

                emotions_dict =  {
                                    'Angry': 0,
                                    'Disgust':0,
                                    'Fear': 0,
                                    'Happy':0,
                                    'Sad':0,
                                    'Surprise':0}

                for element in self.queueprediction:
                    emotions_dict[element[0]] +=1
                print(emotions_dict)

                # #draw emotion on images
                cv2.putText(image2, f'{prediction} {str(score)}', (start_point[0]-50, start_point[1]-30), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                 2, (255, 255, 255), 2)


                #draw rectangle arouond face
                cv2.rectangle(image2, start_point, end_point,(255,255,255), 2)

            return faces, image2


        
        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            image = frame.to_ndarray(format="rgb24")
            faces, annotated_image = self.find_faces(image)
            return av.VideoFrame.from_ndarray(annotated_image, format="rgb24")
   

    class VideoTransformer(VideoTransformerBase):

        frame_lock: threading.Lock  # `transform()` is running in another thread, then a lock object is used here for thread-safety.
        in_image: Union[np.ndarray, None]
        out_image: Union[np.ndarray, None]


        def transform(self, frame: av.VideoFrame) -> np.ndarray:
            in_image = frame.to_ndarray(format="bgr24")

            out_image = in_image[:, ::-1, :]  # Simple flipping for example.

            with self.frame_lock:
                self.in_image = in_image
                self.out_image = out_image

            return out_image

    ctx = webrtc_streamer(
        key="emotion-detection",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=EmotionPredictor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

 
    ctx = webrtc_streamer(key="snapshot", video_transformer_factory=VideoTransformerBase)

#     if ctx.video_transformer:
#         if st.button("Snapshot"):
#             with ctx.video_transformer.frame_lock:
#                 in_image = ctx.video_transformer.in_image
#                 out_image = ctx.video_transformer.out_image

#             if in_image is not None and out_image is not None:
#                 st.write("Input image:")
#                 st.image(in_image, channels="BGR")
#                 st.write("Output image:")
#                 st.image(out_image, channels="BGR")
#             else:
#                 st.warning("No frames available yet.")
 
 
    
############################ About #################################################

st.markdown(f'''
# Homepage (Intro)

Created: March 8, 2022 10:43 AM

# Happy, with 20% chance of sadness

> Deep learning for AI facial detector
> 

### Helping Artificial Intelligence connect better to how we feel

Artificial Intelligence technology is developing fast. Whist AI technologies stride to improve efficiency in our everyday lives, the soft side[not sure] of AI is still falling behind. Since we will be interacting with computers more than ever, we see that it is crucial to develop AI that communicates smoothly to us just like another human. This allows the endless possibilities to advance AI applications in areas such as caring for elderlies or detecting drunk drivers. As a result, we spun off a Facial Expression Detector model. The model is trained by deep learning CNNs model [[and VGG16 transfer learning?]] to detect human emotions from the camera.

### Try it out yourself

Navigate to the ‘Live Camera’ section 👈 for the show!

### About our model

CNNs model is trained with tensorflow [VGG16](https://www.tensorflow.org/api_docs/python/tf/keras/applications/vgg16/VGG16) transfer learning model. We use is [FER - CK+ - KDEF](https://www.kaggle.com/sudarshanvaidya/corrective-reannotation-of-fer-ck-kdef) ********dataset which ********contains 32,900 + images including 8 emotion categories – anger, contempt, disgust, fear, happiness, neutrality, sadness and surprise.

For better results, we narrow down to 6 emotion categories – anger, disgust, fear, happiness, sadness and surprise.

We finally ended up with 70% training accuracy and XX testing accuracy.
''')

    
    
    
############################ Sidebar + launching #################################################

#object_detection_page = "Try our Emotional Live Detector!"

#app_mode = st.sidebar.selectbox(
#     "Choose the app mode",
#     [
#         object_detection_page,
#     ],
# )
# st.subheader(app_mode)
# if app_mode == object_detection_page:
#     app_emotion_detection()



def main():
    st.title('Face Detection App')
    st.text('Build with Streamlit and OpenCV')


    activities =['Upload your emotion!',"About", "Live Emotion Detector"]
    choice = st.sidebar.selectbox('Select Activity',activities)

    if choice=='Upload your emotion!':
        st.subheader('Emotion Detection')

        image_file=st.file_uploader('Upload Image',type=["jpg","png","jpeg"])

    elif choice == 'About':
        st.subheader('About')

    elif choice ==  "Live Emotion Detector":
        app_emotion_detection()

        


        
        
      
if __name__ == '__main__':
    main()



#     elif choice ==  "Live Emotion Detector":
#         object_detection_page = "Live Emotion Detector"
#         app_mode = st.sidebar.selectbox(
#             "Live Detection",
#                 [
#                     object_detection_page,
#                 ],
#             )
#         st.subheader(app_mode)
#         if app_mode == object_detection_page:
#                 app_emotion_detection()


# if __name__ == '__main__':
#     main()
