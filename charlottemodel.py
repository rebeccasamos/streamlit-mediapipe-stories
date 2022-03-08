import streamlit as st
import av
from tensorflow.keras.models import load_model
import numpy as np
from streamlit_webrtc import (
    RTCConfiguration,
    VideoProcessorBase,
    WebRtcMode,
    webrtc_streamer,
)


#Import for handling image
import cv2
from cvzone.FaceDetectionModule import FaceDetector


RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

#Create a dict for classes
emotion = {0:'neutral',1:'happy',2:'sad',3:'surprise',4:'fear',
              5:'disgust',6:'anger',7:'contempt'}


#download model
@st.cache(allow_output_mutation=True)
def retrieve_model():

    model = load_model("/Users/rebeccasamossanchez/code/rebeccasamos/live-streaming-app/emotion-video-tuto/my_checkpoint_model_AffectNetVGG16_deep.h5")
    return model
#Main inelligence of the file, class to launch a webcam, detect faces, then detect emotion and output probability for each emotion

def app_emotion_detection():
    class EmotionPredictor(VideoProcessorBase):

        def __init__(self) -> None:
            # Sign detector
            self.face_detector = FaceDetector(    )
            self.model = retrieve_model()

        def find_faces(self, image):
            image2 = image.copy()
            image_face, faces = self.face_detector.findFaces(image)
            # loop over all faces and print them on the video + apply prediction
            for face in faces:
            #convert colour
                def img_convert(image):
                    image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
                    return image
                print(face)

            #make prediction
                def predict(image,shape,reshape):
                    img_resized = cv2.resize(image, shape).reshape(reshape)
                    pred = self.model.predict(img_resized/255.)[0]
                    return emotion[np.argmax(pred)]

                #set shape of image and the shape of the input
                SHAPE = (224,224)
                RESHAPE = (1,224,224,3)




                xmin = int(face['bbox'][0])
                ymin = int(face['bbox'][1])
                deltax = int(face['bbox'][2])
                deltay = int(face['bbox'][3])

                im2crop = image2
                im2crop = im2crop[int(ymin - 0.3*deltay):int(ymin + 1.3*deltay),int(xmin - 0.3*deltax):int(xmin + 1.3*deltax)]

                from PIL import Image
                im = Image.fromarray(im2crop)
                im.save("your_file.jpeg")


                prediction = predict(im2crop,SHAPE,RESHAPE)
                print(prediction)

                # #draw emotion on images
                cv2.putText(image2, prediction, (xmin -50, ymin -30), cv2.FONT_HERSHEY_PLAIN,
                                2, (255, 0, 255), 2)




                #draw rectangle arouond face
                # Start coordinate represents the top left corner of rectangle
                start_point = (xmin, ymin+deltay)

                # Ending coordinate represents the bottom right corner of rectangle
                end_point = (xmin+deltax, ymin)


                cv2.rectangle(image2, start_point, end_point,(204,255,204), 2)

            return faces, image2

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            image = frame.to_ndarray(format="rgb24")
            faces, annotated_image = self.find_faces(image)
            return av.VideoFrame.from_ndarray(annotated_image, format="rgb24")

    webrtc_ctx = webrtc_streamer(
        key="emotion-detection",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=EmotionPredictor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )


############################ Sidebar + launching #################################################

object_detection_page = "Emotion video detector"

app_mode = st.sidebar.selectbox(
    "Choose the app mode",
    [
        object_detection_page,
    ],
)
st.subheader(app_mode)
if app_mode == object_detection_page:
    app_emotion_detection()
