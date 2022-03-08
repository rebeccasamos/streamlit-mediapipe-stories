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
)


#Import for handling image
import cv2
from cvzone.FaceDetectionModule import FaceDetector


RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

#Create a dict for classes
emotion = {0:'anger',1:'contempt',2:'disgust',3:'fear',4:'happiness',
              5:'neutrality',6:'sadness',7:'surprise'}


#download model
@st.cache(allow_output_mutation=True)
def retrieve_model():

    model = load_model("/Users/rebeccasamossanchez/code/rebeccasamos/live-streaming-app/emotion-video-tuto/my_checkpoint_pn_30k_model.h5")
    return model
#Main inelligence of the file, class to launch a webcam, detect faces, then detect emotion and output probability for each emotion

def app_emotion_detection():
    class EmotionPredictor(VideoProcessorBase):

        def __init__(self) -> None:
            # Sign detector
            self.face_detector = FaceDetector(    )
            self.model = retrieve_model()

        def find_faces(self, image):

            image_face, faces = self.face_detector.findFaces(image)
            # loop over all faces and print them on the video + apply prediction
            for face in faces:
            #convert colour
                def img_convert(image):
                    image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
                    return image

            #make prediction
                def predict(image,shape,reshape):
                    img_resized = cv2.resize(image, shape).reshape(reshape)
                    pred = self.model.predict(img_resized/255.)[0]
                    return emotion[np.argmax(pred)]

                #set shape of image and the shape of the input
                SHAPE = (224,224)
                RESHAPE = (1,224,224,3)

                prediction = predict(image,SHAPE,RESHAPE)
                print(prediction)


                # #draw emotion on images
               #cv2.putText(image_face, prediction, cv2.FONT_HERSHEY_PLAIN,
                                #2, (255, 0, 255), 2)





                            # Write some Text

                # font                   = cv2.FONT_HERSHEY_SIMPLEX
                # bottomLeftCornerOfText = (10,500)
                # fontScale              = 1
                # fontColor              = (255,255,255)
                # thickness              = 1
                # lineType               = 2

                # cv2.putText(image_face,str(prediction),
                #         bottomLeftCornerOfText,
                #         font,
                #         fontScale,
                #         fontColor,
                #         thickness,
                #         lineType)


            return faces, image_face

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



# #draw emotion on images
                  #cv2.putText(image_face, pred, (bbox[0] + 130, bbox[1] - 30), cv2.FONT_HERSHEY_PLAIN,
                                #l2, (255, 0, 255), 2)
