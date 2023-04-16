from flask import *
from flask import Flask, render_template, g
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.layers import LSTM, Dropout, Dense, BatchNormalization, Conv1D, MaxPooling1D, Bidirectional
from keras.layers import GRU, Dense



mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results


def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS) # Draw face connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Draw pose connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand connections

def draw_styled_landmarks(image, results):
    # # Draw face connections
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS, 
                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                            ) 
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             ) 
    
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
    # Draw right hand connections  
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             ) 

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([lh, rh, pose])




#normal





# Define a function to generate the video stream
def generate():
    
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,258)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(7, activation='softmax'))

    sequence = []
    sentence = []
    predictions = []
    sequence_length = 30
    threshold = 0.5
    
    actions = np.array(['APPLE', 'BANANA', 'COCONUT', 'MANGO', 'PINEAPPLE', 'STRAWBERRY',
       'WATERMELON'])
    
    
    
    
    cap = cv2.VideoCapture(0)
    with mp_holistic.Holistic(min_detection_confidence=0.1, min_tracking_confidence=0.1) as holistic:
        while True:
            
            category = "fruits"
            category = app.config['category']
                            
            lstmcategory = ['fruits','vegtables']
            grucategory = ['places','shapes','adjectives','drinks','weather','house']
            customgrucategory = ['school','pronouns','verbs','foods','body','clothes']


            model_path = os.path.join(os.path.dirname(__file__),'static','models','FRUITS','normal','accalphamodel45.h5')

            if category == 'adjectives':
                actions = np.array(['BEAUTIFUL', 'BIG', 'CUTE', 'DRY', 'EASY',
                                    'HANDSOME', 'HARD', 'HUNGRY', 'SHORT', 'SMALL',
                                    'SOFT', 'STRONG', 'TALL', 'TIRED', 'UGLY', 'WEAK'])

                model_path = os.path.join(os.path.dirname(__file__),'static','models','ADJECTIVES','gru','lossalphamodel45.h5')

            elif category == 'alphabet':
                actions = np.array(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
                                    'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
                                    'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'])

            elif category == 'body':
                actions = np.array(['ARM', 'BODY', 'CHEEK', 'CHIN', 'EARS', 
                                    'ELBOW', 'EYEBROWS', 'EYES', 'FACE', 'FEET',
                                    'FINGERS', 'HAIR', 'HAND', 'HEAD', 'LIPS',
                                    'MOUTH', 'NECK', 'NOSE', 'SHOULDER'])
                model_path = os.path.join(os.path.dirname(__file__),'static','models','BODY','gru','accalphamodel45.h5')

            elif category == 'clothes':
                actions = np.array(['BACKPACK', 'BAG', 'BELT', 'BRA', 'BRACELET',
                                    'BRIEF', 'CAP', 'DRESS', 'EARRING', 'EYEGLASS', 
                                    'HANKY', 'HAT', 'JACKET', 'PANTY','RUBBER',
                                    'SANDO', 'SHOES', 'SKIRT', 'SLIPPER', 'SOCKS', 'TSHIRT'])
                model_path = os.path.join(os.path.dirname(__file__),'static','models','CLOTHES','kent','accalphamodel45.h5')

            elif category == 'colors':
                actions = np.array(['BLACK', 'BLUE', 'GRAY', 'GREEN', 'ORANGE', 'PINK', 
                                    'PURPLE','RED', 'VIOLET', 'WHITE', 'YELLOW'])


            elif category == 'drinks':
                actions = np.array(['COFFEE', 'DRINK', 'DRINK MILK', 'DRINK WATER',
                                    'HOT CHOCOLATE', 'JUICE', 'SODA', 'TEA'])
                model_path = os.path.join(os.path.dirname(__file__),'static','models','DRINKS','gru','lossalphamodel45.h5')
            
            elif category == 'emotions':
                actions = np.array(['ANGRY', 'ANNOY', 'HAPPY', 'HUNGRY', 'LOVE', 'SAD', 
                                    'SCARED','SHOCK', 'SORRY', 'SURPRISE', 'WORRY'])

            elif category == 'family':
                actions = np.array(['BABY', 'BROTHER', 'DAUGHTER', 'FATHER', 'GRANDMA',
                                    'GRANDPA','MOTHER', 'RELATIVE', 'SISTER', 'SON'])

            elif category == 'food':
                actions = np.array(['BREAD', 'BREAKFAST', 'BUTTER', 'CEREAL', 'CHICKEN',
                                    'DINNER', 'EGG', 'FOOD', 'HAM', 'HONEY', 'HOTDOG',
                                    'LONGGANISA', 'LUNCH', 'MAYO', 'PANCAKE', 'RICE', 'TOCINO'])
                model_path = os.path.join(os.path.dirname(__file__),'static','models','FOODS','kent','accalphamodel45.h5')

            elif category == 'fruits':
                actions = np.array(['APPLE', 'BANANA', 'COCONUT', 'MANGO',
                                    'PINEAPPLE', 'STRAWBERRY', 'WATERMELON'])
                model_path = os.path.join(os.path.dirname(__file__),'static','models','FRUITS','normal','accalphamodel45.h5')

            elif category == 'house':
                actions = np.array(['BED', 'BRUSHING TEETH', 'CALENDAR', 'CEILING',
                                    'CLEAN', 'DOOR', 'FLOOR', 'PHONE', 'SHOWER',
                                    'TABLE', 'TAKE A BATH', 'WINDOW'])
                model_path = os.path.join(os.path.dirname(__file__),'static','models','HOUSE','gru','accalphamodel45.h5')


            elif category == 'number':
                actions = np.array(['ONE (1)', 'TEN (10)', 'ELEVEN (11)', 'TWELVE (12)',
                                    'THIRTEEN (13)', 'FOURTEEN (14)', 'FIFTEEN (15)',
                                    'SIXTEEN (16)', 'SEVENTEEN (17)', 'EIGHTEEN (18)',
                                    'NINETEEN (19)','TWO (2)', 'TWENTY (20)', 'THREE (3)', 'FOUR (4)',
                                    'FIVE (5)', 'SIX (6)', 'SEVEN (7)', 'EIGHT (8)',
                                    'NINE (9)'])

            elif category == 'places':
                actions = np.array(['BAKERY', 'BARBERSHOP', 'BRIDGE', 'CAFE',
                                    'CHURCH', 'CLINIC', 'COMPUTER SHOP', 'CONVENIENCE STORE',
                                    'FIRE STATION', 'HAIR SALON', 'HOSPITAL', 'MALL', 
                                    'MARKET', 'NEIGHBORHOOD', 'PHARMACY','PULIS STATION', 'SCHOOL'])
                model_path = os.path.join(os.path.dirname(__file__),'static','models','PLACES','gru','lastepochalpha.h5')

            elif category == 'pronouns':
                actions = np.array(['HE', 'HER', 'HERS', 'HIM', 'HIS', 'I', 'IT',
                                    'MINE', 'MY', 'OUR', 'SHE', 'THAT', 'THEIR',
                                    'THEM', 'THESE', 'THEY', 'THIS', 'THOSE',
                                    'US', 'WE', 'YOU', 'YOURS'])
                model_path = os.path.join(os.path.dirname(__file__),'static','models','PRONOUNS','kent','lastepochalpha.h5')


            elif category == 'school':
                actions = np.array(['BACKPACK', 'BAG', 'BOOK', 'ERASER', 'EYEGLASS', 
                                    'PAPER', 'PENCIL', 'RULER', 'STAPLER'])
                model_path = os.path.join(os.path.dirname(__file__),'static','models','SCHOOL','kent','accalphamodel45.h5')

            elif category == 'shapes':
                actions = np.array(['CIRCLE', 'DIAMOND', 'HEART', 'OVAL', 
                                    'RECTANGLE', 'SQUARE', 'STAR', 'TRIANGLE'])
                model_path = os.path.join(os.path.dirname(__file__),'static','models','SHAPES','gru','lossalphamodel45.h5')

            elif category == 'vegetables':
                actions = np.array(['AMPALAYA', 'BEANS', 'CABBAGE', 'ONION', 
                                    'SQUASH', 'TOMATO'])
                model_path = os.path.join(os.path.dirname(__file__),'static','models','VEGGIES','normal','accalphamodel45.h5')


            elif category == 'verb':
                actions = np.array(['ASK', 'BATH', 'BREAK', 'BRING', 'BUY',
                                    'CATCH', 'CLEAN', 'COME', 'COOK', 'CRY',
                                    'DANCE', 'DRAW', 'DRINK', 'EAT', 'FALL', 
                                    'FEEL', 'FIGHT', 'GET', 'GO', 'HEAR',
                                    'HELP', 'JUMP', 'LAUGH', 'LISTEN', 'LOOK',
                                    'MAKE', 'PAY', 'PRAY', 'READ', 'REST', 'RUN',
                                    'SEE', 'SING', 'SIT', 'STAND', 'STUDY',
                                    'WALK', 'WANT', 'WRITE'])
                model_path = os.path.join(os.path.dirname(__file__),'static','models','VERB','kent','lossalphamodel45.h5')

            elif category == 'weather':
                actions = np.array(['CLOUD', 'HOT', 'LIGHTING', 'RAIN', 'RAINBOW', 
                                    'STORM', 'SUN', 'THUNDER', 'TORNADO'])
                model_path = os.path.join(os.path.dirname(__file__),'static','models','WEATHER','gru','accalphamodel45.h5')

            
            print(actions.shape[0])
            # first case is normal lstm
            if category in lstmcategory :
    
                model = Sequential()
                model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(45,258)))
                model.add(LSTM(128, return_sequences=True, activation='relu'))
                model.add(LSTM(64, return_sequences=False, activation='relu'))
                model.add(Dense(64, activation='relu'))
                model.add(Dense(32, activation='relu'))
                model.add(Dense(actions.shape[0], activation='softmax'))
                
            #second case is base gru model
            elif category in customgrucategory:
                model = Sequential()
                model.add(GRU(128, return_sequences=True, activation='relu',input_shape=(45,258)))
                model.add(Dropout(0.1))
                model.add(Dense(64, activation='relu'))
                model.add(GRU(64, return_sequences=True, activation='relu'))
                model.add(Dropout(0.1))
                model.add(Dense(64, activation='relu'))
                model.add(GRU(32, return_sequences=False, activation='relu'))
                model.add(BatchNormalization())
                model.add(Dense(32, activation='relu'))
                model.add(Dense(actions.shape[0], activation='softmax'))
            
        
            #third case is configured gru mode; 
            elif category in grucategory:
                model = Sequential()
                model.add(GRU(64, return_sequences=True, activation='relu',input_shape=(45,258)))
                model.add(GRU(128, return_sequences=True, activation='relu'))
                model.add(GRU(64, return_sequences=False, activation='relu'))
                model.add(Dense(64, activation='relu'))
                model.add(Dense(32, activation='relu'))
                model.add(Dense(actions.shape[0], activation='softmax'))
           
                
            model.load_weights(model_path)
    
            ret, frame = cap.read()
            
            if not ret:
                break
            else:
                # Make detections
                image, results = mediapipe_detection(frame, holistic)
                
                # Draw landmarks
                draw_styled_landmarks(image, results)
                
                # 2. Prediction logic
                keypoints = extract_keypoints(results)
                sequence.append(keypoints)
                sequence = sequence[-sequence_length:]
                if len(sequence) == sequence_length :
                    res = model.predict(np.expand_dims(sequence, axis=0))[0]
                    
                    predictions.append(np.argmax(res))
                    
                #3. Viz logic
                    if np.unique(predictions[-10:])[0]==np.argmax(res): 
                        if res[np.argmax(res)] > threshold: 
                            
                            if len(sentence) > 0: 
                                if actions[np.argmax(res)]:
                                    sentence = actions[np.argmax(res)]
                            else:
                                sentence = actions[np.argmax(res)] 
                    #print(actions[np.argmax(res)])
                    app.config['translatedword'] = actions[np.argmax(res)] 
                    
                # Convert the frame to JPEG format
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                # Yield the frame in a Response object with the MIME type "multipart/x-mixed-replace; boundary=frame"
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')



def showvideo():
      
    cap = cv2.VideoCapture(1)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        else:
            # Convert the frame to JPEG format
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            # Yield the frame in a Response object with the MIME type "multipart/x-mixed-replace; boundary=frame"
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')



app = Flask(__name__)
app.config['translatedword'] = 'Translated Word'
app.config['category'] = 'none'

@app.route('/')
def home(): 
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    # Return the video stream in a Response object
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
    
    
    
@app.route('/get_video')
def get_video():    
    # Return the video stream in a Response object
    return Response(showvideo(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/get_data')
def get_data():
    data = app.config['translatedword']
    return str(data)

@app.route('/button-clicked')
def handle_button_click():
    button_info = request.args.get('info')
    # Do something with the button_info
    app.config['category'] = button_info
    return "Received button click: " + button_info


@app.route('/snt')
def snt():
    return render_template('snt.html')

@app.route('/tas')
def tas():
    return render_template('tas.html')




if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
