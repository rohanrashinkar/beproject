from flask import Flask, render_template, Response, request, redirect, url_for, session
import numpy as np
import cv2
from keras.models import load_model
import time
import sqlite3
import secrets
from flask import flash
import json

app = Flask(__name__)
app.secret_key = secrets.token_hex(32)

detected_emotions = {0:'None',1:'None'}

def save_emotions(emotions):
    # Load existing data from the JSON file
    with open('lastemotion.json', 'r') as f:
        data = json.load(f)
    
    # Update the emotions in the data dictionary
    data['emotion1'] = emotions[0]
    data['emotion2'] = emotions[1]
    
    # Save the updated data back to the JSON file
    with open('lastemotion.json', 'w') as f:
        json.dump(data, f)

def load_emotions():
    # Load the emotions from the JSON file
    with open('lastemotion.json', 'r') as f:
        data = json.load(f)
    
    # Return the emotions as a dictionary
    return {
        0: data['emotion1'],
        1: data['emotion2']
    }

print('k')
camera = cv2.VideoCapture(0)  # use 0 for web camera
def gen_frames():  # generate frame by frame from camera
    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    model = load_model('emotion_recognition.h5')
    cap = cv2.VideoCapture(0)

    faceCascade = face_detector
    font = cv2.FONT_HERSHEY_SIMPLEX

    emotions = {0:'Angry',1:'Fear',2:'Happy',3:'Sad',4:'Surprised',5:'Neutral'}

    frame_count = 0

    while True:
        ret, frame = cap.read()
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        start_time = time.time()
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=7,
            minSize=(100, 100),)
        
        y0 = 15
        for index in range(6):
            cv2.putText(frame, emotions[index] + ': ', (5, y0), font,
                        0.4, (255, 0, 255), 1, cv2.LINE_AA)
            y0 += 15
       
        FIRSTFACE = True
        if len(faces) == 1:
            for x, y, width, height in faces:
                cropped_face = gray[y:y + height,x:x + width]
                test_image = cv2.resize(cropped_face, (48, 48))
                test_image = test_image.reshape([-1,48,48,1])

                test_image = np.multiply(test_image, 1.0 / 255.0)

                # Probablities of all classes
                #Finding class probability takes approx 0.05 seconds
                start_time = time.time()
                if frame_count % 5 == 0:
                    probab = model.predict(test_image)[0] * 100
                    #print("--- %s seconds ---" % (time.time() - start_time))

                    #Finding label from probabilities
                    #Class having highest probability considered output label
                    label = np.argmax(probab)
                    probab_predicted = int(probab[label])
                    predicted_emotion = emotions[label]
                    frame_count = 0

                frame_count += 1
                font_size = width / 300
                filled_rect_ht = int(height / 5)
                #Drawing probability graph for first detected face
                if FIRSTFACE:
                    y0 = 8
                    for score in probab.astype('int'):
                        cv2.putText(frame, str(score) + '% ', (80 + score, y0 + 8),
                                    font, 0.3, (0, 0, 255),1, cv2.LINE_AA)
                        cv2.rectangle(frame, (75, y0), (75 + score, y0 + 8),
                                      (0, 255, 255), cv2.FILLED)
                        y0 += 15
                        FIRSTFACE =False

                #Drawing rectangle and showing output values on frame
                print(predicted_emotion)
                # if (predicted_emotion=='Happy'):
                #         print('play happy song')
                # if  (predictdetected_emotions)
                #         print('play sad song')
                
                #Drawing rectangle and showing output values on frame
                detected_emotions[0] = predicted_emotion
                detected_emotions[1] = 'None'
                save_emotions(detected_emotions)
                print(detected_emotions[0] + 'This is detected text')

                cv2.rectangle(frame, (x, y), (x + width, y + height),(155,155, 0),2)
                cv2.rectangle(frame, (x-1, y+height), (x+1 + width, y + height+filled_rect_ht),
                              (155, 155, 0),cv2.FILLED)
                cv2.putText(frame, predicted_emotion+' '+ str(probab_predicted)+'%',
                            (x, y + height+ filled_rect_ht-10), font,font_size,(255,255,255), 1, cv2.LINE_AA)
            
     
        print(len(faces))
        if len(faces) == 2:
            emo=[]
            for x, y, width, height in faces:
                cropped_face = gray[y:y + height,x:x + width]
                test_image = cv2.resize(cropped_face, (48, 48))
                test_image = test_image.reshape([-1,48,48,1])

                test_image = np.multiply(test_image, 1.0 / 255.0)

                # Probablities of all classes
                #Finding class probability takes approx 0.05 seconds
                start_time = time.time()
                if frame_count % 5 == 0:
                    probab = model.predict(test_image)[0] * 100
                    #print("--- %s seconds ---" % (time.time() - start_time))

                    #Finding label from probabilities
                    #Class having highest probability considered output label
                    label = np.argmax(probab)
                    probab_predicted = int(probab[label])
                    predicted_emotion = emotions[label]
                    frame_count = 0

                frame_count += 1
                font_size = width / 300
                filled_rect_ht = int(height / 5)
                #Drawing probability graph for first detected face
                if FIRSTFACE:
                    y0 = 8
                    for score in probab.astype('int'):
                        cv2.putText(frame, str(score) + '% ', (80 + score, y0 + 8),
                                    font, 0.3, (0, 0, 255),1, cv2.LINE_AA)
                        cv2.rectangle(frame, (75, y0), (75 + score, y0 + 8),
                                      (0, 255, 255), cv2.FILLED)
                        y0 += 15
                        FIRSTFACE =False

                emo.append(predicted_emotion)
                cv2.rectangle(frame, (x, y), (x + width, y + height),(155,155, 0),2)
                cv2.rectangle(frame, (x-1, y+height), (x+1 + width, y + height+filled_rect_ht),
                              (155, 155, 0),cv2.FILLED)
                cv2.putText(frame, predicted_emotion+' '+ str(probab_predicted)+'%',
                            (x, y + height+ filled_rect_ht-10), font,font_size,(255,255,255), 1, cv2.LINE_AA)

               

                #Drawing rectangle and showing output values on frame
                
            print(emo)
            if ('Happy' in emo) and ('Sad' in emo):
                    print('play sad song')
                    
            if (emo[0]=='Happy') and (emo[1]=='Happy'):
                    print('Happy song')
            if (emo[0]=='Sad') and (emo[1]=='Sad'):
                    print('Sad song')
        
            detected_emotions[0] = emo[0]
            detected_emotions[1] = emo[1]
            save_emotions(detected_emotions)

            print(detected_emotions[0] + ' This is detected text ' + detected_emotions[1])

        ret, buffer = cv2.imencode('.jpg', frame)
        
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result
        if cv2.waitKey(1) == 27:
            break

        time.sleep(1)


@app.route('/video_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def Recommended_playlist():
    with open('emotion.json') as f:
        playlist_data = json.load(f)
    
    playlist = playlist_data['playlists']
    
    last_emotions = load_emotions()

    output = 'Neutral'
    playlist_id = None

    if (last_emotions[0] == 'None') and (last_emotions[1] == 'None'):
        print("Nothing to do here")
    elif (last_emotions[0] != 'None') and (last_emotions[1] == 'None'):
        output = last_emotions[0]
    elif (last_emotions[0] != 'None') and (last_emotions[1] != 'None'):
        emo0 = last_emotions[0]
        emo1 = last_emotions[1]
        if (emo0 == 'Happy' and emo1 == 'Happy'):
            output = 'Happy'
        elif (emo0 == 'Happy' and emo1 == 'Sad'):
            output = 'Neutral'
        elif (emo0 == 'Sad' and emo1 == 'Happy'):
            output = 'Neutral'
        elif (emo0 == 'Sad' and emo1 == 'Sad'):
            output = 'Sad'
        else:
            output = 'Neutral'

    for item in playlist:
        if item['emotion'] == output:
            playlist_id = item['id']

    return playlist_id



print("This is the :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::" + Recommended_playlist())

@app.route('/home')
def home():
    return render_template('home.html', playlist=Recommended_playlist())


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # Check if the user exists in the database
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE username = ? AND password = ?', (request.form['username'], request.form['password']))
        user = cursor.fetchone()
        conn.close()

        if user:
            # If the user exists, redirect them to the home page
            return redirect(url_for('home'))
        else:
            # If the user doesn't exist, show an error message
            return render_template('login.html', error='Invalid username or password')

    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        # Check if the username is already taken
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE username = ?', (request.form['username'],))
        user = cursor.fetchone()

        if user:
            # If the username is taken, show an error message
            conn.close()
            return render_template('register.html', error='Username already taken')

        # If the username is available, add the new user to the database
        cursor.execute('INSERT INTO users (username, password) VALUES (?, ?)', (request.form['username'], request.form['password']))
        conn.commit()
        conn.close()

        # Redirect the user to the login page
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/logout', methods=['GET', 'POST'])
def logout():
    session.pop('logged_in', None)
    flash('You were logged out')
    return redirect(url_for('index'))



if __name__ == "__main__":
  app.debug = True
  app.run()
