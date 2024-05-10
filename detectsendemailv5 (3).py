# import the required libraries
import cv2
import time
import smtplib
import numpy as np
import tflite_runtime
import tflite_runtime.interpreter as tflite
from email.mime.multipart import MIMEMultipart # for constructing email messages.
from email.mime.image import MIMEImage
from yolov5_tflite import yolov5_tflite
import picamera
# sender and recipient email credentials
sender_email = "officialpsri@gmail.com"
sender_password = "azbq sfce liuk qagk"
recipient_email = "psvictory5.trav@gmail.com"
# initialize video capture for Pi camera
cap = cv2.VideoCapture(0)
#camera = picamera.PiCamera()
with picamera.PiCamera() as camera:
    camera.resolution = (416,416)
    camera.framerate = 10
# Initialize GPIO pins
import RPi.GPIO as GPIO
GPIO.setmode(GPIO.BCM) # set the pin numbering mode to the broadcom SOC channel
firepin = 21
smoke_pin = 4
GPIO.setup(firepin, GPIO.IN)
GPIO.setup(smoke_pin, GPIO.IN)
fd = 0 # indicates no fire initially
# callback function for flame sensor pin changes
def flame_callback(channel):
    #global flame_detected
    flame_detected = GPIO.input(firepin)
    if flame_detected:
        fd = 1
        print(flame_detected)
        print("Flame detected!")
        capture_and_send_email("fire")
    else:
        pass
# read the initial state of the smoke pin and store in last_smoke_state
last_smoke_state = GPIO.input(smoke_pin)
# callback function for smoke sensor pin changes
def smoke_callback(channel):
    #global last_smoke_state
    # read the current state of the smoke pin
    smoke_state = GPIO.input(smoke_pin)
    # check if the smoke pin state has changed
    if smoke_state != last_smoke_state:
        # i.e. if smoke is detected
        if smoke_state == 0:
            sd = 1
            print(sd)
            print('Smoke (gas) detected!')
            capture_and_send_email("smoke")
    else:
        pass
# set up event detection on both pins for rising and falling edges(BOTH) with 300ms debounce time
GPIO.add_event_detect(firepin, GPIO.BOTH, callback=flame_callback, bouncetime=300)
GPIO.add_event_detect(smoke_pin, GPIO.BOTH, callback=smoke_callback, bouncetime=300)
# send an email with the captured image using SMTP
def send_email_with_capture(image_path):
    # create a MIMEMultipart object, set sender and recipient email addresses and subjects
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = recipient_email
    msg['Subject'] = "Fire or smoke detected!"
    # read image data from image_path and create a MIMEImage attachment
    with open(image_path, 'rb') as f:
        img_data = f.read()
    image = MIMEImage(img_data, name = image_path)
    msg.attach(image)
    # establish SMTP connection with Gmail server
    with smtplib.SMTP('smtp.gmail.com',587) as server:
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, recipient_email, msg.as_string())
# initialize yolov5 model
model = yolov5_tflite()
# capture an image and send email notification
def capture_and_send_email(sensor_type):
    # generate a timestamp using time.strftime
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S")
    image_path = f"{sensor_type}detection{timestamp}.jpg"
    # read a frame from the camera using cap.read()
    ret, frame = cap.read()
    if ret:
        # resize the frame to match the model's input size
        frame_resized = cv2.resize(frame, (416,416))
        # detect objects using YOLOv5 model
        boxes, scores, classnames = model.detect(frame_resized)
        # draw bounding boxes on the flipped frame
        for i, class_name in enumerate(classnames):
          if class_name in ['fire', 'smoke']:
            x, y, w, h = boxes[i]
            flipped_frame = cv2.flip(frame, -1)
            cv2.rectangle(flipped_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # capture the frame if fire/smoke is detected
            if i <= 2:
                cv2.imwrite(image_path, flipped_frame)
                # send email notifications with the captured frame
                send_email_with_capture(image_path)  # Pass the image path, not the results
            else:
                pass
    else:
        print('Error capturing image')

flame_detected = False
try:
    while True:
        # Read the current state of fire and smoke sensors
        current_fire_state = GPIO.input(firepin)
        current_smoke_state = GPIO.input(smoke_pin)
        # Check if there is a change in state for fire or smoke detection
        if flame_detected != current_fire_state or last_smoke_state != current_smoke_state:
            # Update the detected states
            flame_detected = current_fire_state
            last_smoke_state = current_smoke_state
            # Determine the sensor type based on the detected state
            sensor_type = 'fire' if flame_detected else 'smoke'
            # Call the function to capture and send an email
            capture_and_send_email(sensor_type)
            #time.sleep(1)
            break
except KeyboardInterrupt:
    print('Detection stopped by user.')
finally:
    # stop the video capture
    cap.release()
    cv2.destroyAllWindows()  # close any open OpenCV windows
    # clean all GPIO pins and reset to default state
    GPIO.cleanup()