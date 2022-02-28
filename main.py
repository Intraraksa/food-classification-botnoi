import tensorflow as tf
import cv2
import numpy as np

model = tf.keras.models.load_model("C:/Users/n_int/Desktop/botnoi  computer vision/food.h5")

with open('classes_name.txt','r') as f:
    classes_name = f.read().split('\n')

def predict_img(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,(224,224))
    img = tf.expand_dims(img,0)
    results = model.predict(img)
    food_class = np.argmax(results)
    if results[0][food_class] >= 0.95:
        return food_class
    else:
        return "Nothing"

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret,frame = cap.read()
    food_class = predict_img(frame)
    cv2.putText(frame,f"{classes_name[int(food_class)]}",(20,20),cv2.FONT_HERSHEY_COMPLEX,1,(200,10,150),1)
    cv2.imshow("video",frame)
    if cv2.waitKey(1)  == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()