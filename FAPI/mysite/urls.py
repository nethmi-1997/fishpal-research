"""mysite URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from rest_framework.response import Response
from rest_framework.decorators import api_view
from django.conf.urls import url, include
import nltk, re, pprint, string
from nltk import word_tokenize, sent_tokenize
from nltk.util import ngrams
from nltk.corpus import stopwords
import nltk
from django.core.files.storage import FileSystemStorage
import cv2 
import argparse 
import os 
from scipy.spatial.distance import euclidean
from imutils import perspective
from imutils import contours
import numpy as np
import imutils
import cv2
import numpy as np
import pytesseract 
from PIL import Image 
from numpy import asarray
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from django.core.files.storage import FileSystemStorage
from django.shortcuts import render
from keras.applications.mobilenet import preprocess_input
from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import (ImageDataGenerator, img_to_array,
                                       load_img)
from sklearn.metrics import auc, confusion_matrix, roc_curve
from tensorflow import keras
import fileinput
import operator
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.eager.context import executing_eagerly_v1
from tensorflow.python.keras.callbacks import TensorBoard
import speech_recognition as sr 
import os 
from pydub import AudioSegment
from pydub.silence import split_on_silence
import moviepy.editor as mp
import speech_recognition as sr
import cv2 
import argparse 
import os 
import pytesseract 
from PIL import Image 
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
import cv2 
import numpy as np
from matplotlib import pyplot as plt
import tensorflow.compat.v1 as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format
from nltk.corpus import stopwords
import string
from nltk.stem import WordNetLemmatizer
from pathlib import Path
import subprocess

model_behaviour=load_model('models/behaviour_analyzer.h5')
model_color=load_model('models/water_detector.h5')
model_filter=load_model('models/filter_detector.h5')
model_foodAmount=load_model('models/food_amount.h5')
model_foodType=load_model('models/food_type.h5')

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file("objectdetection_config/pipeline.config")
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join("objectdetection_config/", 'ckpt-6')).expect_partial()

category_index = label_map_util.create_category_index_from_labelmap( 'objectdetection_config/label_map.pbtxt')


# Load pipeline config and build a detection model
configs_side = config_util.get_configs_from_pipeline_file("objectdetection_config_side/pipeline.config")
detection_model_side = model_builder.build(model_config=configs_side['model'], is_training=False)

# Restore checkpoint
ckpt_side = tf.compat.v2.train.Checkpoint(model=detection_model_side)
ckpt_side.restore(os.path.join("objectdetection_config_side/", 'ckpt-6')).expect_partial()

category_index_side = label_map_util.create_category_index_from_labelmap( 'objectdetection_config_side/label_map.pbtxt')



@api_view(['POST'])
def behaviour(request):
    print(request)
    print("video uploading************")
    fileObj=request.FILES['filePath']
    fs=FileSystemStorage()
    filePathName=fs.save(fileObj.name,fileObj)
    filePathName='media/'+filePathName
    for file in os.listdir('./media/behaviour/images/'):
        img_path='./media/behaviour/images/'+file
        os.remove(img_path)
    save_i_keyframes(filePathName)
    behaviour=detectBehaviour()
    return Response({"type":behaviour})


@api_view(['POST'])
def waterColor(request):
    print(request)
    print("image uploading************")
    fileObj=request.FILES['filePath']
    fs=FileSystemStorage()
    filePathName=fs.save(fileObj.name,fileObj)
    filePathName='media/'+filePathName
    detectedColor=detectColor(filePathName)
    return Response({"color":detectedColor})

@api_view(['POST'])
def topFish(request):
    print(request)
    print("image uploading************")
    fileObj=request.FILES['filePath']
    fs=FileSystemStorage()
    filePathName=fs.save(fileObj.name,fileObj)
    filePathName='media/'+filePathName
    count=detectObject(filePathName)
    return Response({"objectRes":count})

@api_view(['POST'])
def sideFish(request):
    print(request)
    print("image uploading************")
    fileObj=request.FILES['filePath']
    fs=FileSystemStorage()
    filePathName=fs.save(fileObj.name,fileObj)
    filePathName='media/'+filePathName
    count=detectSide(filePathName)
    return Response({"objectRes":count})

@api_view(['POST'])
def filter(request):
    print(request)
    print("image uploading************")
    fileObj=request.FILES['filePath']
    fs=FileSystemStorage()
    filePathName=fs.save(fileObj.name,fileObj)
    filePathName='media/'+filePathName
    detectedColor=detectFilter(filePathName)
    return Response({"color":detectedColor})


@api_view(['POST'])
def consumption(request):
    species=request.data['species']
 
    fishCount=request.data['fishCount']
   
    fishStage=request.data['fishStage']
 
    foodType=getFoodType(species,fishCount,fishStage)
    foodAmount=getFoodConsumption(species,fishCount,fishStage)
    return Response({"foodType":foodType,"foodAmount":round(foodAmount,1)})


@api_view(['POST'])
def fishLength(request):
    print(request)
    print("length image uploading************")
    fileObj=request.FILES['filePath']
    fs=FileSystemStorage()
    filePathName=fs.save(fileObj.name,fileObj)
    filePathName='media/'+filePathName
    length=findLength(filePathName)
    return Response({"objectRes":length})

def findLength(img):
    img_path = img

    # Read image and preprocess
    image = cv2.imread(img_path)
    height, width, channels = image.shape 
    contour_list=[]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)

    edged = cv2.Canny(blur, 50, 100)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)

    # Find contours
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    (cnts, _) = contours.sort_contours(cnts)

    cnts = [x for x in cnts if cv2.contourArea(x) > 100]

    ref_object = cnts[0]
    box = cv2.minAreaRect(ref_object)
    box = cv2.boxPoints(box)
    box = np.array(box, dtype="int")
    box = perspective.order_points(box)
    (tl, tr, br, bl) = box
    dist_in_pixel = euclidean(tl, tr)
    dist_in_cm = 2
    pixel_per_cm = dist_in_pixel/dist_in_cm

    for cnt in cnts:
        box = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(box)
        box = np.array(box, dtype="int")
        box = perspective.order_points(box)
        (tl, tr, br, bl) = box
        cv2.drawContours(image, [box.astype("int")], -1, (0, 0, 255), 2)
        mid_pt_horizontal = (tl[0] + int(abs(tr[0] - tl[0])/2), tl[1] + int(abs(tr[1] - tl[1])/2))
        mid_pt_verticle = (tr[0] + int(abs(tr[0] - br[0])/2), tr[1] + int(abs(tr[1] - br[1])/2))
        wid = euclidean(tl, tr)/pixel_per_cm
        ht = euclidean(tr, br)/pixel_per_cm
        contour_list.append(wid)
        print(ht)
        cv2.putText(image, "{:.1f}cm".format(wid), (int(mid_pt_horizontal[0] - 15), int(mid_pt_horizontal[1] - 10)), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        cv2.putText(image, "{:.1f}cm".format(ht), (int(mid_pt_verticle[0] + 10), int(mid_pt_verticle[1])), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
    length=max(contour_list)*(width/96)
    
    return round(length,1)

def get_frame_types(video_fn):
    command = 'ffprobe -v error -show_entries frame=pict_type -of default=noprint_wrappers=1'.split()
    out = subprocess.check_output(command + [video_fn]).decode()
    frame_types = out.replace('pict_type=','').split()
    return zip(range(len(frame_types)), frame_types)


def save_i_keyframes(video_fn):
    
    print(video_fn)
    frame_types = get_frame_types(video_fn)
    i_frames = [x[0] for x in frame_types if x[1]=='I']
    if i_frames:
        basename = os.path.splitext(os.path.basename(video_fn))[0]
        cap = cv2.VideoCapture(video_fn)
        for frame_no in i_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ret, frame = cap.read()
            outname = 'media/behaviour/images/_i_frame_'+str(frame_no)+'.jpg'
            cv2.imwrite(outname, frame)
            print ('Saved: '+outname)
        cap.release()
    else:
        print ('No I-frames in '+video_fn)

def detectBehaviour():
    path  = ('./media/behaviour/images')
    filenames = os.listdir(path)
    totaDetectedlImage=0
    normalImages=0
    hungryImages=0
    abnormalImages=0
    for file in filenames:
        img_path='./media/behaviour/images/'+file
        img = image.load_img(img_path, target_size=(224,224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        result1=model_behaviour.predict(x)
        list1 = result1.tolist()
        finalresult=list1[0]
        max_value = max(finalresult)
        max_index = finalresult.index(max_value)
        print(max_value)
        print(max_index)
        if(max_value>0.9):
            if(max_index==0):
                abnormalImages+=1
            elif (max_index==1):
                hungryImages+=1
            elif (max_index==2):
                normalImages+=1
    behaviourDict={"abnormal":abnormalImages,"normal":normalImages,"hungry":hungryImages}
    print(behaviourDict)
    
    print(max(behaviourDict.items(), key=operator.itemgetter(1))[0])
    return (max(behaviourDict.items(), key=operator.itemgetter(1))[0])
    # detectedBehaviour=behaviourDict.get(max(behaviourDict))
    # print(detectedBehaviour)
    # if detectedBehaviour==1:
    #     return "abnormal"
    # elif detectedBehaviour==2:
    #     return "hungry"
    # elif detectedBehaviour==3:
    #     return "normal"
    

def detectColor(img_path):
    img = image.load_img(img_path, target_size=(224,224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    result1=model_color.predict(x)
    list1 = result1.tolist()
    finalresult=list1[0]
    max_value = max(finalresult)
    max_index = finalresult.index(max_value)
    if max_index==0:
        return "BrownWater"
    elif max_index==1:
        return "DarkGreenWater"
    elif max_index==2:
        return "NormalWater"

def detectFilter(img_path):
    img = image.load_img(img_path, target_size=(224,224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    result1=model_filter.predict(x)
    list1 = result1.tolist()
    finalresult=list1[0]
    max_value = max(finalresult)
    max_index = finalresult.index(max_value)
    if max_index==0:
        return "BadConditionFilter"
    elif max_index==1:
        return "GoodConditionFilter"
 
def getFoodConsumption(species,fishCount,fishStage):
    GlobalReference.y_train=np.reshape(GlobalReference.y_train, (-1,1))
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    scaler_x.fit(GlobalReference.X_train)
    scaler_y.fit(GlobalReference.y_train)
    Xnew = [[1,species,fishCount,fishStage]]
    Xnew=scaler_x.transform(Xnew)
    ynew = model_foodAmount.predict(Xnew)
    ynew = scaler_y.inverse_transform(ynew)
    foodAmount=(round(ynew[0][0],2))
    return round(foodAmount,1)

def getFoodType(species,fishCount,fishStage):
    GlobalReference.y_train_type=np.reshape(GlobalReference.y_train_type, (-1,1))
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    scaler_x.fit(GlobalReference.X_train)
    scaler_y.fit(GlobalReference.y_train_type)
    Xnew = [[1,species,fishCount,fishStage]]
    Xnew=scaler_x.transform(Xnew)
    ynew = model_foodType.predict(Xnew)
    ynew = scaler_y.inverse_transform(ynew)
    foodCon=(round(ynew[0][0],0))
    print("foodCon"+str(foodCon))
    if foodCon==0:
        return "Cichlid Pellets"
    elif foodCon==1:
        return "Crushed Fish Flakes"
    elif foodCon==2:
        return "Decap Powder"
    elif foodCon==3:
        return "Fish Flakes"
    elif foodCon==4:
        return "Fish Powder"
    elif foodCon==5:
        return "Food Pellets"
     
@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
#     print(prediction_dict)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections


@tf.function
def detect_fn_side(image):
    image, shapes = detection_model_side.preprocess(image)
    prediction_dict = detection_model_side.predict(image, shapes)
#     print(prediction_dict)
    detections = detection_model_side.postprocess(prediction_dict, shapes)
    return detections

def detectObject(img_path):
    img = cv2.imread(img_path)
    image_np = np.array(img)
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)
    
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                for key, value in detections.items()}
    detections['num_detections'] = num_detections
    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    label_id_offset = 1
    image_np_with_detections = image_np.copy()
    viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'],
                detections['detection_classes']+label_id_offset,
                detections['detection_scores'],
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=20,
                min_score_thresh=.3,
                agnostic_mode=False,line_thickness=8)
    plt.figure(figsize = (8,90))
    plt.imshow(cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB))
    objects = []
    for index, value in enumerate(detections['detection_classes']):
        object_dict = {}
        if detections['detection_scores'][index] > .6:
            object_dict[(category_index.get(value+1)).get('name').encode('utf8')] = detections['detection_scores'][index]
            objects.append(object_dict)
    print("Fish count is")
    print(len(objects))
    return len(objects)
    
 

def detectSide(img_path):
    img = cv2.imread(img_path)
    image_np = np.array(img)
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn_side(input_tensor)
    
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                for key, value in detections.items()}
    detections['num_detections'] = num_detections
    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    label_id_offset = 1
    image_np_with_detections = image_np.copy()
    viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'],
                detections['detection_classes']+label_id_offset,
                detections['detection_scores'],
                category_index_side,
                use_normalized_coordinates=True,
                max_boxes_to_draw=20,
                min_score_thresh=.45,
                agnostic_mode=False,line_thickness=8)
    plt.figure(figsize = (8,90))
    plt.imshow(cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB))
    objects = []
    for index, value in enumerate(detections['detection_classes']):
        object_dict = {}
        if detections['detection_scores'][index] > .45:
            object_dict[(category_index_side.get(value+1)).get('name').encode('utf8')] = detections['detection_scores'][index]
            objects.append(object_dict)
    print("Fish count is")
    print(len(objects))
    return len(objects)

class GlobalReference:
    fireImage=0
    fallImage=0
    X_train=[[1.00e+00,2.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,8.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,9.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,2.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,1.10e+01,1.00e+03,2.00e+00],
            [1.00e+00,4.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,0.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,1.10e+01,5.00e+01,0.00e+00],
            [1.00e+00,1.00e+01,1.25e+03,1.00e+00],
            [1.00e+00,3.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,1.10e+01,1.50e+03,2.00e+00],
            [1.00e+00,3.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,2.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,2.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,2.00e+00,5.00e+02,2.00e+00],
            [1.00e+00,5.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,7.00e+00,5.00e+02,2.00e+00],
            [1.00e+00,0.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,1.10e+01,1.00e+03,2.00e+00],
            [1.00e+00,1.00e+01,1.00e+02,0.00e+00],
            [1.00e+00,6.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,1.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,4.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,6.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,7.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,5.00e+00,5.00e+02,2.00e+00],
            [1.00e+00,6.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,3.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,1.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,1.10e+01,5.00e+01,0.00e+00],
            [1.00e+00,7.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,5.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,3.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,2.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,9.00e+00,5.00e+02,2.00e+00],
            [1.00e+00,7.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,1.00e+01,1.00e+02,0.00e+00],
            [1.00e+00,0.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,5.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,8.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,6.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,1.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,6.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,4.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,5.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,5.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,9.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,8.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,9.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,2.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,5.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,1.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,3.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,6.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,1.10e+01,1.00e+02,0.00e+00],
            [1.00e+00,1.10e+01,5.00e+02,1.00e+00],
            [1.00e+00,1.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,1.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,6.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,1.00e+01,2.00e+02,0.00e+00],
            [1.00e+00,1.00e+01,5.00e+02,1.00e+00],
            [1.00e+00,8.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,1.10e+01,2.00e+02,0.00e+00],
            [1.00e+00,6.00e+00,5.00e+02,2.00e+00],
            [1.00e+00,1.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,2.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,6.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,0.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,8.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,1.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,1.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,5.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,9.00e+00,5.00e+02,2.00e+00],
            [1.00e+00,7.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,8.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,1.00e+01,1.00e+02,0.00e+00],
            [1.00e+00,1.10e+01,1.50e+03,2.00e+00],
            [1.00e+00,5.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,6.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,5.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,1.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,1.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,6.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,4.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,8.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,2.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,6.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,1.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,0.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,8.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,9.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,6.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,1.10e+01,5.00e+01,0.00e+00],
            [1.00e+00,2.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,1.00e+01,1.00e+03,1.00e+00],
            [1.00e+00,9.00e+00,5.00e+02,2.00e+00],
            [1.00e+00,1.00e+01,1.00e+03,2.00e+00],
            [1.00e+00,4.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,4.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,1.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,3.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,9.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,5.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,2.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,0.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,0.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,0.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,3.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,4.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,2.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,1.00e+01,1.25e+03,1.00e+00],
            [1.00e+00,8.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,1.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,1.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,7.00e+00,5.00e+02,2.00e+00],
            [1.00e+00,9.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,2.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,2.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,1.20e+01,2.00e+02,0.00e+00],
            [1.00e+00,7.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,2.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,1.10e+01,1.00e+02,0.00e+00],
            [1.00e+00,4.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,1.00e+01,1.00e+03,1.00e+00],
            [1.00e+00,1.00e+01,1.50e+03,2.00e+00],
            [1.00e+00,7.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,0.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,1.00e+01,1.00e+03,1.00e+00],
            [1.00e+00,5.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,1.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,0.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,8.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,8.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,9.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,1.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,1.00e+01,1.00e+03,1.00e+00],
            [1.00e+00,6.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,1.10e+01,2.00e+02,0.00e+00],
            [1.00e+00,5.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,1.10e+01,5.00e+01,0.00e+00],
            [1.00e+00,6.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,0.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,4.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,2.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,5.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,0.00e+00,5.00e+02,2.00e+00],
            [1.00e+00,1.00e+01,5.00e+01,0.00e+00],
            [1.00e+00,6.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,0.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,2.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,6.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,7.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,0.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,9.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,0.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,2.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,7.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,4.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,7.00e+00,5.00e+02,2.00e+00],
            [1.00e+00,2.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,2.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,1.00e+01,1.00e+03,1.00e+00],
            [1.00e+00,1.10e+01,1.00e+02,0.00e+00],
            [1.00e+00,2.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,8.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,1.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,3.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,7.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,6.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,9.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,0.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,1.00e+01,5.00e+01,0.00e+00],
            [1.00e+00,3.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,2.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,0.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,7.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,8.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,4.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,1.10e+01,1.00e+03,1.00e+00],
            [1.00e+00,1.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,6.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,4.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,4.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,0.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,6.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,6.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,9.00e+00,5.00e+02,2.00e+00],
            [1.00e+00,2.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,1.10e+01,2.00e+02,0.00e+00],
            [1.00e+00,1.10e+01,1.50e+03,2.00e+00],
            [1.00e+00,4.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,3.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,8.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,8.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,0.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,4.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,2.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,8.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,6.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,8.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,9.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,1.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,9.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,7.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,9.00e+00,5.00e+02,2.00e+00],
            [1.00e+00,1.10e+01,5.00e+02,2.00e+00],
            [1.00e+00,3.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,8.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,8.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,6.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,6.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,4.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,1.10e+01,1.00e+02,0.00e+00],
            [1.00e+00,1.00e+01,1.25e+03,1.00e+00],
            [1.00e+00,5.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,6.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,8.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,0.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,8.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,6.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,1.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,7.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,7.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,6.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,5.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,5.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,1.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,2.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,9.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,1.10e+01,1.00e+02,0.00e+00],
            [1.00e+00,4.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,2.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,6.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,2.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,4.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,5.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,6.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,9.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,9.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,1.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,2.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,2.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,5.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,8.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,1.00e+01,1.00e+03,2.00e+00],
            [1.00e+00,5.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,0.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,9.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,3.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,1.10e+01,5.00e+02,2.00e+00],
            [1.00e+00,8.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,9.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,0.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,7.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,0.00e+00,5.00e+02,2.00e+00],
            [1.00e+00,7.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,1.10e+01,1.25e+03,1.00e+00],
            [1.00e+00,1.00e+01,2.00e+02,0.00e+00],
            [1.00e+00,8.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,9.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,2.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,6.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,3.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,3.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,1.00e+01,1.00e+03,1.00e+00],
            [1.00e+00,2.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,1.10e+01,1.50e+03,2.00e+00],
            [1.00e+00,1.10e+01,1.25e+03,1.00e+00],
            [1.00e+00,1.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,2.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,1.10e+01,2.00e+02,0.00e+00],
            [1.00e+00,1.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,5.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,5.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,1.10e+01,1.25e+03,1.00e+00],
            [1.00e+00,8.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,1.00e+00,5.00e+02,2.00e+00],
            [1.00e+00,6.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,1.10e+01,1.00e+02,0.00e+00],
            [1.00e+00,7.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,1.10e+01,5.00e+01,0.00e+00],
            [1.00e+00,8.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,8.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,2.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,4.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,1.10e+01,5.00e+02,2.00e+00],
            [1.00e+00,1.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,5.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,5.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,3.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,4.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,6.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,4.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,1.00e+01,1.25e+03,1.00e+00],
            [1.00e+00,8.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,5.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,1.00e+01,1.25e+03,1.00e+00],
            [1.00e+00,1.10e+01,2.00e+02,0.00e+00],
            [1.00e+00,8.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,6.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,2.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,6.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,8.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,7.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,5.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,7.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,5.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,0.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,3.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,7.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,5.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,4.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,9.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,3.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,4.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,8.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,1.00e+01,1.00e+02,0.00e+00],
            [1.00e+00,2.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,8.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,8.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,1.00e+01,1.00e+03,2.00e+00],
            [1.00e+00,7.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,8.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,3.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,4.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,2.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,5.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,3.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,0.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,1.00e+01,5.00e+01,0.00e+00],
            [1.00e+00,5.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,1.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,5.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,1.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,9.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,4.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,4.00e+00,5.00e+02,2.00e+00],
            [1.00e+00,1.00e+01,5.00e+01,0.00e+00],
            [1.00e+00,2.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,9.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,7.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,6.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,2.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,1.10e+01,2.00e+02,0.00e+00],
            [1.00e+00,6.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,0.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,0.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,5.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,4.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,3.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,5.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,8.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,6.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,5.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,6.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,1.10e+01,5.00e+01,0.00e+00],
            [1.00e+00,6.00e+00,5.00e+02,2.00e+00],
            [1.00e+00,9.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,1.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,1.10e+01,1.00e+03,2.00e+00],
            [1.00e+00,1.10e+01,1.00e+03,2.00e+00],
            [1.00e+00,7.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,1.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,1.10e+01,1.25e+03,1.00e+00],
            [1.00e+00,9.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,1.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,8.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,9.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,0.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,8.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,9.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,4.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,6.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,0.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,0.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,1.00e+01,1.50e+03,2.00e+00],
            [1.00e+00,3.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,9.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,3.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,9.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,1.00e+01,1.00e+02,0.00e+00],
            [1.00e+00,5.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,1.10e+01,5.00e+02,2.00e+00],
            [1.00e+00,6.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,8.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,7.00e+00,5.00e+02,2.00e+00],
            [1.00e+00,5.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,8.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,1.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,2.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,5.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,7.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,5.00e+00,5.00e+02,2.00e+00],
            [1.00e+00,8.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,0.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,4.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,1.20e+01,2.00e+02,0.00e+00],
            [1.00e+00,8.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,9.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,6.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,1.00e+01,5.00e+01,0.00e+00],
            [1.00e+00,0.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,5.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,4.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,1.00e+01,5.00e+02,1.00e+00],
            [1.00e+00,7.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,5.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,9.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,5.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,1.00e+01,1.00e+02,0.00e+00],
            [1.00e+00,2.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,1.00e+01,5.00e+01,0.00e+00],
            [1.00e+00,1.00e+01,1.00e+02,0.00e+00],
            [1.00e+00,1.00e+01,1.00e+02,0.00e+00],
            [1.00e+00,1.00e+01,5.00e+02,1.00e+00],
            [1.00e+00,2.00e+00,5.00e+02,2.00e+00],
            [1.00e+00,4.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,1.00e+01,1.00e+03,2.00e+00],
            [1.00e+00,9.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,1.10e+01,1.00e+03,1.00e+00],
            [1.00e+00,1.00e+01,2.00e+02,0.00e+00],
            [1.00e+00,1.10e+01,1.50e+03,2.00e+00],
            [1.00e+00,6.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,8.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,1.10e+01,5.00e+02,1.00e+00],
            [1.00e+00,9.00e+00,5.00e+02,2.00e+00],
            [1.00e+00,1.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,1.10e+01,1.25e+03,1.00e+00],
            [1.00e+00,2.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,8.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,9.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,8.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,9.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,1.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,8.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,1.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,1.00e+01,1.00e+02,0.00e+00],
            [1.00e+00,8.00e+00,5.00e+02,2.00e+00],
            [1.00e+00,1.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,1.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,7.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,0.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,6.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,1.00e+01,5.00e+02,1.00e+00],
            [1.00e+00,3.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,0.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,6.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,1.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,3.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,5.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,1.00e+01,2.00e+02,0.00e+00],
            [1.00e+00,1.10e+01,1.00e+03,1.00e+00],
            [1.00e+00,3.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,3.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,8.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,3.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,7.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,0.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,1.00e+01,5.00e+01,0.00e+00],
            [1.00e+00,0.00e+00,5.00e+02,2.00e+00],
            [1.00e+00,1.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,9.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,9.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,7.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,2.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,7.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,3.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,1.00e+01,1.50e+03,2.00e+00],
            [1.00e+00,3.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,5.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,3.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,1.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,9.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,2.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,1.10e+01,5.00e+01,0.00e+00],
            [1.00e+00,0.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,0.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,0.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,9.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,8.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,1.00e+01,5.00e+01,0.00e+00],
            [1.00e+00,5.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,1.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,9.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,6.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,7.00e+00,5.00e+02,2.00e+00],
            [1.00e+00,8.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,1.00e+00,5.00e+02,2.00e+00],
            [1.00e+00,5.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,9.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,4.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,2.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,1.10e+01,1.00e+03,2.00e+00],
            [1.00e+00,0.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,3.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,9.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,2.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,1.00e+01,1.00e+03,1.00e+00],
            [1.00e+00,1.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,1.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,5.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,4.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,3.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,1.10e+01,1.00e+02,0.00e+00],
            [1.00e+00,1.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,5.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,0.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,3.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,8.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,1.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,0.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,3.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,5.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,3.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,1.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,1.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,6.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,6.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,3.00e+00,5.00e+02,2.00e+00],
            [1.00e+00,1.10e+01,1.00e+02,0.00e+00],
            [1.00e+00,2.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,1.00e+00,5.00e+02,2.00e+00],
            [1.00e+00,6.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,1.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,0.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,6.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,1.10e+01,1.25e+03,1.00e+00],
            [1.00e+00,0.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,8.00e+00,5.00e+02,2.00e+00],
            [1.00e+00,4.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,8.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,8.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,9.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,3.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,3.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,8.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,7.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,7.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,2.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,1.10e+01,1.25e+03,1.00e+00],
            [1.00e+00,8.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,3.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,1.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,1.00e+00,5.00e+02,2.00e+00],
            [1.00e+00,3.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,0.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,0.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,7.00e+00,5.00e+02,2.00e+00],
            [1.00e+00,1.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,1.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,8.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,7.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,8.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,4.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,2.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,1.00e+01,5.00e+02,1.00e+00],
            [1.00e+00,4.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,7.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,0.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,7.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,6.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,9.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,6.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,6.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,7.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,1.10e+01,1.00e+03,2.00e+00],
            [1.00e+00,0.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,2.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,6.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,2.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,2.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,1.10e+01,1.50e+03,2.00e+00],
            [1.00e+00,1.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,2.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,9.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,5.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,1.00e+01,1.00e+02,0.00e+00],
            [1.00e+00,1.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,2.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,4.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,1.00e+01,1.25e+03,1.00e+00],
            [1.00e+00,7.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,4.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,5.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,3.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,2.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,4.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,9.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,7.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,9.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,7.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,4.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,1.10e+01,1.00e+03,1.00e+00],
            [1.00e+00,5.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,1.10e+01,1.00e+03,2.00e+00],
            [1.00e+00,1.10e+01,1.00e+03,2.00e+00],
            [1.00e+00,5.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,9.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,0.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,1.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,1.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,0.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,2.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,8.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,9.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,1.00e+01,5.00e+01,0.00e+00],
            [1.00e+00,9.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,1.10e+01,1.00e+03,1.00e+00],
            [1.00e+00,6.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,2.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,0.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,1.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,1.00e+01,1.25e+03,1.00e+00],
            [1.00e+00,8.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,4.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,1.00e+01,5.00e+01,0.00e+00],
            [1.00e+00,9.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,7.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,1.00e+01,5.00e+01,0.00e+00],
            [1.00e+00,1.10e+01,1.50e+03,2.00e+00],
            [1.00e+00,2.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,2.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,1.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,2.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,7.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,7.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,7.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,5.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,3.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,3.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,4.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,5.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,9.00e+00,5.00e+02,2.00e+00],
            [1.00e+00,8.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,8.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,6.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,6.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,8.00e+00,5.00e+02,2.00e+00],
            [1.00e+00,1.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,4.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,8.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,7.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,1.00e+01,1.00e+03,2.00e+00],
            [1.00e+00,0.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,1.00e+01,1.00e+02,0.00e+00],
            [1.00e+00,6.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,1.10e+01,2.00e+02,0.00e+00],
            [1.00e+00,1.10e+01,1.00e+02,0.00e+00],
            [1.00e+00,1.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,5.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,0.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,2.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,9.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,1.00e+01,5.00e+01,0.00e+00],
            [1.00e+00,3.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,9.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,1.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,0.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,8.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,1.10e+01,5.00e+01,0.00e+00],
            [1.00e+00,1.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,1.10e+01,2.00e+02,0.00e+00],
            [1.00e+00,6.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,2.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,3.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,7.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,8.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,0.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,6.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,5.00e+00,5.00e+02,2.00e+00],
            [1.00e+00,5.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,8.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,8.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,6.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,2.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,9.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,6.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,2.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,5.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,7.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,1.00e+01,5.00e+01,0.00e+00],
            [1.00e+00,2.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,6.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,4.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,7.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,2.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,4.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,2.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,1.10e+01,1.00e+02,0.00e+00],
            [1.00e+00,6.00e+00,5.00e+02,2.00e+00],
            [1.00e+00,7.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,9.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,2.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,6.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,7.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,2.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,1.00e+01,1.00e+02,0.00e+00],
            [1.00e+00,1.00e+01,5.00e+01,0.00e+00],
            [1.00e+00,1.00e+01,1.00e+02,0.00e+00],
            [1.00e+00,8.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,1.10e+01,1.00e+03,2.00e+00],
            [1.00e+00,5.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,3.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,1.10e+01,5.00e+01,0.00e+00],
            [1.00e+00,0.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,4.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,3.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,7.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,1.00e+01,5.00e+01,0.00e+00],
            [1.00e+00,7.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,1.10e+01,2.00e+02,0.00e+00],
            [1.00e+00,9.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,1.00e+01,1.25e+03,1.00e+00],
            [1.00e+00,9.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,9.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,9.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,2.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,3.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,5.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,4.00e+00,5.00e+02,2.00e+00],
            [1.00e+00,8.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,8.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,4.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,1.00e+01,5.00e+02,1.00e+00],
            [1.00e+00,0.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,1.10e+01,1.00e+03,1.00e+00],
            [1.00e+00,2.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,1.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,9.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,4.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,9.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,1.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,1.10e+01,2.00e+02,0.00e+00],
            [1.00e+00,4.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,8.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,8.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,9.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,7.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,2.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,3.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,2.00e+00,5.00e+02,2.00e+00],
            [1.00e+00,6.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,1.00e+01,5.00e+02,2.00e+00],
            [1.00e+00,2.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,3.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,5.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,1.10e+01,1.50e+03,2.00e+00],
            [1.00e+00,5.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,1.10e+01,5.00e+02,2.00e+00],
            [1.00e+00,3.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,1.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,3.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,1.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,7.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,0.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,7.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,5.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,4.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,2.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,8.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,0.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,7.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,0.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,9.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,7.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,7.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,6.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,0.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,6.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,1.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,2.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,6.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,1.10e+01,2.00e+02,0.00e+00],
            [1.00e+00,4.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,9.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,8.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,9.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,8.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,2.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,6.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,7.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,4.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,5.00e+00,5.00e+02,2.00e+00],
            [1.00e+00,3.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,4.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,5.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,8.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,3.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,2.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,9.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,8.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,1.00e+01,5.00e+02,2.00e+00],
            [1.00e+00,9.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,1.10e+01,5.00e+02,2.00e+00],
            [1.00e+00,6.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,7.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,3.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,1.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,2.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,5.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,1.00e+01,1.00e+03,2.00e+00],
            [1.00e+00,3.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,7.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,5.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,3.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,0.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,1.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,6.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,7.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,4.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,0.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,5.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,4.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,1.00e+01,1.00e+03,2.00e+00],
            [1.00e+00,6.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,3.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,1.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,0.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,1.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,0.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,1.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,7.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,1.10e+01,1.00e+02,0.00e+00],
            [1.00e+00,5.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,8.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,4.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,9.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,1.10e+01,5.00e+02,1.00e+00],
            [1.00e+00,3.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,4.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,4.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,1.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,8.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,6.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,2.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,7.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,1.10e+01,1.00e+03,1.00e+00],
            [1.00e+00,8.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,1.00e+01,2.00e+02,0.00e+00],
            [1.00e+00,0.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,7.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,6.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,9.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,9.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,7.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,0.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,1.00e+01,2.00e+02,0.00e+00],
            [1.00e+00,1.10e+01,5.00e+01,0.00e+00],
            [1.00e+00,3.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,1.00e+01,5.00e+02,1.00e+00],
            [1.00e+00,6.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,3.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,5.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,6.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,3.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,3.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,2.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,7.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,8.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,0.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,1.10e+01,1.00e+03,2.00e+00],
            [1.00e+00,7.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,2.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,5.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,7.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,1.10e+01,2.00e+02,0.00e+00],
            [1.00e+00,4.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,1.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,0.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,0.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,9.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,3.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,1.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,0.00e+00,5.00e+02,2.00e+00],
            [1.00e+00,1.10e+01,1.25e+03,1.00e+00],
            [1.00e+00,2.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,1.10e+01,1.00e+03,2.00e+00],
            [1.00e+00,1.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,1.00e+01,5.00e+02,1.00e+00],
            [1.00e+00,5.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,1.00e+01,1.00e+03,1.00e+00],
            [1.00e+00,2.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,8.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,3.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,4.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,1.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,0.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,3.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,6.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,3.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,2.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,0.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,1.00e+01,5.00e+02,2.00e+00],
            [1.00e+00,2.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,2.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,8.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,9.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,0.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,1.00e+01,1.00e+03,1.00e+00],
            [1.00e+00,1.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,5.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,4.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,9.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,8.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,9.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,7.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,3.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,3.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,4.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,4.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,5.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,8.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,4.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,6.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,6.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,4.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,0.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,4.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,3.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,8.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,8.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,1.00e+01,1.00e+03,1.00e+00],
            [1.00e+00,1.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,8.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,2.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,3.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,7.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,9.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,9.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,1.10e+01,2.00e+02,0.00e+00],
            [1.00e+00,1.00e+01,2.00e+02,0.00e+00],
            [1.00e+00,4.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,7.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,5.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,6.00e+00,5.00e+02,2.00e+00],
            [1.00e+00,5.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,0.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,2.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,7.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,3.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,0.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,2.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,5.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,1.10e+01,5.00e+02,1.00e+00],
            [1.00e+00,1.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,2.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,1.00e+01,1.00e+03,2.00e+00],
            [1.00e+00,0.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,9.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,8.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,8.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,4.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,3.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,6.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,6.00e+00,5.00e+02,2.00e+00],
            [1.00e+00,8.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,1.10e+01,1.00e+02,0.00e+00],
            [1.00e+00,1.10e+01,1.25e+03,1.00e+00],
            [1.00e+00,7.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,9.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,6.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,0.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,1.10e+01,1.50e+03,2.00e+00],
            [1.00e+00,8.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,2.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,2.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,3.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,0.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,4.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,4.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,2.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,0.00e+00,5.00e+02,2.00e+00],
            [1.00e+00,0.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,3.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,0.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,9.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,7.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,2.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,5.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,7.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,8.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,1.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,0.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,1.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,6.00e+00,5.00e+02,2.00e+00],
            [1.00e+00,2.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,2.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,3.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,6.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,8.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,1.10e+01,5.00e+02,1.00e+00],
            [1.00e+00,5.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,1.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,8.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,9.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,1.10e+01,1.00e+02,0.00e+00],
            [1.00e+00,8.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,4.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,1.00e+01,1.00e+02,0.00e+00],
            [1.00e+00,2.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,6.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,3.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,3.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,6.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,5.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,9.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,3.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,3.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,1.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,4.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,9.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,4.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,9.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,1.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,1.00e+01,1.00e+03,2.00e+00],
            [1.00e+00,1.10e+01,1.00e+02,0.00e+00],
            [1.00e+00,4.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,1.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,2.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,3.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,1.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,9.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,1.00e+01,2.00e+02,0.00e+00],
            [1.00e+00,6.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,1.00e+01,1.25e+03,1.00e+00],
            [1.00e+00,2.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,9.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,7.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,1.00e+01,1.00e+03,2.00e+00],
            [1.00e+00,1.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,4.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,1.00e+01,1.00e+03,1.00e+00],
            [1.00e+00,7.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,1.10e+01,1.00e+03,2.00e+00],
            [1.00e+00,3.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,4.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,4.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,8.00e+00,5.00e+02,2.00e+00],
            [1.00e+00,7.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,6.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,5.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,2.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,8.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,0.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,0.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,1.00e+01,1.00e+03,2.00e+00],
            [1.00e+00,6.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,4.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,8.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,1.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,5.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,9.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,1.10e+01,1.00e+03,2.00e+00],
            [1.00e+00,9.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,7.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,2.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,1.10e+01,2.00e+02,0.00e+00],
            [1.00e+00,2.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,6.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,2.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,2.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,8.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,1.10e+01,1.00e+03,1.00e+00],
            [1.00e+00,9.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,7.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,1.00e+01,2.00e+02,0.00e+00],
            [1.00e+00,2.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,5.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,1.00e+01,1.50e+03,2.00e+00],
            [1.00e+00,9.00e+00,5.00e+02,2.00e+00],
            [1.00e+00,7.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,7.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,1.10e+01,5.00e+01,0.00e+00],
            [1.00e+00,6.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,1.10e+01,1.25e+03,1.00e+00],
            [1.00e+00,1.10e+01,2.00e+02,0.00e+00],
            [1.00e+00,2.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,1.10e+01,1.00e+03,1.00e+00],
            [1.00e+00,4.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,6.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,4.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,1.10e+01,1.00e+02,0.00e+00],
            [1.00e+00,2.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,8.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,4.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,2.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,3.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,3.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,3.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,4.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,5.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,1.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,7.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,1.00e+01,1.25e+03,1.00e+00],
            [1.00e+00,9.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,7.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,8.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,0.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,4.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,6.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,0.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,6.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,1.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,8.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,8.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,5.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,9.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,3.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,9.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,6.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,5.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,9.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,0.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,1.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,2.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,4.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,3.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,1.00e+01,1.00e+02,0.00e+00],
            [1.00e+00,2.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,1.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,5.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,2.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,1.10e+01,1.00e+03,2.00e+00],
            [1.00e+00,1.00e+01,1.00e+02,0.00e+00],
            [1.00e+00,6.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,4.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,0.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,3.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,8.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,1.00e+01,5.00e+02,1.00e+00],
            [1.00e+00,7.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,7.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,0.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,0.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,9.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,3.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,7.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,1.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,7.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,0.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,3.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,7.00e+00,5.00e+02,2.00e+00],
            [1.00e+00,3.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,9.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,1.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,8.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,8.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,7.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,8.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,1.10e+01,1.00e+03,1.00e+00],
            [1.00e+00,7.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,1.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,1.00e+01,1.25e+03,1.00e+00],
            [1.00e+00,5.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,1.10e+01,1.25e+03,1.00e+00],
            [1.00e+00,9.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,6.00e+00,5.00e+02,2.00e+00],
            [1.00e+00,4.00e+00,5.00e+02,2.00e+00],
            [1.00e+00,1.00e+01,1.00e+03,1.00e+00],
            [1.00e+00,1.00e+01,5.00e+02,2.00e+00],
            [1.00e+00,6.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,1.10e+01,1.50e+03,2.00e+00],
            [1.00e+00,7.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,3.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,2.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,3.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,7.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,9.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,7.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,1.00e+01,5.00e+02,1.00e+00],
            [1.00e+00,8.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,3.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,1.10e+01,5.00e+02,1.00e+00],
            [1.00e+00,1.10e+01,1.25e+03,1.00e+00],
            [1.00e+00,1.10e+01,1.00e+03,1.00e+00],
            [1.00e+00,9.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,1.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,8.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,2.00e+00,5.00e+02,2.00e+00],
            [1.00e+00,1.00e+01,5.00e+02,1.00e+00],
            [1.00e+00,6.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,2.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,2.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,7.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,1.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,7.00e+00,5.00e+02,2.00e+00],
            [1.00e+00,1.10e+01,1.00e+03,1.00e+00],
            [1.00e+00,5.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,6.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,1.00e+01,1.00e+03,1.00e+00],
            [1.00e+00,9.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,9.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,4.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,0.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,1.10e+01,1.25e+03,1.00e+00],
            [1.00e+00,6.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,9.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,5.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,1.10e+01,1.00e+03,1.00e+00],
            [1.00e+00,1.10e+01,5.00e+01,0.00e+00],
            [1.00e+00,0.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,4.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,0.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,1.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,1.10e+01,5.00e+02,1.00e+00],
            [1.00e+00,7.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,4.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,1.10e+01,5.00e+02,1.00e+00],
            [1.00e+00,9.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,1.10e+01,1.25e+03,1.00e+00],
            [1.00e+00,3.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,9.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,8.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,5.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,0.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,9.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,7.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,9.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,3.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,5.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,1.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,1.00e+01,1.00e+03,2.00e+00],
            [1.00e+00,2.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,4.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,1.20e+01,2.00e+02,0.00e+00],
            [1.00e+00,6.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,6.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,1.00e+01,1.25e+03,1.00e+00],
            [1.00e+00,1.10e+01,1.25e+03,1.00e+00],
            [1.00e+00,8.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,0.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,0.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,1.10e+01,5.00e+01,0.00e+00],
            [1.00e+00,5.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,1.10e+01,1.50e+03,2.00e+00],
            [1.00e+00,9.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,5.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,4.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,9.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,2.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,5.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,1.00e+01,5.00e+02,1.00e+00],
            [1.00e+00,1.10e+01,1.00e+03,1.00e+00],
            [1.00e+00,1.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,7.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,1.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,4.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,7.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,2.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,1.00e+01,5.00e+02,2.00e+00],
            [1.00e+00,3.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,8.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,1.10e+01,1.25e+03,1.00e+00],
            [1.00e+00,1.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,8.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,5.00e+00,5.00e+02,2.00e+00],
            [1.00e+00,5.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,8.00e+00,5.00e+02,2.00e+00],
            [1.00e+00,7.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,9.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,2.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,1.00e+01,1.00e+03,2.00e+00],
            [1.00e+00,8.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,1.10e+01,1.00e+02,0.00e+00],
            [1.00e+00,8.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,2.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,2.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,1.00e+01,1.50e+03,2.00e+00],
            [1.00e+00,1.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,7.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,7.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,9.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,0.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,1.10e+01,1.25e+03,1.00e+00],
            [1.00e+00,6.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,2.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,0.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,1.00e+01,1.00e+02,0.00e+00],
            [1.00e+00,7.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,5.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,6.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,6.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,0.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,4.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,1.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,1.00e+01,5.00e+02,1.00e+00],
            [1.00e+00,1.10e+01,5.00e+02,1.00e+00],
            [1.00e+00,5.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,4.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,7.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,4.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,5.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,0.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,0.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,8.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,3.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,0.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,2.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,1.00e+01,5.00e+02,2.00e+00],
            [1.00e+00,4.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,1.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,2.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,3.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,3.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,9.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,9.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,4.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,1.00e+01,1.00e+03,1.00e+00],
            [1.00e+00,8.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,3.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,2.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,0.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,3.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,8.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,9.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,7.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,1.10e+01,5.00e+02,1.00e+00],
            [1.00e+00,0.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,7.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,1.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,1.00e+01,1.00e+02,0.00e+00],
            [1.00e+00,1.10e+01,2.00e+02,0.00e+00],
            [1.00e+00,9.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,1.00e+01,1.00e+03,2.00e+00],
            [1.00e+00,6.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,9.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,8.00e+00,5.00e+02,2.00e+00],
            [1.00e+00,4.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,2.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,7.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,6.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,3.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,1.00e+01,1.00e+03,1.00e+00],
            [1.00e+00,6.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,9.00e+00,5.00e+02,2.00e+00],
            [1.00e+00,6.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,6.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,5.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,5.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,5.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,1.00e+01,1.25e+03,1.00e+00],
            [1.00e+00,8.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,0.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,8.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,9.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,5.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,3.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,5.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,2.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,1.10e+01,2.00e+02,0.00e+00],
            [1.00e+00,4.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,1.10e+01,1.00e+03,2.00e+00],
            [1.00e+00,1.10e+01,5.00e+02,1.00e+00],
            [1.00e+00,1.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,1.00e+01,5.00e+02,2.00e+00],
            [1.00e+00,2.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,8.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,0.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,9.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,7.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,4.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,1.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,7.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,8.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,9.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,1.00e+01,1.25e+03,1.00e+00],
            [1.00e+00,0.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,1.10e+01,1.00e+03,2.00e+00],
            [1.00e+00,3.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,7.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,1.00e+01,1.00e+02,0.00e+00],
            [1.00e+00,6.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,8.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,7.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,3.00e+00,5.00e+02,2.00e+00],
            [1.00e+00,5.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,1.00e+01,1.25e+03,1.00e+00],
            [1.00e+00,3.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,3.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,2.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,6.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,1.10e+01,5.00e+01,0.00e+00],
            [1.00e+00,1.20e+01,2.00e+02,0.00e+00],
            [1.00e+00,1.00e+01,5.00e+02,1.00e+00],
            [1.00e+00,1.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,4.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,5.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,5.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,6.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,3.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,6.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,9.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,1.00e+01,5.00e+01,0.00e+00],
            [1.00e+00,2.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,1.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,4.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,7.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,8.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,9.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,4.00e+00,5.00e+02,2.00e+00],
            [1.00e+00,1.10e+01,1.00e+03,2.00e+00],
            [1.00e+00,1.10e+01,1.00e+02,0.00e+00],
            [1.00e+00,5.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,3.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,6.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,0.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,3.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,1.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,5.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,5.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,9.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,1.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,1.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,7.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,3.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,2.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,8.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,0.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,1.00e+01,1.00e+02,0.00e+00],
            [1.00e+00,1.10e+01,2.00e+02,0.00e+00],
            [1.00e+00,7.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,1.00e+01,1.25e+03,1.00e+00],
            [1.00e+00,5.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,1.00e+01,1.50e+03,2.00e+00],
            [1.00e+00,3.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,0.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,4.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,1.10e+01,1.00e+03,1.00e+00],
            [1.00e+00,1.10e+01,1.00e+03,2.00e+00],
            [1.00e+00,4.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,5.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,6.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,3.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,5.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,2.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,1.00e+01,1.00e+02,0.00e+00],
            [1.00e+00,2.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,0.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,5.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,7.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,4.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,1.00e+01,1.50e+03,2.00e+00],
            [1.00e+00,3.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,5.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,5.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,6.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,6.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,3.00e+00,5.00e+02,2.00e+00],
            [1.00e+00,1.10e+01,1.00e+03,1.00e+00],
            [1.00e+00,7.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,1.00e+01,1.00e+03,1.00e+00],
            [1.00e+00,1.00e+00,5.00e+02,2.00e+00],
            [1.00e+00,1.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,8.00e+00,5.00e+02,2.00e+00],
            [1.00e+00,1.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,9.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,7.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,6.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,6.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,9.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,2.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,7.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,1.00e+01,1.00e+02,0.00e+00],
            [1.00e+00,1.00e+01,1.50e+03,2.00e+00],
            [1.00e+00,0.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,6.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,1.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,7.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,1.20e+01,2.00e+02,0.00e+00],
            [1.00e+00,3.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,5.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,7.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,6.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,1.00e+00,5.00e+02,2.00e+00],
            [1.00e+00,0.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,1.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,4.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,1.10e+01,5.00e+02,1.00e+00],
            [1.00e+00,3.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,9.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,1.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,7.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,9.00e+00,5.00e+02,2.00e+00],
            [1.00e+00,2.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,0.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,6.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,0.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,1.10e+01,1.00e+02,0.00e+00],
            [1.00e+00,9.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,4.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,1.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,1.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,6.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,1.00e+01,1.25e+03,1.00e+00],
            [1.00e+00,0.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,1.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,1.00e+01,5.00e+01,0.00e+00],
            [1.00e+00,1.00e+01,5.00e+02,1.00e+00],
            [1.00e+00,8.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,1.10e+01,5.00e+02,2.00e+00],
            [1.00e+00,3.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,6.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,3.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,7.00e+00,5.00e+02,2.00e+00],
            [1.00e+00,1.00e+01,1.00e+03,2.00e+00],
            [1.00e+00,8.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,6.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,7.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,6.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,3.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,3.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,2.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,2.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,4.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,1.10e+01,5.00e+02,2.00e+00],
            [1.00e+00,1.10e+01,1.00e+02,0.00e+00],
            [1.00e+00,4.00e+00,5.00e+02,2.00e+00],
            [1.00e+00,5.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,1.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,3.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,9.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,5.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,2.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,9.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,0.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,7.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,0.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,9.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,1.10e+01,1.00e+03,2.00e+00],
            [1.00e+00,1.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,0.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,8.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,1.10e+01,2.00e+02,0.00e+00],
            [1.00e+00,1.00e+01,2.00e+02,0.00e+00],
            [1.00e+00,2.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,1.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,8.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,3.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,7.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,1.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,7.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,3.00e+00,5.00e+02,2.00e+00],
            [1.00e+00,9.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,2.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,3.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,6.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,8.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,4.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,1.00e+01,1.00e+02,0.00e+00],
            [1.00e+00,9.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,4.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,2.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,1.20e+01,2.00e+02,0.00e+00],
            [1.00e+00,1.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,8.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,1.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,1.00e+01,1.00e+03,2.00e+00],
            [1.00e+00,4.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,4.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,7.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,8.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,1.00e+01,1.25e+03,1.00e+00],
            [1.00e+00,6.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,0.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,4.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,6.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,7.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,3.00e+00,5.00e+02,2.00e+00],
            [1.00e+00,3.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,1.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,9.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,8.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,2.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,1.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,7.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,4.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,2.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,6.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,1.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,1.10e+01,2.00e+02,0.00e+00],
            [1.00e+00,8.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,5.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,8.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,7.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,3.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,9.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,7.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,0.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,0.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,9.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,4.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,1.10e+01,5.00e+02,1.00e+00],
            [1.00e+00,9.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,7.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,5.00e+00,5.00e+02,2.00e+00],
            [1.00e+00,9.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,9.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,1.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,1.00e+01,5.00e+02,1.00e+00],
            [1.00e+00,1.20e+01,2.00e+02,0.00e+00],
            [1.00e+00,6.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,3.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,1.10e+01,1.00e+03,1.00e+00],
            [1.00e+00,8.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,5.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,2.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,8.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,6.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,1.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,1.10e+01,1.50e+03,2.00e+00],
            [1.00e+00,7.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,1.10e+01,1.00e+03,1.00e+00],
            [1.00e+00,1.00e+01,1.00e+03,2.00e+00],
            [1.00e+00,4.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,4.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,4.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,5.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,9.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,7.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,3.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,3.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,6.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,0.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,5.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,0.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,1.00e+01,2.00e+02,0.00e+00],
            [1.00e+00,1.00e+01,5.00e+02,1.00e+00],
            [1.00e+00,7.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,6.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,8.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,0.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,1.10e+01,2.00e+02,0.00e+00],
            [1.00e+00,1.00e+01,5.00e+02,2.00e+00],
            [1.00e+00,2.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,4.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,4.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,1.00e+01,1.50e+03,2.00e+00],
            [1.00e+00,9.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,1.00e+01,1.50e+03,2.00e+00],
            [1.00e+00,7.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,1.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,9.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,6.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,9.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,8.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,7.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,3.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,8.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,1.00e+01,1.00e+02,0.00e+00],
            [1.00e+00,1.10e+01,5.00e+01,0.00e+00],
            [1.00e+00,7.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,7.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,1.00e+01,1.00e+03,1.00e+00],
            [1.00e+00,7.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,8.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,0.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,1.10e+01,1.25e+03,1.00e+00],
            [1.00e+00,0.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,1.00e+01,1.25e+03,1.00e+00],
            [1.00e+00,1.00e+01,2.00e+02,0.00e+00],
            [1.00e+00,1.10e+01,1.25e+03,1.00e+00],
            [1.00e+00,6.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,5.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,6.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,0.00e+00,5.00e+02,2.00e+00],
            [1.00e+00,2.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,5.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,1.10e+01,5.00e+02,1.00e+00],
            [1.00e+00,1.00e+01,1.25e+03,1.00e+00],
            [1.00e+00,3.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,7.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,3.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,1.10e+01,5.00e+01,0.00e+00],
            [1.00e+00,8.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,8.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,1.00e+01,2.00e+02,0.00e+00],
            [1.00e+00,6.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,0.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,0.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,4.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,3.00e+00,5.00e+02,2.00e+00],
            [1.00e+00,7.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,8.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,9.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,7.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,7.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,1.10e+01,1.50e+03,2.00e+00],
            [1.00e+00,5.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,7.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,4.00e+00,5.00e+02,2.00e+00],
            [1.00e+00,9.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,7.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,1.10e+01,1.25e+03,1.00e+00],
            [1.00e+00,9.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,7.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,5.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,6.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,1.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,0.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,5.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,9.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,9.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,8.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,1.10e+01,5.00e+01,0.00e+00],
            [1.00e+00,4.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,1.00e+01,1.50e+03,2.00e+00],
            [1.00e+00,1.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,5.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,7.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,1.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,1.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,5.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,0.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,3.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,9.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,8.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,9.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,8.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,7.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,1.10e+01,1.00e+03,1.00e+00],
            [1.00e+00,1.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,0.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,5.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,5.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,1.00e+01,5.00e+02,2.00e+00],
            [1.00e+00,1.00e+01,1.00e+02,0.00e+00],
            [1.00e+00,1.00e+01,2.00e+02,0.00e+00],
            [1.00e+00,0.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,1.10e+01,1.00e+03,1.00e+00],
            [1.00e+00,5.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,3.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,2.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,9.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,2.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,7.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,5.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,3.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,8.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,6.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,1.20e+01,2.00e+02,0.00e+00],
            [1.00e+00,2.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,8.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,4.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,5.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,8.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,9.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,9.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,7.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,8.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,1.20e+01,2.00e+02,0.00e+00],
            [1.00e+00,1.10e+01,1.50e+03,2.00e+00],
            [1.00e+00,1.10e+01,2.00e+02,0.00e+00],
            [1.00e+00,0.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,1.20e+01,2.00e+02,0.00e+00],
            [1.00e+00,1.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,5.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,0.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,9.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,3.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,3.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,4.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,1.10e+01,1.25e+03,1.00e+00],
            [1.00e+00,0.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,6.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,2.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,8.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,1.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,1.00e+01,1.25e+03,1.00e+00],
            [1.00e+00,1.00e+01,5.00e+02,1.00e+00],
            [1.00e+00,5.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,3.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,7.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,1.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,1.00e+01,5.00e+02,1.00e+00],
            [1.00e+00,9.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,5.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,9.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,1.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,2.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,0.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,1.10e+01,2.00e+02,0.00e+00],
            [1.00e+00,9.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,9.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,5.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,2.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,9.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,1.10e+01,1.50e+03,2.00e+00],
            [1.00e+00,2.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,8.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,1.00e+01,1.50e+03,2.00e+00],
            [1.00e+00,0.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,6.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,6.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,5.00e+00,5.00e+02,2.00e+00],
            [1.00e+00,6.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,2.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,1.00e+01,5.00e+01,0.00e+00],
            [1.00e+00,3.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,1.10e+01,5.00e+02,1.00e+00],
            [1.00e+00,9.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,7.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,5.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,0.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,9.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,0.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,9.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,1.00e+01,1.25e+03,1.00e+00],
            [1.00e+00,9.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,6.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,0.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,4.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,0.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,3.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,5.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,5.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,8.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,6.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,1.00e+01,2.00e+02,0.00e+00],
            [1.00e+00,6.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,5.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,1.10e+01,5.00e+02,1.00e+00],
            [1.00e+00,9.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,7.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,2.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,2.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,2.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,1.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,1.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,0.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,5.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,0.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,3.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,9.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,7.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,9.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,1.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,5.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,0.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,3.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,1.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,7.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,0.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,4.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,9.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,8.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,1.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,1.00e+01,5.00e+02,1.00e+00],
            [1.00e+00,1.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,1.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,5.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,0.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,0.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,1.10e+01,1.25e+03,1.00e+00],
            [1.00e+00,1.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,6.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,1.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,3.00e+00,5.00e+02,2.00e+00],
            [1.00e+00,7.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,1.10e+01,5.00e+01,0.00e+00],
            [1.00e+00,0.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,1.00e+01,2.00e+02,0.00e+00],
            [1.00e+00,1.00e+01,1.00e+03,2.00e+00],
            [1.00e+00,6.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,1.00e+01,1.25e+03,1.00e+00],
            [1.00e+00,2.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,1.00e+01,1.50e+03,2.00e+00],
            [1.00e+00,6.00e+00,5.00e+02,2.00e+00],
            [1.00e+00,3.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,8.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,4.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,3.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,1.10e+01,1.50e+03,2.00e+00],
            [1.00e+00,8.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,2.00e+00,5.00e+02,2.00e+00],
            [1.00e+00,3.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,1.10e+01,5.00e+02,2.00e+00],
            [1.00e+00,3.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,6.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,9.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,1.10e+01,5.00e+01,0.00e+00],
            [1.00e+00,0.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,5.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,1.00e+01,5.00e+01,0.00e+00],
            [1.00e+00,5.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,8.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,8.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,1.00e+01,2.00e+02,0.00e+00],
            [1.00e+00,1.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,4.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,8.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,1.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,4.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,3.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,6.00e+00,5.00e+02,2.00e+00],
            [1.00e+00,0.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,2.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,3.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,7.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,3.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,6.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,3.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,7.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,2.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,6.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,1.00e+00,5.00e+02,2.00e+00],
            [1.00e+00,1.00e+01,1.50e+03,2.00e+00],
            [1.00e+00,5.00e+00,5.00e+02,2.00e+00],
            [1.00e+00,8.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,6.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,9.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,5.00e+00,5.00e+02,2.00e+00],
            [1.00e+00,8.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,2.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,1.00e+01,5.00e+02,2.00e+00],
            [1.00e+00,1.00e+01,5.00e+01,0.00e+00],
            [1.00e+00,1.00e+01,1.00e+03,2.00e+00],
            [1.00e+00,4.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,0.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,6.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,1.00e+01,5.00e+02,1.00e+00],
            [1.00e+00,3.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,1.00e+01,1.00e+02,0.00e+00],
            [1.00e+00,5.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,1.00e+01,1.00e+03,1.00e+00],
            [1.00e+00,7.00e+00,5.00e+02,2.00e+00],
            [1.00e+00,1.10e+01,5.00e+01,0.00e+00],
            [1.00e+00,1.10e+01,2.00e+02,0.00e+00],
            [1.00e+00,4.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,9.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,0.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,6.00e+00,5.00e+02,2.00e+00],
            [1.00e+00,4.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,2.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,1.00e+01,1.50e+03,2.00e+00],
            [1.00e+00,1.00e+01,5.00e+01,0.00e+00],
            [1.00e+00,6.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,6.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,1.10e+01,5.00e+01,0.00e+00],
            [1.00e+00,4.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,4.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,5.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,1.00e+01,2.00e+02,0.00e+00],
            [1.00e+00,4.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,6.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,3.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,8.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,6.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,1.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,3.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,8.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,6.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,6.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,1.10e+01,2.00e+02,0.00e+00],
            [1.00e+00,1.00e+01,1.00e+02,0.00e+00],
            [1.00e+00,7.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,7.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,6.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,0.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,1.00e+01,1.00e+03,1.00e+00],
            [1.00e+00,6.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,6.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,8.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,1.10e+01,1.00e+03,1.00e+00],
            [1.00e+00,6.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,8.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,1.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,5.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,4.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,3.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,5.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,1.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,1.10e+01,5.00e+01,0.00e+00],
            [1.00e+00,9.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,1.00e+00,5.00e+02,2.00e+00],
            [1.00e+00,4.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,3.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,7.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,3.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,8.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,1.10e+01,2.00e+02,0.00e+00],
            [1.00e+00,4.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,3.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,1.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,6.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,7.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,4.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,4.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,6.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,1.10e+01,1.25e+03,1.00e+00],
            [1.00e+00,9.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,1.00e+01,5.00e+01,0.00e+00],
            [1.00e+00,0.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,1.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,6.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,2.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,4.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,7.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,0.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,2.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,6.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,9.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,2.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,1.10e+01,1.25e+03,1.00e+00],
            [1.00e+00,2.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,6.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,8.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,5.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,1.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,2.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,1.00e+01,5.00e+02,1.00e+00],
            [1.00e+00,0.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,1.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,5.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,2.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,9.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,5.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,4.00e+00,5.00e+02,2.00e+00],
            [1.00e+00,9.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,1.00e+01,5.00e+02,1.00e+00],
            [1.00e+00,2.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,5.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,2.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,1.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,8.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,6.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,6.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,1.10e+01,1.00e+03,1.00e+00],
            [1.00e+00,1.00e+01,2.00e+02,0.00e+00],
            [1.00e+00,7.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,4.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,1.10e+01,2.00e+02,0.00e+00],
            [1.00e+00,6.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,2.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,1.00e+01,5.00e+01,0.00e+00],
            [1.00e+00,7.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,8.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,6.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,2.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,1.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,6.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,5.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,4.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,8.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,6.00e+00,5.00e+02,2.00e+00],
            [1.00e+00,8.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,0.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,8.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,7.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,8.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,1.10e+01,1.00e+03,2.00e+00],
            [1.00e+00,1.00e+01,1.50e+03,2.00e+00],
            [1.00e+00,3.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,9.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,4.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,2.00e+00,5.00e+02,2.00e+00],
            [1.00e+00,1.00e+01,5.00e+02,1.00e+00],
            [1.00e+00,1.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,3.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,0.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,7.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,4.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,5.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,1.10e+01,2.00e+02,0.00e+00],
            [1.00e+00,7.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,5.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,4.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,1.10e+01,5.00e+01,0.00e+00],
            [1.00e+00,2.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,1.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,2.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,1.10e+01,1.25e+03,1.00e+00],
            [1.00e+00,0.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,1.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,4.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,1.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,4.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,6.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,4.00e+00,5.00e+02,2.00e+00],
            [1.00e+00,1.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,7.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,5.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,1.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,9.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,3.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,1.00e+01,1.50e+03,2.00e+00],
            [1.00e+00,7.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,6.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,5.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,3.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,3.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,2.00e+00,5.00e+02,2.00e+00],
            [1.00e+00,2.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,6.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,5.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,3.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,8.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,1.00e+01,5.00e+02,1.00e+00],
            [1.00e+00,1.00e+01,1.00e+03,1.00e+00],
            [1.00e+00,7.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,1.00e+01,1.50e+03,2.00e+00],
            [1.00e+00,4.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,0.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,3.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,8.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,1.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,0.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,1.00e+01,5.00e+01,0.00e+00],
            [1.00e+00,4.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,2.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,0.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,3.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,8.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,8.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,4.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,5.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,5.00e+00,5.00e+02,2.00e+00],
            [1.00e+00,1.10e+01,1.00e+02,0.00e+00],
            [1.00e+00,3.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,7.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,9.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,4.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,2.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,8.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,9.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,8.00e+00,5.00e+02,2.00e+00],
            [1.00e+00,1.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,4.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,1.10e+01,1.00e+03,1.00e+00],
            [1.00e+00,8.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,5.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,6.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,1.10e+01,1.00e+02,0.00e+00],
            [1.00e+00,9.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,7.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,0.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,1.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,5.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,9.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,5.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,0.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,2.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,6.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,8.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,9.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,3.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,4.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,3.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,5.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,5.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,1.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,1.00e+00,5.00e+02,2.00e+00],
            [1.00e+00,8.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,1.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,1.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,7.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,8.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,4.00e+00,5.00e+02,2.00e+00],
            [1.00e+00,1.00e+01,5.00e+02,1.00e+00],
            [1.00e+00,3.00e+00,5.00e+02,2.00e+00],
            [1.00e+00,7.00e+00,5.00e+02,2.00e+00],
            [1.00e+00,0.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,4.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,2.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,4.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,2.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,9.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,6.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,2.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,5.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,1.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,1.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,1.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,2.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,2.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,3.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,9.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,8.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,3.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,2.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,4.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,1.00e+01,1.00e+03,1.00e+00],
            [1.00e+00,0.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,8.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,3.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,0.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,7.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,5.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,1.00e+01,2.00e+02,0.00e+00],
            [1.00e+00,5.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,1.00e+01,1.25e+03,1.00e+00],
            [1.00e+00,1.00e+01,1.25e+03,1.00e+00],
            [1.00e+00,7.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,0.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,0.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,3.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,7.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,1.10e+01,1.25e+03,1.00e+00],
            [1.00e+00,3.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,0.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,1.10e+01,1.00e+02,0.00e+00],
            [1.00e+00,2.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,8.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,6.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,1.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,2.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,3.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,0.00e+00,5.00e+02,2.00e+00],
            [1.00e+00,7.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,7.00e+00,5.00e+02,2.00e+00],
            [1.00e+00,7.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,6.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,4.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,1.00e+01,2.00e+02,0.00e+00],
            [1.00e+00,2.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,7.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,3.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,8.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,6.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,4.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,1.10e+01,1.50e+03,2.00e+00],
            [1.00e+00,3.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,8.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,1.10e+01,5.00e+01,0.00e+00],
            [1.00e+00,7.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,0.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,2.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,7.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,7.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,9.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,1.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,1.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,1.00e+01,1.25e+03,1.00e+00],
            [1.00e+00,6.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,1.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,5.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,1.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,1.00e+01,2.00e+02,0.00e+00],
            [1.00e+00,9.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,1.10e+01,1.50e+03,2.00e+00],
            [1.00e+00,4.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,6.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,6.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,5.00e+00,5.00e+02,2.00e+00],
            [1.00e+00,1.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,2.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,9.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,9.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,7.00e+00,5.00e+02,2.00e+00],
            [1.00e+00,1.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,2.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,8.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,8.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,5.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,6.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,7.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,2.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,9.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,7.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,5.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,1.10e+01,1.50e+03,2.00e+00],
            [1.00e+00,4.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,1.00e+01,1.00e+02,0.00e+00],
            [1.00e+00,1.00e+00,5.00e+02,2.00e+00],
            [1.00e+00,6.00e+00,5.00e+02,2.00e+00],
            [1.00e+00,2.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,9.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,1.10e+01,1.50e+03,2.00e+00],
            [1.00e+00,4.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,9.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,3.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,5.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,3.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,7.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,4.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,1.00e+01,1.00e+03,2.00e+00],
            [1.00e+00,5.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,1.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,1.10e+01,5.00e+02,1.00e+00],
            [1.00e+00,8.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,3.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,1.10e+01,5.00e+01,0.00e+00],
            [1.00e+00,1.00e+01,5.00e+02,1.00e+00],
            [1.00e+00,7.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,3.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,1.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,8.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,2.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,1.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,4.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,2.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,3.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,1.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,5.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,1.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,8.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,0.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,5.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,5.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,7.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,7.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,6.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,8.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,2.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,0.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,0.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,1.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,8.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,8.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,3.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,0.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,6.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,1.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,7.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,7.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,8.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,5.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,0.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,7.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,4.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,1.00e+01,5.00e+02,1.00e+00],
            [1.00e+00,1.10e+01,5.00e+01,0.00e+00],
            [1.00e+00,3.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,1.00e+01,1.00e+03,1.00e+00],
            [1.00e+00,1.00e+01,1.00e+03,1.00e+00],
            [1.00e+00,4.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,4.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,9.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,1.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,3.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,9.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,3.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,1.10e+01,2.00e+02,0.00e+00],
            [1.00e+00,1.00e+01,5.00e+02,1.00e+00],
            [1.00e+00,1.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,1.10e+01,5.00e+02,1.00e+00],
            [1.00e+00,0.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,1.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,3.00e+00,5.00e+02,2.00e+00],
            [1.00e+00,1.00e+01,1.00e+02,0.00e+00],
            [1.00e+00,2.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,1.10e+01,5.00e+02,1.00e+00],
            [1.00e+00,3.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,7.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,9.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,0.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,2.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,2.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,5.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,1.10e+01,1.50e+03,2.00e+00],
            [1.00e+00,1.10e+01,1.50e+03,2.00e+00],
            [1.00e+00,6.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,1.00e+01,2.00e+02,0.00e+00],
            [1.00e+00,9.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,5.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,9.00e+00,5.00e+02,2.00e+00],
            [1.00e+00,0.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,1.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,1.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,0.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,3.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,4.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,1.00e+01,1.25e+03,1.00e+00],
            [1.00e+00,4.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,1.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,1.00e+01,5.00e+02,1.00e+00],
            [1.00e+00,2.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,1.00e+01,1.00e+02,0.00e+00],
            [1.00e+00,3.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,7.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,0.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,0.00e+00,5.00e+02,2.00e+00],
            [1.00e+00,9.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,6.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,6.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,7.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,1.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,1.10e+01,5.00e+01,0.00e+00],
            [1.00e+00,8.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,9.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,5.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,6.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,1.00e+01,1.00e+02,0.00e+00],
            [1.00e+00,9.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,1.10e+01,1.25e+03,1.00e+00],
            [1.00e+00,2.00e+00,5.00e+02,2.00e+00],
            [1.00e+00,8.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,5.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,2.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,1.10e+01,1.00e+03,1.00e+00],
            [1.00e+00,1.00e+01,1.00e+03,2.00e+00],
            [1.00e+00,9.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,8.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,4.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,1.00e+01,1.50e+03,2.00e+00],
            [1.00e+00,1.10e+01,5.00e+02,2.00e+00],
            [1.00e+00,1.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,3.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,1.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,3.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,8.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,2.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,8.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,2.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,6.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,4.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,6.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,4.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,1.10e+01,1.00e+03,1.00e+00],
            [1.00e+00,9.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,1.00e+01,1.00e+03,2.00e+00],
            [1.00e+00,3.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,6.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,4.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,2.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,2.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,9.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,5.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,7.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,4.00e+00,5.00e+02,2.00e+00],
            [1.00e+00,2.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,1.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,6.00e+00,5.00e+02,2.00e+00],
            [1.00e+00,5.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,7.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,8.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,4.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,5.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,1.10e+01,1.25e+03,1.00e+00],
            [1.00e+00,1.00e+00,5.00e+02,2.00e+00],
            [1.00e+00,0.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,2.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,2.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,2.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,1.10e+01,5.00e+02,1.00e+00],
            [1.00e+00,7.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,9.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,1.00e+01,2.00e+02,0.00e+00],
            [1.00e+00,6.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,1.00e+01,1.50e+03,2.00e+00],
            [1.00e+00,9.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,1.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,7.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,1.00e+01,2.00e+02,0.00e+00],
            [1.00e+00,5.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,8.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,4.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,2.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,8.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,0.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,1.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,2.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,1.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,3.00e+00,5.00e+02,2.00e+00],
            [1.00e+00,3.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,4.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,2.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,4.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,5.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,3.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,3.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,5.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,7.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,6.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,8.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,3.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,0.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,2.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,1.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,9.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,0.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,7.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,1.00e+01,5.00e+01,0.00e+00],
            [1.00e+00,1.20e+01,2.00e+02,0.00e+00],
            [1.00e+00,9.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,6.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,1.00e+01,5.00e+02,1.00e+00],
            [1.00e+00,1.10e+01,1.00e+02,0.00e+00],
            [1.00e+00,8.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,8.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,2.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,1.00e+01,5.00e+02,1.00e+00],
            [1.00e+00,5.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,9.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,3.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,7.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,0.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,7.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,3.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,6.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,0.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,9.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,2.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,1.00e+01,1.00e+03,1.00e+00],
            [1.00e+00,1.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,3.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,3.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,5.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,1.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,3.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,8.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,7.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,2.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,6.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,4.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,3.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,9.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,1.00e+01,5.00e+01,0.00e+00],
            [1.00e+00,8.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,6.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,6.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,1.00e+01,2.00e+02,0.00e+00],
            [1.00e+00,7.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,4.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,0.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,1.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,2.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,5.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,4.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,4.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,9.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,1.00e+01,1.25e+03,1.00e+00],
            [1.00e+00,1.00e+01,5.00e+01,0.00e+00],
            [1.00e+00,1.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,1.00e+01,5.00e+02,1.00e+00],
            [1.00e+00,3.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,2.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,3.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,5.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,5.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,0.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,5.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,1.10e+01,1.25e+03,1.00e+00],
            [1.00e+00,8.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,0.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,0.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,1.10e+01,1.00e+02,0.00e+00],
            [1.00e+00,1.10e+01,5.00e+02,1.00e+00],
            [1.00e+00,1.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,1.10e+01,1.50e+03,2.00e+00],
            [1.00e+00,2.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,4.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,3.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,1.10e+01,1.00e+03,2.00e+00],
            [1.00e+00,7.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,1.10e+01,2.00e+02,0.00e+00],
            [1.00e+00,6.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,8.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,9.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,2.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,3.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,1.10e+01,1.00e+03,1.00e+00],
            [1.00e+00,1.00e+00,5.00e+02,2.00e+00],
            [1.00e+00,3.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,9.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,5.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,8.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,6.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,3.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,1.10e+01,1.00e+03,1.00e+00],
            [1.00e+00,4.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,8.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,0.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,4.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,2.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,1.10e+01,1.00e+03,1.00e+00],
            [1.00e+00,0.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,2.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,3.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,6.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,1.00e+01,2.00e+02,0.00e+00],
            [1.00e+00,8.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,6.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,1.10e+01,5.00e+02,1.00e+00],
            [1.00e+00,4.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,9.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,1.10e+01,2.00e+02,0.00e+00],
            [1.00e+00,0.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,8.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,2.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,8.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,5.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,9.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,8.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,5.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,3.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,2.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,1.10e+01,1.00e+02,0.00e+00],
            [1.00e+00,4.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,5.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,7.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,1.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,5.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,9.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,1.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,1.10e+01,1.00e+03,1.00e+00],
            [1.00e+00,1.00e+01,2.00e+02,0.00e+00],
            [1.00e+00,5.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,0.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,7.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,1.10e+01,5.00e+02,1.00e+00],
            [1.00e+00,9.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,5.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,4.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,3.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,0.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,1.00e+01,1.00e+03,1.00e+00],
            [1.00e+00,4.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,0.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,2.00e+00,5.00e+02,2.00e+00],
            [1.00e+00,1.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,8.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,3.00e+00,5.00e+02,2.00e+00],
            [1.00e+00,7.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,5.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,7.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,1.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,6.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,8.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,7.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,8.00e+00,1.00e+03,2.00e+00],
            [1.00e+00,5.00e+00,5.00e+02,2.00e+00],
            [1.00e+00,1.10e+01,1.25e+03,1.00e+00],
            [1.00e+00,1.10e+01,1.00e+03,1.00e+00],
            [1.00e+00,1.10e+01,1.50e+03,2.00e+00],
            [1.00e+00,3.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,3.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,7.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,9.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,6.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,6.00e+00,5.00e+02,1.00e+00],
            [1.00e+00,4.00e+00,1.25e+03,1.00e+00],
            [1.00e+00,9.00e+00,5.00e+01,0.00e+00],
            [1.00e+00,9.00e+00,1.00e+03,1.00e+00],
            [1.00e+00,4.00e+00,1.50e+03,2.00e+00],
            [1.00e+00,4.00e+00,2.00e+02,0.00e+00],
            [1.00e+00,6.00e+00,1.00e+02,0.00e+00],
            [1.00e+00,4.00e+00,2.00e+02,0.00e+00]]

    y_train=[7.3,16.2,8.,3.2,3.2,5.1,4.7,4.3,18.8,4.,4.5,3.1
        ,5.,16.,1.8,7.2,1.3,8.3,3.6,10.,9.2,14.2,13.,6.
        ,3.6,2.,11.8,18.,18.,4.2,8.8,3.2,8.6,4.,2.6,1.
        ,9.8,16.9,13.3,16.7,2.7,14.,14.8,5.2,17.,15.6,12.2,5.8
        ,5.2,8.5,3.2,6.3,18.,9.9,8.4,5.1,13.9,22.9,4.,19.3
        ,6.6,16.,15.,1.5,11.,5.3,14.8,11.5,8.4,7.1,4.3,3.6
        ,2.2,3.6,12.6,9.9,4.4,2.9,14.6,7.5,5.1,4.4,19.8,15.9
        ,3.7,10.8,12.1,2.8,2.2,6.3,8.,10.,4.2,16.,14.8,1.7
        ,4.,16.,5.2,22.9,3.9,6.5,13.4,5.7,4.,4.,11.8,5.4
        ,3.6,17.2,17.4,6.7,5.7,17.7,0.6,16.,7.,5.8,17.,2.8
        ,16.1,8.5,5.2,14.6,6.7,0.7,2.8,13.8,7.2,23.,4.5,12.6
        ,8.2,4.8,17.,14.7,19.9,16.,3.5,4.2,4.4,2.,15.9,17.1
        ,3.4,0.6,4.6,4.4,4.3,3.2,15.2,12.,4.3,15.8,17.,8.6
        ,0.8,2.3,1.,16.,5.1,14.8,8.4,12.4,6.2,16.,1.4,2.6
        ,5.2,4.,1.4,4.4,1.5,6.2,4.4,3.5,6.2,12.9,11.2,12.4
        ,9.8,3.7,10.1,4.,5.,9.8,3.,3.5,16.9,4.1,3.6,18.
        ,16.6,16.2,4.,3.9,11.6,7.2,13.8,6.,7.,17.8,12.,4.3
        ,2.5,2.1,3.8,16.8,4.9,2.5,20.,3.8,8.4,19.,3.9,3.2
        ,8.2,11.7,16.8,11.8,18.,12.6,2.5,6.1,15.4,16.6,7.2,16.
        ,5.,8.5,13.,4.1,4.,8.6,9.6,15.2,12.,4.4,15.2,23.
        ,12.4,5.9,3.,5.,5.2,14.,8.3,15.2,3.8,2.,4.9,4.1
        ,11.7,4.6,1.7,17.2,13.6,19.7,16.9,3.2,17.2,14.5,5.3,5.5
        ,14.7,4.2,6.3,13.9,10.2,2.7,16.9,5.6,6.2,15.4,13.8,8.2
        ,1.8,4.8,8.4,16.,4.3,8.1,12.6,16.,3.8,1.9,17.9,5.8
        ,13.2,17.8,2.1,5.8,10.1,19.,8.,15.4,19.,17.,16.7,4.9
        ,6.6,20.,5.1,4.,13.5,3.2,3.3,8.4,3.,12.6,15.8,13.
        ,7.4,17.9,10.,3.8,9.6,5.6,6.3,16.3,5.,8.,8.1,3.7
        ,10.,16.,3.,11.6,2.6,4.4,3.2,11.2,17.,7.,4.4,16.
        ,0.8,4.5,4.1,12.3,10.2,4.6,15.9,16.8,18.8,8.4,8.6,13.6
        ,10.,17.2,7.2,4.5,11.9,13.3,2.8,4.2,2.1,3.,4.6,4.1
        ,3.5,4.2,23.,13.9,12.,23.,8.5,11.8,8.4,12.6,16.,1.6
        ,20.1,2.8,8.6,5.5,2.8,4.,8.9,3.2,9.7,6.8,1.7,20.
        ,4.,0.8,7.1,12.2,18.,4.6,13.5,9.8,2.2,11.2,8.8,3.9
        ,16.,16.8,8.,12.1,4.8,0.9,3.2,2.9,6.4,3.8,15.,16.
        ,6.8,9.,16.,4.3,9.7,10.,7.1,1.9,4.,4.3,6.,11.
        ,19.4,5.6,18.6,8.3,5.4,1.9,4.2,13.9,8.6,8.3,5.4,4.
        ,4.2,7.,12.6,7.1,9.8,2.2,13.8,18.,4.3,8.,3.8,7.4
        ,5.4,3.3,10.2,6.4,17.4,7.,19.5,11.1,2.9,11.4,7.4,14.
        ,18.1,2.8,4.4,1.2,23.,6.,15.9,12.8,4.,1.8,3.9,6.1
        ,10.2,13.2,11.7,5.8,3.9,4.8,3.8,11.6,3.2,3.4,12.,16.7
        ,3.4,3.3,16.2,8.,12.1,1.9,7.8,1.9,7.,3.9,7.8,8.4
        ,4.5,17.,9.,3.8,12.4,14.6,5.4,5.8,15.5,15.9,3.7,8.4
        ,7.2,4.2,9.1,4.8,15.,7.6,9.2,3.2,3.2,3.,7.,14.
        ,20.1,9.7,1.6,8.4,8.4,1.7,5.7,13.9,4.2,6.3,13.9,16.8
        ,2.5,2.4,2.9,11.4,6.8,17.8,3.2,4.1,5.4,2.2,16.,14.
        ,12.8,3.1,13.8,0.9,17.6,4.4,3.8,0.7,7.9,18.,16.6,10.2
        ,4.2,5.1,8.6,7.5,11.2,10.1,16.7,5.,19.9,16.,7.,4.8
        ,5.,4.3,2.5,5.7,5.2,12.4,5.8,6.1,5.8,6.,5.7,6.8
        ,9.6,11.4,4.2,9.8,19.,12.7,3.7,15.5,18.,17.,10.,15.4
        ,10.4,12.1,18.3,13.,10.7,2.6,4.2,2.7,7.,4.8,4.,16.8
        ,5.8,3.7,15.8,6.2,7.9,4.7,12.2,11.1,5.6,7.2,3.2,4.8
        ,19.,16.7,3.8,4.6,6.7,9.8,4.5,5.4,16.,16.9,7.8,2.9
        ,10.3,1.3,4.2,13.4,9.2,3.4,2.2,2.,2.4,16.8,4.,14.9
        ,19.9,2.7,14.2,10.,8.5,18.3,4.4,8.8,9.9,4.1,16.9,8.4
        ,4.7,16.8,4.4,3.7,3.8,4.6,13.9,3.7,7.2,17.,16.8,4.3
        ,5.,16.5,20.,4.6,4.3,10.1,3.1,4.4,5.1,1.4,4.4,6.9
        ,4.2,19.9,17.3,3.,11.9,12.4,6.8,1.7,4.8,3.9,15.,3.2
        ,17.9,17.2,3.6,5.5,8.6,1.9,17.8,8.,4.8,11.8,9.8,4.
        ,9.7,4.4,9.9,16.2,2.2,13.6,5.4,4.2,4.4,5.1,13.2,4.2
        ,4.,2.2,16.9,16.,18.8,6.,7.9,15.9,6.8,8.4,1.9,1.1
        ,12.3,6.2,15.,7.4,7.6,11.3,15.9,5.7,14.6,7.7,6.,22.9
        ,17.,7.2,4.2,15.8,2.8,10.2,15.2,9.,2.7,10.6,1.7,12.3
        ,9.,6.8,5.3,17.1,1.4,18.1,11.2,8.2,14.2,8.9,8.6,12.6
        ,15.6,3.8,3.9,6.1,11.6,8.8,1.2,14.2,12.6,1.1,6.4,9.
        ,14.2,3.6,5.6,5.1,15.8,10.,14.8,4.,7.3,6.2,16.,4.8
        ,7.4,15.9,2.4,5.1,10.2,15.,5.6,4.2,16.,6.,16.5,1.9
        ,12.1,1.6,3.1,9.6,7.8,5.8,11.2,3.9,2.9,18.2,9.,16.8
        ,11.2,4.4,23.,20.1,2.,4.6,11.6,7.4,16.,3.8,4.1,10.6
        ,18.,11.4,23.,8.4,14.4,18.2,8.4,3.,8.3,4.5,8.,5.4
        ,13.9,11.6,7.8,3.9,12.5,19.6,5.8,12.6,11.,4.1,19.4,11.5
        ,4.2,20.1,8.,16.,5.,4.3,19.6,4.2,11.6,6.,9.9,4.5
        ,7.2,10.,5.7,12.6,17.2,2.3,4.1,17.,3.,5.3,16.,2.4
        ,4.1,17.1,7.9,11.1,17.1,4.4,5.9,4.3,11.2,1.4,14.,3.9
        ,2.1,11.1,7.4,15.4,14.8,17.2,16.7,1.7,4.2,7.2,4.,4.2
        ,3.,17.8,12.3,1.,2.7,11.4,16.6,3.8,3.8,2.7,14.8,7.1
        ,17.,3.9,16.,8.,6.3,12.7,4.5,9.,15.6,3.8,3.1,15.8
        ,12.8,4.8,5.7,3.7,2.1,13.,3.5,16.7,12.6,14.6,5.9,4.3
        ,2.8,14.,2.4,4.6,7.,16.8,19.4,5.5,18.1,13.,2.4,7.4
        ,11.6,4.9,10.,5.,8.8,4.,4.6,4.4,22.2,12.4,3.3,0.8
        ,3.6,6.2,8.3,16.,12.8,12.1,2.5,8.4,8.6,13.8,2.1,15.4
        ,5.9,16.8,4.,11.6,15.4,8.6,8.9,4.,3.8,2.5,12.4,0.7
        ,16.8,1.3,3.,8.,2.,12.4,11.8,4.,15.9,5.7,8.8,17.9
        ,1.7,16.8,15.8,11.5,5.1,4.2,5.4,7.,6.,17.,11.2,8.2
        ,12.6,11.8,9.7,8.6,19.9,9.2,18.1,20.,4.1,6.1,5.2,18.1
        ,7.2,16.,16.,7.8,4.,14.,4.1,8.4,7.7,4.2,5.8,14.
        ,11.4,3.9,18.,5.,18.4,16.6,4.,3.8,4.7,11.2,7.7,14.8
        ,3.3,2.3,11.6,12.,5.2,1.7,18.,4.4,4.2,17.,15.,17.
        ,4.2,4.9,6.1,5.3,6.2,13.,16.8,8.,3.7,15.3,4.2,6.7
        ,16.9,4.5,13.6,7.2,3.1,6.1,10.8,8.1,9.6,19.4,12.4,3.7
        ,6.5,2.3,10.2,5.4,4.2,6.1,13.8,16.9,8.5,11.2,15.2,10.8
        ,3.7,8.5,7.4,6.,7.7,7.9,4.2,14.1,9.,7.8,12.,5.8
        ,2.9,19.1,12.,8.7,15.6,8.2,5.2,20.,11.6,4.6,23.,12.5
        ,7.2,17.,15.3,4.5,8.,5.8,15.4,8.1,16.9,2.7,17.2,10.
        ,4.6,10.,12.2,22.9,13.8,17.2,2.9,10.,6.1,14.4,8.4,3.8
        ,6.2,7.4,5.,5.2,16.8,4.,5.3,5.2,3.4,6.8,9.2,4.4
        ,2.,1.8,4.4,3.9,18.,15.7,6.2,17.8,15.4,11.2,2.1,17.9
        ,18.9,7.3,13.8,2.9,1.6,1.9,14.8,2.3,5.1,5.1,12.6,4.3
        ,5.8,10.8,17.7,5.3,8.2,6.8,6.2,13.9,5.3,13.,10.8,16.
        ,21.2,16.8,2.2,7.3,4.2,15.,16.,8.8,22.,1.5,10.4,13.5
        ,4.,14.7,3.9,15.6,12.9,3.9,13.9,4.6,15.6,13.2,10.9,3.2
        ,3.,3.8,11.6,23.2,4.2,3.,4.4,5.4,7.2,14.,5.2,6.
        ,2.6,17.,9.2,7.9,17.8,11.8,8.9,13.4,3.3,3.2,6.5,15.6
        ,16.6,14.5,14.8,19.,13.8,14.8,9.2,10.6,3.4,14.2,4.9,8.1
        ,15.6,7.8,5.1,16.,17.,7.4,11.2,5.7,4.6,11.2,15.9,10.3
        ,12.4,1.8,11.6,4.8,12.4,6.8,8.2,1.3,6.,2.8,8.8,5.2
        ,17.2,2.7,4.4,8.5,16.3,8.6,8.5,6.6,3.,10.,12.8,16.
        ,11.8,13.4,5.9,3.4,3.2,10.,4.4,13.4,10.2,6.2,3.,10.
        ,4.9,7.4,5.2,17.,3.9,3.9,15.9,7.1,11.5,3.2,4.4,17.8
        ,3.8,17.2,1.5,3.8,23.,12.4,5.4,12.,6.,15.4,13.,14.8
        ,12.4,5.3,12.4,16.9,4.3,12.6,15.4,4.,5.2,11.6,12.2,7.2
        ,9.9,17.1,15.6,4.5,4.2,12.4,2.6,16.,4.2,8.8,11.6,5.
        ,13.6,19.7,2.,4.9,15.1,3.5,3.2,5.2,18.5,16.3,3.9,16.2
        ,10.8,7.1,2.7,4.5,3.6,17.1,6.2,2.,5.5,11.4,1.6,15.6
        ,6.2,8.3,7.5,0.5,3.6,3.7,5.1,8.3,3.1,18.9,11.2,3.9
        ,14.,8.8,9.8,6.,4.2,8.8,1.5,17.,18.9,17.4,13.9,16.
        ,10.,4.4,15.,7.4,2.6,15.8,3.2,13.,5.1,14.1,6.7,15.7
        ,4.6,17.1,13.8,13.1,3.1,7.6,4.,2.1,2.8,7.,13.6,17.7
        ,4.8,4.4,4.4,5.7,13.6,15.6,6.,11.2,5.7,8.8,14.,8.5
        ,7.5,4.8,10.,16.9,17.8,19.,13.6,5.9,11.6,11.6,13.,11.3
        ,2.6,12.9,15.4,19.,14.1,15.3,12.4,10.,5.9,4.,7.2,4.1
        ,7.,6.2,5.2,15.6,4.8,12.2,9.,2.1,9.6,18.3,14.7,1.1
        ,5.8,1.9,11.2,15.4,10.2,4.9,6.9,12.3,4.,8.2,10.1,7.4
        ,9.,6.1,17.8,7.8,15.8,5.1,3.,1.9,3.9,1.3,8.8,5.6
        ,4.1,5.4,11.6,12.2,11.,18.2,2.8,17.2,3.9,2.6,11.6,8.4
        ,16.1,3.8,6.1,5.8,4.9,19.,8.6,23.,4.7,7.6,5.,2.6
        ,5.2,14.7,4.4,1.1,2.8,4.3,10.,4.5,4.2,11.5,9.1,3.
        ,6.1,5.3,2.3,8.3,0.9,16.,5.8,4.,4.1,4.7,5.8,5.
        ,16.7,17.8,3.9,12.3,4.6,6.6,2.9,16.9,17.,19.6,4.,17.8
        ,12.6,18.,3.7,7.,4.,1.7,3.4,4.3,11.5,6.6,16.2,10.
        ,9.6,16.,13.,12.4,16.2,17.9,16.2,5.8,3.4,10.3,1.9,12.4
        ,6.1,19.,19.9,2.4,2.7,14.5,4.,1.9,2.7,7.2,15.,16.2
        ,4.,11.1,8.8,7.7,8.6,5.9,11.2,17.,16.6,15.2,12.7,3.8
        ,17.7,3.9,2.7,3.8,8.6,6.,4.,4.6,2.6,4.,1.6,12.
        ,15.6,13.6,6.2,15.4,10.,4.7,11.2,4.1,16.6,4.1,11.,14.
        ,7.2,4.8,8.5,9.4,3.5,7.8,1.7,13.,3.4,12.4,12.7,4.3
        ,14.,13.2,8.4,7.,1.8,19.8,7.4,12.6,5.8,16.7,8.6,17.
        ,2.8,16.8,3.7,7.4,5.6,6.,5.7,2.9,11.2,15.4,4.6,4.
        ,4.1,18.,5.,15.2,10.,4.2,3.6,8.8,14.4,8.4,16.1,8.4
        ,14.,17.,18.2,19.6,13.8,4.8,15.6,3.6,1.,16.,5.7,5.2
        ,19.,4.3,10.,8.8,4.2,16.2,5.3,19.4,9.9,1.5,2.7,10.
        ,1.2,2.3,5.4,5.1,8.8,5.,6.7,3.7,5.,1.3,7.6,4.4
        ,14.,7.8,3.4,6.8,4.5,10.,9.1,2.,6.1,4.3,16.2,2.8
        ,4.3,6.,14.4,16.9,8.8,6.7,9.8,5.9,9.1,5.3,3.5,7.1
        ,7.2,16.6,16.2,10.8,11.1,4.4,7.1,4.4,2.6,9.4,19.6,2.6
        ,11.4,2.2,4.4,8.5,15.9,7.6,4.1,15.4,18.1,5.,11.9,16.4
        ,16.,4.4,5.2,6.2,6.3,8.,5.8,5.4,14.6,16.8,6.6,16.8
        ,7.8,15.6,22.9,3.1,4.6,15.5,11.7,9.,15.4,13.8,8.4,14.3
        ,17.2,15.8,7.2,18.9,7.5,13.4,4.4,12.6,7.2,7.4,16.,7.2
        ,12.3,5.4,6.4,8.4,17.,4.,4.,16.8,5.9,4.5,5.2,8.4
        ,2.8,7.9,1.1,10.1,12.,1.5,5.9,4.1,4.6,5.4,5.5,3.7
        ,18.,5.,8.4,4.,4.,15.5,19.,15.4,15.,11.6,5.2,7.
        ,8.9,7.,3.4,8.3,5.,19.,4.2,12.4,5.2,11.6,4.1,12.4
        ,4.4,4.7,6.9,5.7,2.9,15.6,3.8,16.,16.,8.6,11.9,23.
        ,5.2,17.,14.,23.,8.7,8.8,10.,5.8,4.2,17.9,7.4,17.4
        ,4.6,7.,8.5,4.4,13.9,7.2,11.8,14.,2.,4.5,4.1,17.
        ,19.8,3.1,2.4,18.,4.,7.,2.6,13.8,6.3,7.8,14.,5.8
        ,8.1,2.4,11.6,2.5,13.9,5.,15.2,4.,4.4,14.8,4.5,3.3
        ,12.6,11.8,19.8,22.6,3.5,3.4,7.1,12.6,17.8,1.8,3.5,12.
        ,18.2,10.,4.2,10.4,2.1,12.6,15.6,20.,1.,8.,2.1,3.6
        ,5.,8.,1.9,16.6,15.4,2.2,4.2,3.9,2.8,3.9,5.,7.4
        ,9.1,9.6,17.,13.4,1.6,4.2,15.4,5.2,8.,17.,2.3,4.2
        ,4.,7.5,3.6,11.7,10.,4.3,8.4,12.2,13.4,19.5,15.7,11.9
        ,5.,8.4,9.8,17.8,18.1,15.6,14.9,11.8,17.2,9.8,10.1,3.
        ,11.9,8.5,14.6,5.9,5.1,6.1,9.8,10.,4.6,22.9,3.4,7.7
        ,5.1,3.,5.8,3.6,4.,1.5,2.8,9.,8.8,16.4,3.2,17.
        ,4.8,18.,18.,5.4,18.2,13.,7.8,9.9,13.9,5.9,4.6,3.
        ,17.8,6.1,5.2,6.4,12.7,9.,8.,5.1,6.4,5.8,13.9,15.9
        ,19.2,12.35,3.8,10.6,5.8,7.6,11.6,23.,13.6,8.4,15.4,3.6
        ,1.8,6.,7.5,2.6,16.6,7.6,7.1,3.6,4.5,5.8,10.6,18.4
        ,11.8,10.,15.6,15.3,3.,4.8,8.6,5.2,4.,12.3,14.,19.9
        ,6.,5.4,8.2,2.8,16.2,8.4,5.1,2.4,5.6,2.4,7.6,11.5
        ,15.7,15.8,2.1,7.4,17.8,14.,11.7,12.8,16.,7.,16.8,1.2
        ,13.2,10.2,4.2,5.7,18.1,5.8,13.8,8.5,11.2,12.9,11.2,5.4
        ,5.,1.,16.4,8.6,2.8,5.,14.8,14.,5.4,8.8,20.1,3.3
        ,9.,14.,2.3,4.,12.2,16.6,4.5,16.2,7.4,14.7,7.6,5.1
        ,2.,9.2,5.,4.2,7.5,3.1,4.9,12.9,5.8,8.8,4.4,16.8
        ,12.6,5.9,14.4,1.7,8.,10.,3.,6.6,3.8,4.2,16.4,8.
        ,2.1,17.8,3.7,10.9,4.4,2.7,6.,8.4,15.2,18.2,11.6,14.
        ,17.2,4.9,4.9,11.6,5.8,3.5,6.2,5.2,11.6,12.9,4.2,7.2
        ,13.6,6.,2.1,12.6,14.,7.3,5.5,4.2,1.5,7.4,1.,1.2
        ,2.4,9.2,5.4,5.8,8.7,6.,5.1,7.8,7.3,11.1,11.2,7.2
        ,5.3,5.4,17.8,15.9,5.2,4.,8.5,7.8,14.6,4.4,12.5,13.9
        ,8.9,8.5,16.8,19.4,3.7,18.9,19.,8.7,16.8,2.1,14.,17.9
        ,14.,4.4,7.4,7.8,12.3,3.,9.9,23.,8.6,4.,0.9,16.4
        ,0.9,17.,12.,9.9,19.2,12.3,18.3,4.2,4.7,10.1,5.4,6.
        ,5.4,5.2,4.3,12.8,2.3,17.1,8.6,0.6,15.4,5.8,2.9,18.9
        ,19.4,22.9,3.1,21.6,19.4,7.7,5.,3.2,2.9,13.4,2.3,5.8
        ,4.,6.,5.9,1.7,5.7,5.,2.8,6.8,2.8,9.4,17.8,5.9
        ,15.4,5.3,5.4,6.8,7.8,9.9,1.6,2.7,17.2,6.9,4.2,15.8
        ,3.9,9.2,13.6,5.3,1.6,5.,3.6,7.,6.5,5.2,16.2,12.4
        ,4.4,7.5,10.6,3.8,10.8,4.3,5.8,21.,3.3,6.9,9.,4.5
        ,13.4,12.8,16.2,8.4,16.8,7.4,8.4,3.2,6.,4.8,12.4,17.
        ,3.4,3.4,16.1,15.6,11.5,11.6,10.,14.,8.8,4.,16.2,3.2
        ,8.7,5.4,3.8,7.4,4.3,3.6,14.7,14.6,3.,15.9,10.6,3.2
        ,16.6,15.3,4.4,17.1,5.8,2.5,5.4,8.6,16.6,1.8,9.9,8.4
        ,3.8,12.2,5.1,16.,11.8,8.6,8.6,6.4,5.5,5.9,6.1,19.4
        ,8.,1.8,2.7,2.2,13.9,14.,11.6,5.2,7.8,19.,5.2,17.2
        ,7.4,5.8,9.8,8.9,5.4,16.8,1.3,5.9,14.4,3.4,5.4,4.
        ,4.2,16.6,6.,16.8,15.,10.,15.9,13.2,1.6,7.3,15.3,8.6
        ,11.2,3.7,15.2,12.6,3.9,7.1,1.8,11.3,4.5,5.5,4.4,16.1
        ,5.7,5.8,5.9,11.9,3.8,9.8,4.8,10.2,11.9,4.8,4.4,8.6
        ,16.,5.8,7.7,4.9,5.8,17.7,2.,14.8,5.8,2.,13.7,12.6
        ,16.2,3.5,6.8,13.9,2.,9.,12.4,7.1,7.,5.3,12.8,15.8
        ,18.8,14.5,6.8,5.6,5.7,9.8,19.3,6.6,4.4,9.,16.,7.4
        ,8.6,14.3,4.,11.2,1.1,4.2,1.5,5.2,5.2,14.6,11.6,14.
        ,13.2,3.9,20.,8.2,4.9,9.,7.8,7.3,3.3,4.,17.8,4.4
        ,16.9,3.7,6.,7.4,7.6,6.4,6.2,12.4,7.,2.1,12.3,2.9
        ,4.2,9.,18.,4.4,5.1,8.8,7.4,2.8,14.7,5.8,9.,5.1
        ,15.8,7.,2.4,4.4,9.9,4.,6.1,4.,9.,4.,4.4,6.1
        ,4.9,15.,19.4,4.,13.,1.7,5.4,14.6,7.4,16.,14.,12.2
        ,19.1,4.8,14.1,7.6,9.,15.9,9.,5.6,3.1,1.6,7.,12.8
        ,5.9,1.3,9.2,8.3,4.8,14.4,4.3,3.6,3.3,5.,3.1,3.4
        ,17.,9.9,8.2,6.,16.2,11.6,10.,1.2,4.2,5.4,5.,4.6
        ,6.,18.2,11.3,3.8,12.6,1.9,5.4,5.,11.1,4.3,8.,16.8
        ,6.5,19.4,17.,5.5,5.4,3.4,2.8,17.,11.6,6.2,5.8,16.8
        ,15.7,15.3,16.2,7.2,8.,6.3,8.4,14.8,6.4,5.2,4.1,12.8
        ,3.4,12.6,11.3,19.4,5.3,4.5,16.6,5.2,7.9,5.6,5.2,9.1
        ,3.9,14.,7.8,8.8,2.,7.2,12.6,1.4,4.6,3.,12.6,17.9
        ,8.8,7.,4.5,2.7,1.8,12.6,11.2,7.,11.6,9.,17.6,6.
        ,12.,6.,13.,4.,12.,5.6,14.6,10.,15.6]


    y_train_type=[1,0,0,0,1,0,1,0,0,0,1,1,1,0,1,5,1,0,1,0,5,3,0,5,1,4,5,0,3,0,0,5,0,1,1,1,0
,0,3,0,2,3,5,0,3,5,0,1,0,0,5,1,0,5,0,0,3,5,2,0,0,0,0,2,5,1,5,0,0,1,1,5,1,0
,0,0,1,4,5,5,1,1,5,0,1,0,5,1,1,0,0,5,0,0,0,1,1,0,0,5,1,1,3,0,0,0,0,0,1,0,0
,1,5,3,1,0,0,0,0,1,0,0,0,0,1,1,1,0,5,5,1,0,0,0,3,0,5,0,4,0,5,1,0,0,5,1,0,5
,0,1,5,0,1,0,0,0,1,1,1,0,1,0,0,0,0,3,1,1,5,0,1,0,1,1,1,1,0,0,0,3,5,0,0,1,5
,5,1,1,0,1,1,0,0,0,0,0,0,1,5,1,1,3,0,1,1,1,0,0,1,2,5,0,0,0,4,2,0,0,0,5,3,0
,1,5,5,3,3,0,1,0,0,0,5,0,0,5,5,1,0,5,0,0,5,1,1,5,0,0,0,1,1,1,0,1,1,0,0,0,0
,0,0,5,0,1,0,0,1,0,5,1,0,3,5,5,0,0,1,5,0,0,0,0,0,0,0,1,3,3,3,0,1,5,0,0,0,5
,0,0,0,2,1,5,1,0,3,1,5,0,0,0,5,0,0,0,0,1,0,0,1,0,1,0,0,1,0,0,4,0,1,0,5,5,3
,3,0,0,1,0,0,0,0,5,0,0,5,0,0,5,0,0,5,0,5,3,2,0,2,1,1,1,1,0,5,0,0,5,0,0,0,0
,0,1,5,1,0,1,1,0,0,1,0,3,1,5,0,1,3,0,3,0,3,0,4,0,0,1,0,0,0,5,0,1,4,1,0,0,5
,0,3,0,0,0,0,0,0,1,1,1,0,0,0,1,5,0,0,1,5,0,0,0,0,1,0,3,0,3,0,1,3,3,0,0,5,0
,0,1,5,3,0,5,0,0,1,0,0,0,0,0,0,1,5,0,0,0,0,1,1,1,0,3,0,5,0,0,0,0,1,0,0,0,0
,5,3,0,5,1,0,1,3,0,0,0,1,0,0,0,0,0,1,5,5,0,1,0,1,4,0,1,0,1,0,1,5,1,3,3,5,5
,1,0,0,1,5,3,0,2,0,0,1,1,1,0,0,0,0,0,0,1,0,0,0,1,3,1,0,0,1,1,1,3,0,0,0,1,0
,0,0,0,0,0,5,0,2,5,0,1,1,0,5,0,0,1,5,1,1,3,0,5,0,0,0,0,0,5,0,0,0,0,0,0,0,0
,0,4,1,1,3,1,0,3,3,1,0,1,0,0,0,0,2,0,1,1,0,0,0,0,1,0,0,1,0,0,1,1,0,1,0,3,0
,1,1,5,1,0,0,5,5,1,3,0,0,0,1,0,0,2,0,0,1,3,0,1,0,0,0,0,3,0,0,0,1,0,5,1,1,0
,1,0,2,4,4,1,0,5,0,0,5,0,3,1,0,0,5,1,0,0,0,1,0,2,0,0,1,5,0,0,0,0,0,0,1,3,0
,0,0,0,0,1,0,1,0,0,0,0,0,0,1,0,4,1,0,0,0,0,0,0,0,5,0,0,0,5,0,0,0,0,0,0,0,0
,1,5,1,0,0,3,1,3,1,0,5,0,3,0,0,0,5,0,0,0,0,0,1,0,0,1,2,0,5,1,1,5,0,0,0,0,1
,0,0,2,0,0,4,0,0,3,0,0,0,0,0,1,0,1,2,0,0,5,0,4,1,0,0,3,0,0,5,5,1,0,0,5,0,1
,2,0,3,0,5,0,3,0,0,4,0,1,0,0,0,0,0,1,0,5,0,0,0,1,0,0,0,5,0,0,0,0,0,0,0,0,5
,0,5,5,1,0,0,1,0,0,1,0,0,5,0,0,0,5,0,0,0,0,5,1,0,0,1,5,0,3,0,0,0,1,1,3,0,0
,2,0,0,1,1,0,0,0,1,1,0,3,3,0,0,0,1,0,0,0,0,0,4,0,0,5,5,0,1,0,1,0,0,0,5,1,1
,0,1,0,0,0,0,1,0,5,2,5,0,1,0,0,0,0,4,0,5,0,1,1,1,0,0,0,0,5,2,0,0,0,1,0,5,0
,1,0,0,0,0,0,1,1,0,1,0,1,0,0,1,0,3,0,0,5,0,3,2,0,0,0,5,0,0,3,1,0,0,0,0,0,0
,0,5,0,0,5,4,0,1,0,3,0,0,0,0,3,1,0,0,1,0,0,5,0,0,5,0,0,0,0,1,5,0,0,1,1,0,0
,0,1,0,2,4,0,0,0,1,1,2,0,0,3,3,0,1,0,0,1,0,1,5,1,1,0,0,0,0,0,0,5,1,1,0,0,0
,5,0,0,0,0,0,5,0,0,1,0,0,1,0,0,0,0,3,5,1,0,0,0,0,0,0,5,0,2,5,0,0,3,0,0,0,5
,5,0,0,1,0,0,0,0,0,5,5,0,1,0,5,0,0,1,0,0,0,0,0,0,1,0,1,3,0,0,1,1,0,0,3,0,0
,0,0,0,1,3,0,5,0,1,2,1,0,1,5,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,5,0,1,0,2,0,0,0
,5,1,0,3,2,0,1,0,0,0,0,5,0,5,0,0,1,0,0,5,0,1,0,0,1,0,0,0,1,3,0,0,0,0,0,3,1
,1,1,0,0,5,5,0,0,0,0,0,0,5,1,0,5,0,1,0,3,0,0,5,0,5,0,0,0,1,0,1,0,1,0,4,4,1
,0,1,0,1,0,0,0,0,0,1,1,0,0,0,0,0,5,0,0,0,0,3,5,2,1,0,1,0,0,3,1,0,0,5,0,0,0
,0,0,0,1,0,5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,0,0,0,1,5,0,1,0,1,0,5,0
,0,5,1,5,5,4,5,4,0,0,0,0,0,5,1,4,1,0,0,1,0,5,1,0,0,0,1,1,0,1,0,0,1,0,0,1,0
,0,0,5,0,0,1,3,0,0,0,0,5,0,0,0,1,0,4,3,5,0,2,0,0,0,3,0,1,0,1,1,1,0,3,0,5,0
,1,1,3,5,0,5,5,0,0,0,1,1,0,0,0,0,3,1,0,0,0,0,1,0,5,5,0,5,0,0,0,0,5,1,0,1,0
,5,4,5,5,1,0,0,0,1,1,1,5,0,0,5,2,0,0,0,0,1,0,5,3,0,0,1,5,1,2,1,0,5,1,0,0,0
,5,0,1,0,0,2,0,0,0,0,1,5,2,0,0,5,0,0,0,1,0,5,0,1,1,0,5,1,2,0,0,0,1,0,1,0,1
,3,5,1,0,4,0,1,0,0,1,0,1,3,1,0,0,0,0,3,0,0,1,3,0,1,0,1,0,2,0,0,0,0,0,0,0,3
,0,5,1,0,1,0,1,0,5,1,1,5,0,1,1,3,0,0,0,5,0,0,0,5,5,0,0,5,0,1,0,0,1,0,0,0,0
,0,0,1,4,0,0,3,0,0,5,1,0,0,3,0,0,5,3,1,0,0,1,0,1,0,5,0,0,0,0,5,0,3,1,0,0,0
,5,0,0,0,1,0,0,0,1,0,1,1,5,0,5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,5,2,1
,0,4,0,0,0,0,0,0,0,1,0,5,1,1,0,1,1,1,1,0,0,1,4,0,1,1,1,0,1,0,3,2,5,0,4,1,1
,0,0,1,1,3,3,0,1,5,4,0,0,1,1,0,0,0,0,5,0,3,4,1,0,0,0,0,4,0,0,0,0,0,5,0,1,5
,0,0,0,0,3,0,0,1,0,0,0,1,0,0,0,5,5,1,0,0,0,0,0,0,5,0,0,3,0,0,5,0,0,3,0,0,5
,0,3,1,0,0,0,0,3,0,1,1,0,1,1,1,5,5,4,5,1,0,0,0,0,0,3,0,0,0,0,0,0,5,0,0,0,0
,3,5,0,5,0,5,3,0,0,0,0,0,1,3,5,1,5,0,0,0,0,0,5,5,0,0,5,0,0,0,0,1,3,0,3,5,3
,0,0,0,3,5,3,1,0,0,0,0,1,2,0,0,1,2,0,0,0,0,1,0,1,0,1,0,5,0,0,0,5,0,4,0,0,0
,5,1,1,3,0,0,2,1,0,0,0,0,5,1,0,0,5,1,1,4,0,2,0,4,0,0,1,0,1,1,0,5,0,0,0,3,0
,1,0,0,0,0,0,2,0,0,1,0,5,5,0,0,0,3,0,0,5,0,0,5,3,0,0,5,5,0,0,0,0,5,0,0,5,5
,0,0,5,1,5,5,0,0,5,5,0,0,1,0,0,0,0,1,0,1,0,3,2,0,0,0,5,0,1,0,0,3,5,0,0,0,0
,0,5,1,1,0,0,5,0,4,5,0,0,0,5,3,0,0,4,1,0,0,0,3,1,3,1,2,2,0,0,0,0,0,5,1,0,0
,1,5,0,3,5,3,1,0,2,0,0,1,1,1,1,1,0,0,0,1,0,3,0,0,0,0,3,0,1,3,0,0,1,3,0,0,0
,5,0,5,0,5,1,3,0,4,5,0,0,1,0,5,4,0,0,1,0,5,3,0,0,0,0,0,1,1,0,1,0,1,1,0,0,0
,0,0,0,0,1,5,4,0,0,1,0,0,0,0,0,1,3,1,0,0,4,5,0,0,0,0,3,3,1,4,0,0,2,0,1,0,0
,0,5,3,3,1,0,3,3,0,0,1,0,1,1,1,0,1,1,0,0,5,1,5,5,5,3,1,0,0,0,0,0,0,0,0,0,0
,0,0,0,3,0,4,0,0,0,0,1,0,0,0,0,0,0,0,1,5,5,0,0,1,0,1,0,5,0,0,0,0,0,1,5,0,1
,1,1,0,0,1,0,0,1,0,5,1,0,5,5,4,5,0,1,1,0,2,5,4,5,0,0,0,1,5,1,0,1,5,5,0,1,0
,0,4,1,0,0,1,2,0,1,1,0,0,0,3,0,1,0,1,5,1,0,0,0,0,0,0,1,5,0,0,5,1,1,0,1,3,3
,0,0,3,5,0,0,5,0,0,0,1,1,0,0,0,0,5,3,0,0,0,5,0,0,0,0,0,1,0,0,1,0,0,1,0,0,0
,0,0,1,0,0,3,1,0,0,0,0,0,0,0,0,0,3,1,1,5,0,0,4,1,1,3,3,0,0,0,0,1,3,0,0,0,0
,0,0,1,0,5,5,0,5,0,0,0,3,5,0,0,0,1,1,5,0,0,1,0,0,0,1,1,5,1,1,0,0,0,0,0,5,0
,5,0,0,0,1,0,5,0,0,1,1,5,0,1,0,5,2,3,0,0,1,3,0,1,0,0,1,1,0,0,0,0,5,1,0,5,0
,0,3,0,0,0,1,0,3,0,5,1,0,1,1,0,5,0,0,3,1,5,0,0,0,0,1,1,0,0,0,0,1,2,0,0,1,0
,0,0,4,0,1,0,0,0,0,5,0,1,0,0,5,0,0,5,1,1,0,0,0,5,1,0,0,0,0,5,5,0,0,0,1,5,0
,5,0,0,0,0,0,3,0,0,0,0,3,5,1,3,0,1,1,0,0,0,3,1,0,1,0,1,0,0,5,0,0,0,0,0,1,1
,1,4,1,5,0,0,0,0,1,0,0,0,0,1,0,2,0,0,2,0,1,1,0,0,0,0,0,5,0,0,5,0,1,0,0,5,0
,1,3,1,3,0,0,4,0,0,0,0,4,0,0,0,0,0,0,1,3,0,1,0,5,0,3,5,1,0,1,4,0,0,1,0,0,0
,0,5,5,0,0,0,1,0,5,0]

urlpatterns = [
  url(r'^behaviour/$', behaviour),
url(r'^watercolor/$', waterColor),
url(r'^filter/$', filter),
url(r'^consumption/$', consumption),
url(r'^topFish/$', topFish),
url(r'^sideFish/$', sideFish),
url(r'^fishLength/$', fishLength),
 
]
    