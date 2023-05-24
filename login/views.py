from django.contrib.auth import login,authenticate,logout
from django.contrib.auth.decorators import login_required
from django.shortcuts import render,redirect
from django.contrib import messages
import os
from django.core.files.storage import FileSystemStorage

name="1"
media="media"
message1='1'
key1="none"

@login_required
def home(request):
    return render(request,'upload.html')

def login_user(request):
    if request.method=="POST":
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect('home')
        else:
            messages.success(request,('Provide valid usename and password..Try again!!!'))
            return redirect('login')
            
    else:
        return render(request,'login.html',{})
def logout_user(request):
    logout(request)
    messages.success(request,('You were logged out!!!'))
    return redirect('login')
@login_required
def upload_verfiy(request):
    global name
    global verification
    global key1
    global message1
    if request.method=="POST" and request.FILES['filePath']:
        if 'filePath' not in request.FILES:
            err='No Image Selected'
            return render(request,'upload.html',{
                'err':err
            })
        f=request.FILES['filePath']
        name=request.POST.get('cid')
        print(name)
        if f=='':
            err='No files selected'
            return render(request,'upload.html',{
                'err':err
            })
        filePath=request.FILES['filePath']
        fss=FileSystemStorage()
        # if fss.exists(filePath.name):
        #     fss.delete(filePath.name)
        file=fss.save(filePath.name,filePath)
        file_url=fss.url(file)
        makeVerification(os.path.join(media,file))
        #dict={"key1":verification,
            #}
        fss.delete(filePath.name)
        if name==message1:
            key1="real"
        else:
            key1="forgery"
        
        return render(request,'upload.html',{
            #'file_url':file_url,
            'key1':key1,
            #'key2':message1
        })

    else:
        return render(request,'upload.html')


def makeVerification(path):
    global name
    global verification
    global message1
    import tensorflow as tf
    import matplotlib.pyplot as plt
    import cv2
    import os
    import numpy as np
    from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array
    from tensorflow.keras.preprocessing import image
    import pandas as pd
    import itertools
    import sklearn
    from tensorflow.keras.models import load_model

    #name = "001"

    # this is for submitted image to array
    folder = "C:/Users/safal/minor project/Signature-Verification--2/media"
    def load_images_from_folder(folder_path):
        for image in os.listdir(folder_path):
            img_arr = cv2.imread(os.path.join(
                folder, image), cv2.IMREAD_GRAYSCALE)
            img_arr = img_arr/255
            new_arr = cv2.resize(img_arr, (224, 224))
            return new_arr
    img = load_images_from_folder(folder)

    # this is for customer id to image array
    #DATADIR = "C:/Users/safal/Desktop/dataset"
    #CATEGORIES = ['001','003','009','sir3','001_forg','003_forg','009_forg','sir3_forg']
    #data = []  # this is a list
    #labl = []

    #def create_image_data():
    #    for category in CATEGORIES:
    #        class_num = CATEGORIES.index(category)
            # path to 001 or 001_forg or 002 ----dir
    #        path = os.path.join(DATADIR, category)
    #        for img in os.listdir(path):
    #            img_array = cv2.imread(os.path.join(
     #               path, img), cv2.IMREAD_GRAYSCALE)
     #           img_array = img_array/255
     #           new_array = cv2.resize(img_array, (100, 100))
                # print(new_array)
     #           data.append([new_array])
     #           labl.append(class_num)

    #create_image_data()
    #labl = np.array(labl)
    #images = np.array(data)
    #image = images[55]

    # cid wala lai label ma change haneko ani tyo label anusar ko image read gareko dataset bata
    #if name == "001":#rea
    #    label = 3
    #    image = images[label]
    #elif name == "002":
    #    label =28
    #    image = images[label]
    #elif name == "003":
    #    label = 55
    #    image = images[label]
    #elif name == "004":
    #    label = 73
    #    image = images[label]
    #else:
    #    error_message = "not a valid customer id"
    # aba model lai load garna lageko

    # customer id bata derieve vako image 'image' variable ma xa ani submit garda deko image chai 'img' array ma aaxa
    # the code below is for loading the model
    model = load_model('cnnro,pr,sa.h5')
    predictions = model.predict(img.reshape((1, 224, 224,1)))
    category= tf.argmax(predictions,axis=1)
    message1 = str(category.numpy()[0])
    print("Predicted class:", message1)
    #verification = siamese_model.predict(
    #    [image.reshape((1, 100, 100)), img.reshape((1, 100, 100))]).flatten()
    #verification = verification*100
    

    