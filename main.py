import numpy as np
import matplotlib.pyplot as plt
from PIL.ImageOps import flip
from tensorflow.keras import layers, models
import tensorflow as tf
from PIL import Image
from sklearn.metrics import confusion_matrix

path = "C:/Users/Jamaica/Desktop/Facultate/An 2/Sem 2/IA/ImageClassification/train/"
#teste preliminare pentru input
# img = Image.open(path + "000000.png")
# rotate_img = img.rotate(90)
# # citirea unei imagini
# np_img = np.array(img)
# # afisarea unei imagini
# plt.imshow(img, cmap = 'binary')
# plt.imshow(rotate_img, cmap = 'binary')
# plt.show()
imgName = ""
imgLabel = 0
train_images = []
train_labels =[]
#citire din fisier
f = open("C:/Users/Jamaica/Desktop/Facultate/An 2/Sem 2/IA/ImageClassification/train.txt" , "r")
try:
    for i in range(30001):
        tempString = f.readline()
        imgName = tempString[:10]
        #salvez numele imaginii
        imgLabel = int(tempString[-2:-1])
        #salvez labelul imaginii
        img = Image.open(path + imgName)
        # if((17 * i + 5) %23 == 13):
        #     img = img.rotate(90)
        # rotate_img = img.rotate(90)
        np_img = np.array(img)/255
        #salvez imaginea ca vector numpy
        train_images.append(np_img)
        #salvez labelul imaginii
        train_labels.append(imgLabel)
finally:
    f.close()

#incercare de data augmentation
    f = open("C:/Users/Jamaica/Desktop/Facultate/An 2/Sem 2/IA/ImageClassification/train.txt", "r")

    # try:
    #     for i in range(30001):
    #         tempString = f.readline()
    #         imgName = tempString[:10]
    #         # salvez numele imaginii
    #         imgLabel = int(tempString[-2:-1])
    #         # salvez labelul imaginii
    #         img = Image.open(path + imgName)
    #         rotate_img = img.rotate(90)
    #         np_img = np.array(rotate_img) / 255.0
    #         # salvez imaginea ca vector numpy
    #         train_images.append(np_img)
    #         # salvez labelul imaginii
    #         train_labels.append(imgLabel)
    # finally:
    #     f.close()

#citire validation
path = "C:/Users/Jamaica/Desktop/Facultate/An 2/Sem 2/IA/ImageClassification/validation/"
validation_images = []
validation_labels =[]
f = open("C:/Users/Jamaica/Desktop/Facultate/An 2/Sem 2/IA/ImageClassification/validation.txt" , "r")
try:
    for i in range(30001, 35001):
        tempString = f.readline()
        imgName = tempString[:10]
        #salvez numele imaginii
        imgLabel = int(tempString[-2:-1])
        #salvez labelul imaginii
        img = Image.open(path + imgName)
        np_img = np.array(img)/255
        #salvez imaginea ca vector numpy
        validation_images.append(np_img)
        #salvez labelul imaginii
        validation_labels.append(imgLabel)
finally:
    f.close()
train_images = np.asarray(train_images)
train_labels = np.asarray(train_labels)

train_images = train_images.reshape(-1 , 32, 32, 1)

validation_labels = np.asarray(validation_labels)
validation_images = np.asarray(validation_images)

validation_images = validation_images.reshape(-1, 32, 32, 1)

#ultimul model rulat - modelul care a generat ultimul csv incarcat pe kaggle
# liniile de cod comentate nu au fost sterse pentru a arata o modificare adusa fata de un model anterior
# model = models.Sequential()
# model.add(layers.Conv2D(64, (5,5),  activation='relu',data_format='channels_last'))
# model.add(layers.MaxPooling2D((2,2), data_format='channels_last', padding='same'))
# # model.add(layers.Conv2D(64 , (5,5) , activation='relu', data_format='channels_last'))
# # model.add(layers.MaxPooling2D((2,2), data_format='channels_last', padding='same'))
# model.add(layers.Conv2D(256 , (5,5), activation='relu', data_format='channels_last'))
# model.add(layers.MaxPooling2D((2,2), data_format='channels_last', padding='same'))
# # model.add(layers.Conv2D(32 , (5,5) , activation='relu', data_format='channels_last', padding='same'))
# # model.add(layers.MaxPooling2D((2,2), data_format='channels_last', padding='same'))
# # model.add(layers.Conv2D(128 , (5,5) , activation='relu', data_format='channels_last', padding='same'))
#
# model.add(layers.Flatten())
# model.add(layers.Dropout(0.5))
# # model.add(layers.Dense(128))
# model.add(layers.Dense(128))
# model.add(layers.Dense(32))
# model.add(layers.Dense(9, activation='softmax'))
# model.compile(optimizer = 'adam' , loss= 'sparse_categorical_crossentropy', metrics=['accuracy'])
# model.fit(train_images, train_labels, epochs=25, validation_data=(validation_images, validation_labels))


#1 : 87.30 salvat ca image_classifier.model
# model = models.Sequential()
# model.add(layers.Conv2D(32, (5,5),  activation='relu',data_format='channels_last'))
# model.add(layers.MaxPooling2D((2,2), data_format='channels_last', padding='same'))
# model.add(layers.Conv2D(64 , (5,5) , activation='relu', data_format='channels_last'))
# model.add(layers.MaxPooling2D((2,2), data_format='channels_last', padding='same'))
# model.add(layers.Conv2D(64 , (5,5), activation='relu', data_format='channels_last'))
# model.add(layers.MaxPooling2D((2,2), data_format='channels_last', padding='same'))
# model.add(layers.Conv2D(32 , (5,5) , activation='relu', data_format='channels_last', padding='same'))
# model.add(layers.MaxPooling2D((2,2), data_format='channels_last', padding='same'))
# model.add(layers.Flatten())
# model.add(layers.Dropout(0.5))
# model.add(layers.Dense(32))
# model.add(layers.Dense(9, activation='softmax'))
#2 : 87.46 salvat ca image_classifier2.model
# model = models.Sequential()
# model.add(layers.Conv2D(32, (5,5),  activation='relu',data_format='channels_last'))
# model.add(layers.MaxPooling2D((2,2), data_format='channels_last', padding='same'))
# model.add(layers.Conv2D(64 , (5,5) , activation='relu', data_format='channels_last'))
# model.add(layers.MaxPooling2D((2,2), data_format='channels_last', padding='same'))
# model.add(layers.Conv2D(64 , (5,5), activation='relu', data_format='channels_last'))
# model.add(layers.MaxPooling2D((2,2), data_format='channels_last', padding='same'))
# model.add(layers.Conv2D(32 , (5,5) , activation='relu', data_format='channels_last', padding='same'))
# model.add(layers.MaxPooling2D((2,2), data_format='channels_last', padding='same'))
# model.add(layers.Conv2D(32 , (5,5) , activation='relu', data_format='channels_last', padding='same'))
# model.add(layers.MaxPooling2D((2,2), data_format='channels_last', padding='same'))
# model.add(layers.Flatten())
# model.add(layers.Dropout(0.7))
# model.add(layers.Dense(32))
# model.add(layers.Dense(9, activation='softmax'))
#3 : 87.42 salvat ca image_classifier3.model
# model = models.Sequential()
# model.add(layers.Conv2D(32, (5,5),  activation='relu',data_format='channels_last'))
# model.add(layers.MaxPooling2D((2,2), data_format='channels_last', padding='same'))
# model.add(layers.Conv2D(64 , (5,5) , activation='relu', data_format='channels_last'))
# model.add(layers.MaxPooling2D((2,2), data_format='channels_last', padding='same'))
# model.add(layers.Conv2D(64 , (5,5), activation='relu', data_format='channels_last'))
# model.add(layers.MaxPooling2D((2,2), data_format='channels_last', padding='same'))
# model.add(layers.Conv2D(128 , (5,5) , activation='relu', data_format='channels_last', padding='same'))
# model.add(layers.MaxPooling2D((2,2), data_format='channels_last', padding='same'))
# model.add(layers.Conv2D(128 , (5,5) , activation='relu', data_format='channels_last', padding='same'))
#model 4 : 87.84 model intermediar
# model.add(layers.Conv2D(32, (5,5),  activation='relu',data_format='channels_last'))
# model.add(layers.MaxPooling2D((2,2), data_format='channels_last', padding='same'))
# model.add(layers.Conv2D(32 , (5,5) , activation='relu', data_format='channels_last'))
# model.add(layers.MaxPooling2D((2,2), data_format='channels_last', padding='same'))
# model.add(layers.Conv2D(64 , (5,5), activation='relu', data_format='channels_last'))
# model.add(layers.MaxPooling2D((2,2), data_format='channels_last', padding='same'))
# model.add(layers.Conv2D(64 , (5,5) , activation='relu', data_format='channels_last', padding='same'))
# model.add(layers.MaxPooling2D((2,2), data_format='channels_last', padding='same'))
# model.add(layers.Conv2D(128 , (5,5) , activation='relu', data_format='channels_last', padding='same'))
# model.add(layers.MaxPooling2D((2,2), data_format='channels_last', padding='same'))
# model.add(layers.Conv2D(128 , (5,5) , activation='relu', data_format='channels_last', padding='same'))
#model 5 : 88.06 validare 90.16 test trimis ca final submission salvat ca image_classifier4.model
# model = models.Sequential()
# model = models.Sequential()
# model.add(layers.Conv2D(32, (5,5),  activation='relu',data_format='channels_last'))
# model.add(layers.MaxPooling2D((2,2), data_format='channels_last', padding='same'))
# model.add(layers.Conv2D(64 , (5,5) , activation='relu', data_format='channels_last'))
# model.add(layers.MaxPooling2D((2,2), data_format='channels_last', padding='same'))
# model.add(layers.Conv2D(64 , (5,5), activation='relu', data_format='channels_last'))
# # model.add(layers.MaxPooling2D((2,2), data_format='channels_last', padding='same'))
# model.add(layers.Conv2D(32 , (5,5) , activation='relu', data_format='channels_last', padding='same'))
# model.add(layers.MaxPooling2D((2,2), data_format='channels_last', padding='same'))
# model.add(layers.Conv2D(128 , (5,5) , activation='relu', data_format='channels_last', padding='same'))
#
# model.add(layers.Flatten())
# model.add(layers.Dropout(0.75))
# model.add(layers.Dense(128))
# model.add(layers.Dense(64))
# model.add(layers.Dense(32))
# model.add(layers.Dense(9, activation='softmax'))
# model.compile(optimizer = 'adam' , loss= 'sparse_categorical_crossentropy', metrics=['accuracy'])
# model.fit(train_images, train_labels, epochs=25, validation_data=(validation_images, validation_labels))\
#model 6 : 88.24 model intermediar nesalvat
# model = models.Sequential()
# model.add(layers.Conv2D(32, (5,5),  activation='relu',data_format='channels_last'))
# model.add(layers.MaxPooling2D((2,2), data_format='channels_last', padding='same'))
# model.add(layers.Conv2D(64 , (5,5) , activation='relu', data_format='channels_last'))
# model.add(layers.MaxPooling2D((2,2), data_format='channels_last', padding='same'))
# model.add(layers.Conv2D(128 , (5,5), activation='relu', data_format='channels_last'))
# # model.add(layers.MaxPooling2D((2,2), data_format='channels_last', padding='same'))
# model.add(layers.Conv2D(64 , (5,5) , activation='relu', data_format='channels_last', padding='same'))
# model.add(layers.MaxPooling2D((2,2), data_format='channels_last', padding='same'))
# model.add(layers.Conv2D(256 , (5,5) , activation='relu', data_format='channels_last', padding='same'))
#
# model.add(layers.Flatten())
# model.add(layers.Dropout(0.75))
# model.add(layers.Dense(512))
# model.add(layers.Dense(128))
# model.add(layers.Dense(32))
# model.add(layers.Dense(9, activation='softmax'))
#model 7 : 88.34 model intermediar nesalvat
# model = models.Sequential()
# model.add(layers.Conv2D(32, (5,5),  activation='relu',data_format='channels_last'))
# model.add(layers.MaxPooling2D((2,2), data_format='channels_last', padding='same'))
# # model.add(layers.Conv2D(64 , (5,5) , activation='relu', data_format='channels_last'))
# # model.add(layers.MaxPooling2D((2,2), data_format='channels_last', padding='same'))
# model.add(layers.Conv2D(128 , (5,5), activation='relu', data_format='channels_last'))
# # model.add(layers.MaxPooling2D((2,2), data_format='channels_last', padding='same'))
# # model.add(layers.Conv2D(32 , (5,5) , activation='relu', data_format='channels_last', padding='same'))
# # model.add(layers.MaxPooling2D((2,2), data_format='channels_last', padding='same'))
# # model.add(layers.Conv2D(128 , (5,5) , activation='relu', data_format='channels_last', padding='same'))
#
# model.add(layers.Flatten())
# model.add(layers.Dropout(0.5))
# # model.add(layers.Dense(128))
# model.add(layers.Dense(64))
# model.add(layers.Dense(32))
# model.add(layers.Dense(9, activation='softmax'))

#model 8: 89.30 model intermediar nesalvat
# model = models.Sequential()
# model.add(layers.Conv2D(64, (5,5),  activation='relu',data_format='channels_last'))
# model.add(layers.MaxPooling2D((2,2), data_format='channels_last', padding='same'))
# # model.add(layers.Conv2D(64 , (5,5) , activation='relu', data_format='channels_last'))
# # model.add(layers.MaxPooling2D((2,2), data_format='channels_last', padding='same'))
# model.add(layers.Conv2D(128 , (5,5), activation='relu', data_format='channels_last'))
# model.add(layers.MaxPooling2D((2,2), data_format='channels_last', padding='same'))
# # model.add(layers.Conv2D(32 , (5,5) , activation='relu', data_format='channels_last', padding='same'))
# # model.add(layers.MaxPooling2D((2,2), data_format='channels_last', padding='same'))
# # model.add(layers.Conv2D(128 , (5,5) , activation='relu', data_format='channels_last', padding='same'))
#
# model.add(layers.Flatten())
# model.add(layers.Dropout(0.5))
# # model.add(layers.Dense(128))
# model.add(layers.Dense(64))
# model.add(layers.Dense(32))
# model.add(layers.Dense(9, activation='softmax'))
# model.compile(optimizer = 'adam' , loss= 'sparse_categorical_crossentropy', metrics=['accuracy'])
# model.fit(train_images, train_labels, epochs=25, validation_data=(validation_images, validation_labels))
#
# loss, accuracy = model.evaluate(validation_images, validation_labels)
# print(f"Loss {loss}")
# print(f"Accuracy {accuracy}")
#model 9 : 89.94 salvat ca image_classifierNou.model model final
# model = models.Sequential()
# model.add(layers.Conv2D(64, (5,5),  activation='relu',data_format='channels_last'))
# model.add(layers.MaxPooling2D((2,2), data_format='channels_last', padding='same'))
# # model.add(layers.Conv2D(64 , (5,5) , activation='relu', data_format='channels_last'))
# # model.add(layers.MaxPooling2D((2,2), data_format='channels_last', padding='same'))
# model.add(layers.Conv2D(256 , (5,5), activation='relu', data_format='channels_last'))
# model.add(layers.MaxPooling2D((2,2), data_format='channels_last', padding='same'))
# # model.add(layers.Conv2D(32 , (5,5) , activation='relu', data_format='channels_last', padding='same'))
# # model.add(layers.MaxPooling2D((2,2), data_format='channels_last', padding='same'))
# # model.add(layers.Conv2D(128 , (5,5) , activation='relu', data_format='channels_last', padding='same'))
#
# model.add(layers.Flatten())
# model.add(layers.Dropout(0.5))
# # model.add(layers.Dense(128))
# model.add(layers.Dense(128))
# model.add(layers.Dense(32))
# model.add(layers.Dense(9, activation='softmax'))
# model.compile(optimizer = 'adam' , loss= 'sparse_categorical_crossentropy', metrics=['accuracy'])
# model.fit(train_images, train_labels, epochs=25, validation_data=(validation_images, validation_labels))
#
# loss, accuracy = model.evaluate(validation_images, validation_labels)
# print(f"Loss {loss}")
# print(f"Accuracy {accuracy}")

# ----------------------
#afisari intermediare
# loss, accuracy = model.evaluate(validation_images, validation_labels)
# print(f"Loss {loss}")
# print(f"Accuracy {accuracy}")
# #testare predictie
# # pred = model.predict(validation_images[0].reshape(1,32,32,1)*255)
# # print(pred)
# # print(validation_labels[0])
#

# -----------------------------------
# salvare
#model.save('image_classifier.model')
#model.save('image_classifier2.model')
#model.save('image_classifier3.model')
#model.save('image_classifier4.model')
# model.save('image_classifierNou.model')
#modelul 9 este salvat in image_classifierNou si are scorul cel mai mare


# --------------------------------------
#afisare in csv

model = models.load_model('image_classifierNou.model')
#liniile comentate au fost lasate in cod pentru a arata procesul de testare
# # model2 = models.load_model('image_classifier.model')
# # model = models.load_model('image_classifier3.model')
#
path = "C:/Users/Jamaica/Desktop/Facultate/An 2/Sem 2/IA/ImageClassification/test/"
test_images = []
f = open("C:/Users/Jamaica/Desktop/Facultate/An 2/Sem 2/IA/ImageClassification/test.txt" , "r")
try:
    for i in range(35001, 40001):
        tempString = f.readline()
        imgName = tempString[:10]
#salvez numele imaginii
        img = Image.open(path + imgName)
        np_img = np.array(img)/255
#       np_img = np_img.reshape(32, 32, 1)
#salvez imaginea ca vector numpy
        test_images.append(np_img)
finally:
    f.close()

test_images = np.asarray(test_images)
test_images = test_images.reshape(5000, 32, 32, 1)
#--------------------------------------------
import csv
# model de scriere in csv
# with open('predictions.csv', mode='w', newline='') as predictions:
#     predictions_writer = csv.writer(predictions, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
#     predictions_writer.writerow(['id', 'label'])
#     pred = model.predict(test_images)
#     for i in range(5000):
#         index = np.argmax(pred[i])
#         name = '0'+ str(i+35001) +'.png'
#         predictions_writer.writerow([name , str(index)])

# --------------------------------
# ultimul fisier scris, cel incarcat la final pe kaggle
with open('predictionsNou4.csv', mode='w', newline='') as predictions:
    predictions_writer = csv.writer(predictions, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    predictions_writer.writerow(['id', 'label'])
    pred = model.predict(test_images)
    for i in range(5000):
        index2 = np.argmax(pred[i])
        name = '0'+ str(i+35001) +'.png'
        predictions_writer.writerow([name , str(index2)])
# -------------------------
#scriere initiala in .txt pentru verificare cod, am incercat sa fac apoi conversia din txt in csv fara succes
# f = open("predictions.txt", "w")
# try:
#     f.write('id,label')
#     f.write('\n')
#     pred = model.predict(test_images)
#     for i in range(5000):
#         index = np.argmax(pred[i])
#         f.write('0'+ str(i+35001) +'.png'+','+ str(index))
#         f.write('\n')
# finally:
#     f.close()
# f = open("predictions2.txt", "w")
# try:
#     f.write('id,label')
#     f.write('\n')
#     pred = model2.predict(test_images)
#     for i in range(5000):
#         index = np.argmax(pred[i])
#         f.write('0'+ str(i+35001) +'.png'+','+ str(index))
#         f.write('\n')
# finally:
#     f.close()

#generare si afisare matrice de confuzie
model = models.load_model('image_classifier.model')
predicted_labels = []
for i in validation_images:
    pred = model.predict(i.reshape(1, 32, 32, 1) * 255)
    prediction = np.argmax(pred)
    predicted_labels.append(prediction)
#testari predictii
# print(true_labels)
# print(predicted_labels)
# pred = model.predict(validation_images[0].reshape(1,32,32,1)*255)
# index = np.argmax(pred)
# print(index)
cm = confusion_matrix(validation_labels, predicted_labels)
print(cm)