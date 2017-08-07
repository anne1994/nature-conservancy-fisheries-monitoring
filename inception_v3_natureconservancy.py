#! /usr/bin/python2.7

import matplotlib
matplotlib.use('Agg')
import itertools


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))


import os
import re

import tensorflow as tf
import tensorflow.python.platform
from tensorflow.python.platform import gfile
import numpy as np
import pandas as pd
import sklearn
from sklearn import cross_validation
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.svm import SVC, LinearSVC
import matplotlib.pyplot as plt
import pickle
from sklearn.calibration import CalibratedClassifierCV
from sklearn import svm
from os.path import isdir, join, exists
from shutil import copyfile
from os import listdir, makedirs
import matplotlib.pyplot as plt

model_dir = '/data/models/slim/'

#img_path = join(model_dir,"svmimages")





image_dir = '/data/models/slim/train/images/'
list_images = [image_dir+f for f in os.listdir(image_dir) if re.search('jpg|JPG', f)]
#list_images = [f for f in os.listdir(images_dir) ]
print("list of images", list_images)

# setup tensorFlow graph initiation
def create_graph():
    with gfile.FastGFile(os.path.join(model_dir, 'tensorflow_inception_graph.pb'), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

# extract all features from pool layer of InceptionV3
def extract_features(list_images):
	nb_features = 2048
	features = np.empty((len(list_images),nb_features))
	labels = []
	create_graph()
	with tf.Session() as sess:
		next_to_last_tensor = sess.graph.get_tensor_by_name('pool_3:0')
		for ind, image in enumerate(list_images):
			print('Processing %s...' % (image))
			if not gfile.Exists(image):
				tf.logging.fatal('File does not exist %s', image)
			image_data = gfile.FastGFile(image, 'rb').read()
			predictions = sess.run(next_to_last_tensor,
			{'DecodeJpeg/contents:0': image_data})
			features[ind,:] = np.squeeze(predictions)
			#labels.append(re.split('\s',image.split('/')[1])[0])
                        image=image.rsplit("/",1)[1]
                        labels.append(image.split("i", 1)[0])
                        #print("labels", labels)
		return features, labels


features,labels = extract_features(list_images)

pickle.dump(features, open('features', 'wb'))
pickle.dump(labels, open('labels', 'wb'))

features = pickle.load(open('features'))
labels = pickle.load(open('labels'))

print("labels", np.array(labels))
print("features", np.array(features))

# run a 10-fold CV SVM using probabilistic outputs.
X_train, X_test, y_train, y_test = cross_validation.train_test_split(features, labels, test_size=0.2, random_state=0)
clf = svm.SVC(kernel='linear', C=1)
clf.fit(X_train, y_train)
clf.score(X_train, y_train)
y_pred=clf.predict(X_test)

y_pred_train=clf.predict(X_train)


labels = sorted(list(set(labels)))
print("\n Testing set Confusion matrix:")
print("Labels: {0}\n".format(",".join(labels)))
print(confusion_matrix(y_test, y_pred, labels=labels))

print("\nTesting set Classification report:")
print(classification_report(y_test, y_pred))

print("Accuracy on the testing images:{0:0.1f}%".format(accuracy_score(y_test,y_pred)*100))

def plot_confusion_matrix(cm, classes,normalize=False, title='confusion_matrix', cmap=plt.cm.Blues):
        plt.imshow(cm,interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks,classes,rotation=45)
        plt.yticks(tick_marks,classes)
        if normalize:
                cm=cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]
                print("Normalized")
        else:
                print("non normalised")
        print(cm)
        thresh = cm.max()/2.
        for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                plt.text(j,i,cm[i,j],horizontalalignment="center",color="white" if cm[i,j]>thresh else "black")
        plt.tight_layout()
        plt.ylabel('true label')
        plt.xlabel('predicted label')

cnf_matrix1 = confusion_matrix(y_test,y_pred)
np.set_printoptions(precision=2)
plt.figure()
plot_confusion_matrix(cnf_matrix1, classes = labels, title='cnf_non_normalized')
plt.show()
plt.savefig("/data/models/slim/confusionmatrix_svm_natureconservancy_validationdata.png")


labels = sorted(list(set(labels)))
print("\n Training set Confusion matrix:")
print("Labels: {0}\n".format(",".join(labels)))
print(confusion_matrix(y_train, y_pred_train, labels=labels))

print("Accuracy on the training set:{0:0.1f}%".format(accuracy_score(y_train,y_pred_train)*100))


def plot_confusion_matrix(cm, classes,normalize=False, title='confusion_matrix', cmap=plt.cm.Blues):
        plt.imshow(cm,interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks,classes,rotation=45)
        plt.yticks(tick_marks,classes)
        if normalize:
                cm=cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]
                print("Normalized")
        else:
                print("non normalised")
        print(cm)
        thresh = cm.max()/2.
        for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                plt.text(j,i,cm[i,j],horizontalalignment="center",color="white" if cm[i,j]>thresh else "black")
        plt.tight_layout()
        plt.ylabel('true label')
        plt.xlabel('predicted label')

cnf_matrix = confusion_matrix(y_train,y_pred_train)
np.set_printoptions(precision=2)
plt.figure()
plot_confusion_matrix(cnf_matrix, classes = labels, title='cnf_non_normalized')
plt.show()
plt.savefig("/data/models/slim/confusionmatrix_svm_natureconservancy.png")




#####test on new images#########

test_dir='/data/models/slim/test_stg1/'
list_images = [test_dir+f for f in os.listdir(test_dir) if re.search('jpg|JPG', f)]


def extract_features(list_images):
       	nb_features = 2048
	features = np.empty((len(list_images),nb_features))
	create_graph()
	with tf.Session() as sess:
		next_to_last_tensor = sess.graph.get_tensor_by_name('pool_3:0')
		for ind, image in enumerate(list_images):
			print('Processing %s...' % (image))
			if not gfile.Exists(image):
				tf.logging.fatal('File does not exist %s', image)
			image_data = gfile.FastGFile(image, 'rb').read()
			predictions = sess.run(next_to_last_tensor,
			{'DecodeJpeg/contents:0': image_data})
			features[ind,:] = np.squeeze(predictions)
		return features


features_test = extract_features(list_images)

y_pred = clf.predict(features_test)

print("prediction on new images", y_pred)
