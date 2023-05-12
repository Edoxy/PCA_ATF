import numpy as np
import matplotlib.pylab as plt
import scipy as sc
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score
from PCA import PCA

'''
Homework 1 Analisi Tempo Frequenza
Created on April 2023

author: Edoardo Vay
'''

img_set = np.zeros((64, 64, 10))
x = plt.imread(f"archive/s1/1.pgm")

# print(type(x))
H = x.shape[0] # height
W = x.shape[1] # width
print('Images Dimensions: ', H, W)

train = []
train_label = []

test = []
test_label = []


n_test = 3
'''Number of faces not used in training for each person'''

n_people = 1
'''Number of people not used in training'''

print('Number of images used for testing for each person: ', n_test)


# main loop to read the dataset
for i in range(1, 41 - n_people, 1):
    for j in range(1, 11 - n_test, 1):
        img = plt.imread(f'archive/s{i}/{j}.pgm')
        img = img.reshape((H * W))
        train.append(img)
        train_label.append(i)

    for j in range(11 - n_test, 11, 1):
        img = plt.imread(f'archive/s{i}/{j}.pgm')
        img = img.reshape((H * W))
        test.append(img)
        test_label.append(i)

new_person = 41 - n_people

for i in range(new_person, 41, 1):

    for j in range(1, 11):
        img = plt.imread(f'archive/s{i}/{j}.pgm')
        img = img.reshape((H * W))
        test.append(img)
        test_label.append(new_person)


train = np.array(train)
train_label = np.array(train_label)
test = np.array(test)
test_label = np.array(test_label)
# print(train.shape, train_label.shape)
# print(test.shape, test_label.shape)

components = 92
print('Number of component used on PCA: ', components)

pca = PCA(n_components=components)
pca.fit(train, svd_decomposition=False)
pca_train = pca.transform(train)
# print(pca_train.shape)
img = pca.inverse_transform(pca_train)
# print(img.shape)

error_train = np.zeros((train.shape[0]))

for i in range(train.shape[0]):
    error_train[i] = np.linalg.norm(img[i] - train[i])

soglia = np.amax(error_train) * 1.5

print(f'Soglia stimata : {soglia:.{2}f}')
#################################
# Plot Facce Ricostruite

n_soggetti = 4
n_foto = 4

fig1, ax1 = plt.subplots(nrows=n_soggetti, ncols=n_foto)

for i in range(n_soggetti):
    for j in range(n_foto):
        ax1[i, j].imshow(img[(10-n_test) * i + j].reshape((H, W)))


#################################
# Riconoscimento

pca_test = pca.transform(test)
projected_test = pca.inverse_transform(pca_test)

error = np.zeros((test.shape[0]))

for i in range(test.shape[0]):
    error[i] = np.linalg.norm(projected_test[i] - test[i])

fig2, ax2 = plt.subplots()
ax2.hist(error, histtype='step', density=True, bins=20)

pred_label = np.zeros((test.shape[0]), dtype=np.int32)
for i, image in enumerate(pca_test):

    # test if person is already present
    if error[i] > soglia:
        pred_label[i] = new_person

    # search which pearson is
    else:
        min_dist = np.inf
        for j, subject in enumerate(pca_train):
            err = np.linalg.norm(image - subject)
            if err < min_dist:
                min_dist = err
                pred_label[i] = train_label[j]

disp = ConfusionMatrixDisplay.from_predictions(test_label, pred_label)
accuracy = accuracy_score(test_label, pred_label)
precision = precision_score(test_label, pred_label, average='macro', zero_division=0)
recall = recall_score(test_label, pred_label, average='macro')

print(f'Accuracy: {accuracy:.{3}f}')
print(f'Precision: {precision:.{3}f}')
print(f'Recall: {recall:.{3}f}')

#################################
plt.show()