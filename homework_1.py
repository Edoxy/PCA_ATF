import numpy as np
import matplotlib.pylab as plt
import scipy as sc
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score
from PCA import PCA
import xlsxwriter as exel

'''
Homework 1 Analisi Tempo Frequenza
Created on April 2023

author: Edoardo Vay and Alessandro Rossi
'''

img_set = np.zeros((64, 64, 10))
x = plt.imread(f"archive/s1/1.pgm")

# print(type(x))
H = x.shape[0] # height
W = x.shape[1] # width
print('Images Dimensions: ', H, W)


wb = exel.Workbook("Eigenfaces.xlsx")
ws = wb.add_worksheet()
ws.write("A1", 'Numero Immagini per Test')
ws.write("B1", 'Numero di Componenti PCA')
ws.write('C1', 'Accuratezza')
ws.write('D1', 'Precisione')
ws.write('E1', 'Richiamo')

k = 2

for n_test in [1, 2, 3, 4, 5]:
    for components in [20, 40, 80, 160]:
        train = []
        train_label = []

        test = []
        test_label = []

        n_people = 0
        '''Number of people not used in training'''

        # print('Number of images used for testing for each person: ', n_test)


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

        # print('Number of component used on PCA: ', components)

        pca = PCA(n_components=components)
        pca.fit(train, svd_decomposition=False)
        pca_train = pca.transform(train)
        # print(pca_train.shape)
        img = pca.inverse_transform(pca_train)
        # print(img.shape)

        error_train = np.zeros((train.shape[0]))

        for i in range(train.shape[0]):
            error_train[i] = np.linalg.norm(img[i] - train[i])

        soglia = 2775 # np.amax(error_train) 

        # print(f'Soglia stimata : {soglia:.{2}f}')
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

        error_test = np.zeros((test.shape[0]))

        for i in range(test.shape[0]):
            error_test[i] = np.linalg.norm(projected_test[i] - test[i])

        fig2, ax2 = plt.subplots()
        ax2.hist(error_test, histtype='step', density=True, bins=20)

        # create an array of the predicted label
        pred_label = np.zeros((test.shape[0]), dtype=np.int32)

        # main loop of 
        for i, image in enumerate(pca_test):
            # test if person is already present
            if error_test[i] > soglia:
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

        ws.write('A{}'.format(k), n_test)
        ws.write('B{}'.format(k), components)
        ws.write('C{}'.format(k), round(accuracy, 3))
        ws.write('D{}'.format(k), round(precision, 3))
        ws.write('E{}'.format(k), round(recall, 3))
        k +=1

wb.close()
        # print(f'Accuracy: {accuracy:.{3}f}')
        # print(f'Precision: {precision:.{3}f}')
        # print(f'Recall: {recall:.{3}f}')




#################################
# plt.show()