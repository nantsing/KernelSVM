import os
import joblib
import numpy as np
from PIL import Image
from dataset import get_data,get_HOG,standardize

from matplotlib import pyplot as plt
from sklearn.svm import SVC

def SaveFig(Array, path):
    X = Image.fromarray(Array)
    X.save(path)
    
def _SaveModel(model, path):
    joblib.dump(model, path)
    
def LoadModel(path):
    model = joblib.load(path)
    return model

def test_Accuracy(SVM, H_test, Y_test):
    Y_predict = SVM.predict(H_test)
    CorrectNum = (Y_test == Y_predict).sum()
    return CorrectNum / len(Y_test)
    
def trainAndtest_SVM(H_train, Y_train, H_test, Y_test, C = 1.0, kernel = 'rbf', degree = 3, gamma = 'scale', coef0 = 0.0, \
    shrinking = True, tol = 1e-3, max_iter = -1, decision_function_shape = 'ovr', random_state = None, is_save = True):
    
    parameters = ['C', 'kernel', 'degree', 'gamma', 'coef0', 'shirinking', 'tol', 'max_iter', 'decision_function_shape', \
        'random_state', 'testAccuracy']
    values = [str(C), str(kernel), str(degree), str(gamma), str(coef0), str(shrinking), str(tol), str(max_iter), \
        str(decision_function_shape), str(random_state)]
    
    is_load = False
    with open('./models/meta.txt', 'r') as file:
        contents = file.read().split('\n')
        count = int(contents[0].split(' ')[1])
        for i in range(1, count + 1):
            content = contents[i].split(' ')[1:20:2]
            # print(content)
            # print(values)
            if content == values:
                is_load = True
                print(i)
                SVM = LoadModel(f'./models/{i}.pkl')
                break
            
    if not is_load:
        SVM = SVC(C= C, kernel= kernel, degree= degree, gamma= gamma, coef0= coef0, shrinking= shrinking, \
            tol= tol, max_iter= max_iter, decision_function_shape= decision_function_shape, random_state= random_state)
        SVM.fit(H_train, Y_train)
    
    testAccuracy = test_Accuracy(SVM, H_test, Y_test)
    values.append(str(testAccuracy))
    
    if is_save and not is_load:
        with open('./models/meta.txt', 'a') as file:
            for p, v in zip(parameters, values):
                file.writelines(f'{p}: {v} ')
            file.write('\n')
        
        # print(count)
        ## Change count
        with open('./models/meta.txt',mode='r') as file:
            data = file.read()
            data = data.replace(f'Count: {count}', f'Count: {count + 1}')

        with open('./models/meta.txt',mode='w') as file:
            file.write(data)

    _SaveModel(SVM, f'./models/{count + 1}.pkl')
    print(f'Test Accuracy: {testAccuracy}')
    
    return SVM

if __name__ == '__main__':
######################## Get train/test dataset ########################
    X_train,X_test,Y_train,Y_test = get_data('dataset')
########################## Get HoG featues #############################
    H_train,H_test = get_HOG(X_train), get_HOG(X_test)
######################## standardize the HoG features ####################
    H_train,H_test = standardize(H_train), standardize(H_test)
########################################################################
######################## Implement you code here #######################
########################################################################

########################### Linear SVM #################################
    for kernel in ['linear', 'poly']   : 
        for C in [0.1, 0.5, 1.0, 1.5, 2.0, 5.0, 10.0]:
            for gamma in ['scale', 'auto']:
                if kernel == 'linear': gamma = 'scale'
                for coef0 in [0.0, 1.0]:
                    if kernel == 'linear': coef0 = 0.0
                    for shrinking in [True, False]:
                        LinearSVM = trainAndtest_SVM(H_train, Y_train, H_test, Y_test, kernel = kernel, C = C, \
                            gamma = gamma, coef0 = coef0, shrinking = shrinking, decision_function_shape = 'ovr')

########################### RBF kernel SVM #############################




####################### polynomial kernel SVM ###########################