import os
import joblib
import numpy as np
from PIL import Image
from dataset import get_data,get_HOG,standardize

from matplotlib import pyplot as plt
from sklearn.svm import SVC

def plot_sv(sv, Alpha, name, SavePath):

    plt.figure()
    for idx, img in enumerate(sv):
        ax = plt.subplot(4, 5, idx + 1)
        ax.axis('off')
        ax.set_title(f'alpha = {round(Alpha[idx], 1)}')
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.8)
        plt.imshow(img, cmap= 'gray')

    plt.suptitle(name)
    plt.savefig(SavePath)
        


def SaveFig(Array, path):
    X = Image.fromarray(Array)
    X.save(path)
    
# Save model and parameter setting to avoid repeated training.
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
    
    ## Determine if this set of parameters has been trained before
    is_load = False
    with open('./models/meta.txt', 'r') as file:
        contents = file.read().split('\n')
        count = int(contents[0].split(' ')[1])
        for i in range(1, count + 1):
            content = contents[i].split(' ')[1:20:2]
            if content == values:
                is_load = True
                print(f'The {i}.pkl has been loaded.')
                SVM = LoadModel(f'./models/{i}.pkl')
                break
            
    ## Train model
    if not is_load:
        SVM = SVC(C= C, kernel= kernel, degree= degree, gamma= gamma, coef0= coef0, shrinking= shrinking, \
            tol= tol, max_iter= max_iter, decision_function_shape= decision_function_shape, random_state= random_state)
        SVM.fit(H_train, Y_train)
    
    ## test model
    testAccuracy = test_Accuracy(SVM, H_test, Y_test)
    values.append(str(testAccuracy))
    
    if is_save and not is_load:
        with open('./models/meta.txt', 'a') as file:
            for p, v in zip(parameters, values):
                file.writelines(f'{p}: {v} ')
            file.write('\n')
        
        ## Change count number
        with open('./models/meta.txt',mode='r') as file:
            data = file.read()
            data = data.replace(f'Count: {count}', f'Count: {count + 1}')

        with open('./models/meta.txt',mode='w') as file:
            file.write(data)

        ## Save model
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
    # for C in [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]:
    #     LinearSVM = trainAndtest_SVM(H_train, Y_train, H_test, Y_test, kernel = 'linear', C = C, \
    #         gamma = 'scale', coef0 = 0.0, shrinking = True, decision_function_shape = 'ovr')
    

    #### Count and plot support vectors
    LinearSVM = trainAndtest_SVM(H_train, Y_train, H_test, Y_test, kernel = 'linear', C = 0.01, \
            gamma = 'scale', coef0 = 0.0, shrinking = True, decision_function_shape = 'ovr')
    
    Indices = LinearSVM.support_
    support_vectors = LinearSVM.support_vectors_
    n_support = LinearSVM.n_support_
    Alpha = LinearSVM.dual_coef_[0]
    
    Indices_neg = []
    Indices_pos = []
    SV_neg = []
    SV_pos = []
    for idx, i in enumerate(Indices):
        if Y_train[i] == 0: 
            Indices_neg.append(idx)
            SV_neg.append(X_train[i])
        else: 
            Indices_pos.append(idx)
            SV_pos.append(X_train[i])
    Indices_neg = np.array(Indices_neg)
    Indices_pos = np.array(Indices_pos)
    SV_neg = np.array(SV_neg)
    SV_pos = np.array(SV_pos)

    print(f'Total support vectors number: {len(support_vectors)}')
    print(f'Positive support vectors number: {len(SV_neg)}')
    print(f'Negtive support vectors number: {len(SV_pos)}')


    alpha_neg = np.abs(Alpha[Indices_neg])
    alpha_pos = Alpha[Indices_pos]

    sorted_i_neg = np.argsort(alpha_neg)[::-1]
    sorted_i_pos = np.argsort(alpha_pos)[::-1]

    SV_neg = SV_neg[sorted_i_neg]
    alpha_neg = alpha_neg[sorted_i_neg]
    SV_pos = SV_pos[sorted_i_pos]
    alpha_pos = alpha_pos[sorted_i_pos]

    plot_sv(SV_neg[:20], alpha_neg[:20], 'Negative Support Vectors (C = 0.01)', './fig/SVnegative.png')
    plot_sv(SV_pos[:20], alpha_pos[:20], 'Positive Support Vectors (C = 0.01)', './fig/SVpositive.png')

    # LinearSVM = trainAndtest_SVM(H_train, Y_train, H_test, Y_test, kernel = 'linear', C = 100.0, \
    #         gamma = 'scale', coef0 = 0.0, shrinking = True, decision_function_shape = 'ovr', is_save = False)


########################### RBF kernel SVM #############################
#     for C in [0.5, 1.0, 5.0, 10.0]:
#         RBFSVM = trainAndtest_SVM(H_train, Y_train, H_test, Y_test, kernel = 'rbf', C = C, \
#             gamma = 0.002, coef0 = 0.0, shrinking = True, decision_function_shape = 'ovr')
    
#     print(f'n_features: {RBFSVM.n_features_in_}')
#     print(f'X.var: {H_train.var()}')
    
#     for gamma in [0.0001, 0.002, 'scale','auto']:
#         RBFSVM = trainAndtest_SVM(H_train, Y_train, H_test, Y_test, kernel = 'rbf', C = 5.0, \
#             gamma = gamma, coef0 = 0.0, shrinking = True, decision_function_shape = 'ovr')

# ####################### Polynomial kernel SVM ###########################
#     for degree in [1, 2, 3, 4]:
#         PolySVM = trainAndtest_SVM(H_train, Y_train, H_test, Y_test, kernel = 'poly', C = 0.5, \
#                     degree = degree, gamma = 'scale', coef0 = 1.0, shrinking = True, decision_function_shape = 'ovr')
        
#     for coef0 in [0.0, 1.0, 5.0]:
#         PolySVM = trainAndtest_SVM(H_train, Y_train, H_test, Y_test, kernel = 'poly', C = 0.1, \
#         degree = 4, gamma = 'scale', coef0 = coef0, shrinking = True, decision_function_shape = 'ovr')