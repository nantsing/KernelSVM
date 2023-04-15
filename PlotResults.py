import numpy as np
from matplotlib import pyplot as plt

def plot_bar(x_list, y_list, xlabel, name, SavePath):
    ylabel = 'Test Accuracy'
    plt.figure()
    plt.bar(x_list, y_list, alpha = 0.6)
    plt.grid(True, linestyle=':', color='r', alpha=0.6)
    
    for a,b in zip(x_list, y_list):
        plt.text(a, b, b, ha='center', va='bottom', fontsize=10);
    
    plt.ylim(0.6, 0.9)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(name)
    plt.savefig(SavePath)

if __name__ == '__main__':
########################### Linear SVM #################################
    x_list = ['0.0001', '0.001', '0.01', '0.1', '1.0', '10.0']
    y_list = [0.828, 0.8455, 0.848, 0.843, 0.846, 0.846]
    plot_bar(x_list, y_list, 'C', 'Different C in Linear SVM', './fig/LinearSVM.png')
    
########################### RBF kernel SVM #############################
################################# C ####################################
    x_list = ['0.5', '1.0', '5.0', '10.0']
    y_list = [0.8705, 0.8755, 0.8875, 0.874]
    plot_bar(x_list, y_list, 'C', 'Different C in RBF kernel SVM (gamma = 0.002)', './fig/RBFSVMC.png')
############################### gamma ##################################
    x_list = ['0.0001', '0.002', 'scale', 'auto', ]
    y_list = [0.849, 0.8875, 0.8825, 0.8825]
    plot_bar(x_list, y_list, 'gamma', 'Different gamma in RBF kernel SVM (C = 5)', './fig/RBFSVMgamma.png')
    
####################### Polynomial kernel SVM ###########################
    x_list = ['1', '2', '3', '4']
    y_list = [0.8445, 0.876, 0.885, 0.8725]
    plot_bar(x_list, y_list, 'degree', 'Different degree in Poly kernel SVM \n (C = 0.5, gamma = scale, coef0 = 1.0)', './fig/PolySVMCdegree.png')
    
    x_list = ['0.0', '1.0', '5.0']
    y_list = [0.6665, 0.88, 0.8625]
    plot_bar(x_list, y_list, 'coef0', 'Different coef0 in Poly kernel SVM \n (C = 0.1, gamma = scale, degree = 4)', './fig/PolySVMCcoef0.png')
    