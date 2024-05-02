import math 
import pandas as pd

def accuracy(correct_pred, total_number): 
    return correct_pred/total_number


def confusionMatrix(tp, tn, fp, fn): 
    mat = pd.DataFrame([[tp, fp], [fn, tn]], index=['Positive', 'Negative'], columns=['Positive', 'Negative'])
    print(mat)

def precision(tp, fp): 
    return tp/(tp+fp)

def recall(tp, fn): 
    return tp/(tp+fn)

def f1Scores(precision, recall):
    return 2*((precision*recall)/(precision+recall))




