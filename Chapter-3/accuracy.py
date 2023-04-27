def accuracy(y_true, y_pred):
    correct_counter = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == yp:
            correct_counter += 1
    return correct_counter / len(y_true)

def true_positive(y_true, y_pred):
    tp = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 1 and yp == 1:
            tp += 1
    return tp

def true_negative(y_true, y_pred):
    tn = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 0 and yp == 0:
            tn += 1
    return tn

def false_positive(y_true, y_pred):
    fp = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 0 and yp == 1:
            fp += 1
    return fp

def false_negative(y_true, y_pred):
    fn = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 1 and yp == 0:
            fn += 1
    return fn

def accuracy_v2(y_true, y_pred):
    tp = true_positive(y_true, y_pred)
    tn = true_negative(y_true, y_pred)
    fp = false_positive(y_true, y_pred)
    fn = false_negative(y_true, y_pred)
    accuracy_score = (tp + tn) / (tp + tn + fp + fn)
    return accuracy_score

def precision(y_true, y_pred):
    tp = true_positive(y_true, y_pred)
    fp = false_positive(y_true, y_pred)
    precision_score = tp / (tp + fp)
    return precision_score

def recall(y_true, y_pred):
    tp = true_positive(y_true, y_pred)
    fn = false_negative(y_true, y_pred)
    recall_score = tp / (tp + fn)
    return recall_score

def f1(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    f1_score = 2 * p * r / (p + r)
    return f1_score

def tpr(y_true, y_pred):
    return recall(y_true, y_pred)

def fpr(y_true, y_pred):
    fp = false_positive(y_true, y_pred)
    tn = true_negative(y_true, y_pred)
    return fp / (fp + tn)

import numpy as np

def log_loss(y_true, y_proba):
    epsilon = 1e-15
    loss = []
    for yt, yp in zip(y_true, y_proba):
        yp = np.clip(yp, epsilon, 1 - epsilon)
        temp_loss = -1.0 * (yt * np.log(yp) + (1 - yt) * np.log(1 - yp))
        loss.append(temp_loss)
    return np.mean(loss)

def macro_precision(y_true, y_pred):
    num_classes = len(np.unique(y_true))

    precision = 0

    for class_ in range(num_classes):
        temp_true = [1 if p == class_ else 0 for p in y_true]
        temp_pred = [1 if p == class_ else 0 for p in y_pred]
        
        tp = true_positive(temp_true, temp_pred)
        fp = false_positive(temp_true, temp_pred)

        temp_precision = tp / (tp + fp)
        precision += temp_precision
    
    precision /= num_classes
    return precision

def micro_precision(y_true, y_pred):
    num_classes = len(np.unique(y_true))

    tp = 0
    fp = 0

    for class_ in range(num_classes):
        temp_true = [1 if p == class_ else 0 for p in y_true]
        temp_pred = [1 if p == class_ else 0 for p in y_pred]
        
        tp += true_positive(temp_true, temp_pred)
        fp += false_positive(temp_true, temp_pred)

    precision = tp / (tp + fp)
    return precision

from collections import Counter

def weighted_precision(y_true, y_pred):
    num_classes = len(np.unique(y_true))

    class_counts = Counter(y_true)

    precision = 0

    for class_ in range(num_classes):
        temp_true = [1 if p == class_ else 0 for p in y_true]
        temp_pred = [1 if p == class_ else 0 for p in y_pred]
        
        tp = true_positive(temp_true, temp_pred)
        fp = false_positive(temp_true, temp_pred)

        temp_precision = tp / (tp + fp)
        weighted_precision = class_counts[class_] * temp_precision
        precision += weighted_precision

    overall_precision = precision / len(y_true)
    return overall_precision

def weighted_f1(y_true, y_pred):
    num_classes = len(np.unique(y_true))

    class_counts = Counter(y_true)

    f1 = 0

    for class_ in range(num_classes):
        temp_true = [1 if p == class_ else 0 for p in y_true]
        temp_pred = [1 if p == class_ else 0 for p in y_pred]
        
        p = precision(temp_true, temp_pred)
        r = recall(temp_true, temp_pred)

        if p + r != 0:
            temp_f1 = 2 * p * r / (p + r)
        else:
            temp_f1 = 0
        
        weighted_f1 = class_counts[class_] * temp_f1
        f1 += weighted_f1

    overall_f1 = f1 / len(y_true)
    return overall_f1

def pk(y_true, y_pred, k):
    if k == 0:
        return 0
    
    y_pred = y_pred[:k]

    pred_set = set(y_pred)

    true_set = set(y_true)

    common_values = pred_set.intersection(true_set)

    return len(common_values) / len(y_pred[:k])

def apk(y_true, y_pred, k):
    pk_values = []

    for i in range(1, k + 1):
        pk_values.append(pk(y_true, y_pred, i))
    
    if len(pk_values) == 0:
        return 0
    
    return sum(pk_values) / len(pk_values)

def mapk(y_true, y_pred, k):
    apk_values = []

    for i in range(len(y_true)):
        apk_values.append(apk(y_true[i], y_pred[i], k))
    
    return sum(apk_values) / len(apk_values)

def mcc(y_true, y_pred):
    tp = true_positive(y_true, y_pred)
    tn = true_negative(y_true, y_pred)
    fp = false_positive(y_true, y_pred)
    fn = false_negative(y_true, y_pred)
    numerator = (tp * tn) - (fp * fn)
    denominator = (
        (tp + fp) *
        (fn + tn) *
        (fp + tn) *
        (tp + fn)
    )
    denominator = denominator ** 0.5
    return numerator / denominator