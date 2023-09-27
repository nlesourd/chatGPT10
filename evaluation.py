from typing import List


def get_confusion_matrix(
    actual: List[int], predicted: List[int]
) -> List[List[int]]:
    """Computes confusion matrix from lists of actual or predicted labels.

    Args:
        actual: List of integers (0 or 1) representing the actual classes of
            some instances.
        predicted: List of integers (0 or 1) representing the predicted classes
            of the corresponding instances.

    Returns:
        List of two lists of length 2 each, representing the confusion matrix.
    """
    FN = 0
    FP = 0
    # Calculation of the false positives and negatives
    for i in range(len(actual)):
        difference = actual[i]-predicted[i]
        if difference == 1:
            FN += 1
        if difference == -1:
            FP += 1

    # Calculation of the true positives and negatives
    nb_1 = actual.count(1)
    TP = nb_1 - FN
    nb_0 = len(actual) - nb_1
    TN = nb_0 - FP

    # Creation of the confusion matrix
    confusion_mat = [[TN, FP], [FN, TP]]
    return confusion_mat


def accuracy(actual: List[int], predicted: List[int]) -> float:
    """Computes the accuracy from lists of actual or predicted labels.

    Args:
        actual: List of integers (0 or 1) representing the actual classes of
            some instances.
        predicted: List of integers (0 or 1) representing the predicted classes
            of the corresponding instances.

    Returns:
        Accuracy as a float.
    """
    # Recuperation of information
    confusion_mat = get_confusion_matrix(actual, predicted)
    TN = confusion_mat[0][0]
    FP = confusion_mat[0][1]
    FN = confusion_mat[1][0]
    TP = confusion_mat[1][1]

    # Calculation
    return (TP+TN)/(TP+FP+FN+TN)


def precision(actual: List[int], predicted: List[int]) -> float:
    """Computes the precision from lists of actual or predicted labels.

    Args:
        actual: List of integers (0 or 1) representing the actual classes of
            some instances.
        predicted: List of integers (0 or 1) representing the predicted classes
            of the corresponding instances.

    Returns:
        Precision as a float.
    """
    # Recuperation of information
    confusion_mat = get_confusion_matrix(actual, predicted)
    FP = confusion_mat[0][1]
    TP = confusion_mat[1][1]

    # Calculation
    return TP/(TP+FP)


def recall(actual: List[int], predicted: List[int]) -> float:
    """Computes the recall from lists of actual or predicted labels.

    Args:
        actual: List of integers (0 or 1) representing the actual classes of
            some instances.
        predicted: List of integers (0 or 1) representing the predicted classes
            of the corresponding instances.

    Returns:
        Recall as a float.
    """
    # Recuperation of information
    confusion_mat = get_confusion_matrix(actual, predicted)
    FN = confusion_mat[1][0]
    TP = confusion_mat[1][1]

    # Calculation
    return TP/(TP+FN)


def f1(actual: List[int], predicted: List[int]) -> float:
    """Computes the F1-score from lists of actual or predicted labels.

    Args:
        actual: List of integers (0 or 1) representing the actual classes of
            some instances.
        predicted: List of integers (0 or 1) representing the predicted classes
            of the corresponding instances.

    Returns:
        float of harmonic mean of precision and recall.
    """
    # Calculation
    prec = precision(actual, predicted)
    rec = recall(actual, predicted)
    return 2*prec*rec/(prec+rec)


def false_positive_rate(actual: List[int], predicted: List[int]) -> float:
    """Computes the false positive rate from lists of actual or predicted
        labels.

    Args:
        actual: List of integers (0 or 1) representing the actual classes of
            some instances.
        predicted: List of integers (0 or 1) representing the predicted classes
            of the corresponding instances.

    Returns:
        float of number of instances incorrectly classified as positive divided
            by number of actually negative instances.
    """
    # Recuperation of information
    confusion_mat = get_confusion_matrix(actual, predicted)
    TN = confusion_mat[0][0]
    FP = confusion_mat[0][1]

    # Calculation
    return FP/(TN+FP)


def false_negative_rate(actual: List[int], predicted: List[int]) -> float:
    """Computes the false negative rate from lists of actual or predicted
        labels.

    Args:
        actual: List of integers (0 or 1) representing the actual classes of
            some instances.
        predicted: List of integers (0 or 1) representing the predicted classes
            of the corresponding instances.

    Returns:
        float of number of instances incorrectly classified as negative divided
            by number of actually positive instances.
    """
    # Recuperation of information
    confusion_mat = get_confusion_matrix(actual, predicted)
    FN = confusion_mat[1][0]
    TP = confusion_mat[1][1]

    # Calculation
    return FN/(FN+TP)
