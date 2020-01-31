##############################################################################
# CO395: Introduction to Machine Learning
# Coursework 1 Skeleton code
# Prepared by: Josiah Wang
#
# Your tasks: 
# Complete the following methods of Evaluator:
# - confusion_matrix()
# - accuracy()
# - precision()
# - recall()
# - f1_score()
##############################################################################

import numpy as np


class Evaluator(object):
    """ Class to perform evaluation
    """
    
    def confusion_matrix(self, prediction, annotation, class_labels=None):
        """ Computes the confusion matrix.
        
        Parameters
        ----------
        prediction : np.array
            an N dimensional numpy array containing the predicted
            class labels
        annotation : np.array
            an N dimensional numpy array containing the ground truth
            class labels
        class_labels : np.array
            a C dimensional numpy array containing the ordered set of class
            labels. If not provided, defaults to all unique values in
            annotation.
        
        Returns
        -------
        np.array
            a C by C matrix, where C is the number of classes.
            Classes should be ordered by class_labels.
            Rows are ground truth per class, columns are predictions.
        """
        if class_labels is None:
#         if not class_labels:
            class_labels = np.unique(annotation)
        
        confusion = np.zeros((len(class_labels), len(class_labels)), dtype=np.int)
        
        
        #######################################################################
        #                 ** TASK 3.1: COMPLETE THIS METHOD **
        #######################################################################
        char_to_int = dict((c, i) for i, c in enumerate(class_labels))
    
        pred = []
        anno = []

        for i in range(len(prediction)):
            pred.append([char_to_int[char] for char in prediction[i]][0])
            anno.append([char_to_int[char] for char in annotation[i]][0])
    
        for a, p in zip(anno, pred):
            confusion[a][p] += 1
        
        return confusion
    
    
    def accuracy(self, confusion):
        """ Computes the accuracy given a confusion matrix.
        
        Parameters
        ----------
        confusion : np.array
            The confusion matrix (C by C, where C is the number of classes).
            Rows are ground truth per class, columns are predictions
        
        Returns
        -------
        float
            The accuracy (between 0.0 to 1.0 inclusive)
        """
        
        # feel free to remove this
        total = 0
        corr_class = 0
        for i in range(len(confusion)):
            corr_class = corr_class + confusion[i][i]
            for j in range(len(confusion)):
                total = total + confusion[i][j]
            
        accuracy = round(corr_class/total, 3)
        
        #######################################################################
        #                 ** TASK 3.2: COMPLETE THIS METHOD **
        #######################################################################
        
        return accuracy
        
    
    def precision(self, confusion):
        """ Computes the precision score per class given a confusion matrix.
        
        Also returns the macro-averaged precision across classes.
        
        Parameters
        ----------
        confusion : np.array
            The confusion matrix (C by C, where C is the number of classes).
            Rows are ground truth per class, columns are predictions.
        
        Returns
        -------
        np.array
            A C-dimensional numpy array, with the precision score for each
            class in the same order as given in the confusion matrix.
        float
            The macro-averaged precision score across C classes.   
        """
        
        # Initialise array to store precision for C classes
        p = np.zeros((len(confusion), ))
        
        #######################################################################
        #                 ** TASK 3.3: COMPLETE THIS METHOD **
        #######################################################################

        t_pos = 0
        tf_pos = 0
        for i in range(len(confusion)):
            t_pos = confusion[i][i]
            for j in range(len(confusion)):
                tf_pos = tf_pos + confusion[j][i]
            p[i] = t_pos/tf_pos
            tf_pos = 0
        
        
        # You will also need to change this
        total = 0
        for i in range(len(p)):
            total = total + p[i]
        macro_p = total/len(p)

        return (p, macro_p)
    
    
    def recall(self, confusion):
        """ Computes the recall score per class given a confusion matrix.
        
        Also returns the macro-averaged recall across classes.
        
        Parameters
        ----------
        confusion : np.array
            The confusion matrix (C by C, where C is the number of classes).
            Rows are ground truth per class, columns are predictions.
        
        Returns
        -------
        np.array
            A C-dimensional numpy array, with the recall score for each
            class in the same order as given in the confusion matrix.
        
        float
            The macro-averaged recall score across C classes.   
        """
        
        # Initialise array to store recall for C classes
        r = np.zeros((len(confusion), ))
        
        #######################################################################
        #                 ** TASK 3.4: COMPLETE THIS METHOD **
        #######################################################################
        
        t_pos = 0
        tp_fn = 0
        for i in range(len(confusion)):
            t_pos = confusion[i][i]
            for j in range(len(confusion)):
                tp_fn = tp_fn + confusion[i][j]
            r[i] = t_pos/tp_fn
            tp_fn = 0
        
        # You will also need to change this 
        total = 0
        for i in range(len(r)):
            total = total + r[i]
        macro_r = total/len(r)
        
        return (r, macro_r)
    
    
    def f1_score(self, confusion):
        """ Computes the f1 score per class given a confusion matrix.
        
        Also returns the macro-averaged f1-score across classes.
        
        Parameters
        ----------
        confusion : np.array
            The confusion matrix (C by C, where C is the number of classes).
            Rows are ground truth per class, columns are predictions.
        
        Returns
        -------
        np.array
            A C-dimensional numpy array, with the f1 score for each
            class in the same order as given in the confusion matrix.
        
        float
            The macro-averaged f1 score across C classes.   
        """
        
        # Initialise array to store recall for C classes
        f = np.zeros((len(confusion), ))
        
        #######################################################################
        #                 ** YOUR TASK: COMPLETE THIS METHOD **
        #######################################################################
        
        p, macro = precision(confusion)
        r, macro = recall(confusion)
        
        for i in range(len(confusion)): 
            f[i] = (2 * (p[i] * r[i]) / (p[i] + r[i]))
        
        # You will also need to change this    
        total = 0
        for i in range(len(f)):
            total = total + f[i]
        macro_f = total/len(f)
        
        return (f, macro_f)
   
 
