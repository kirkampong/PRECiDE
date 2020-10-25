## Logistic Regression
**Output:**   
no_crisis    965  
crisis        94  
Name: banking_crisis, dtype: int64  
(847, 8) (847, 1)   
(212, 8) (212, 1)  
/Users/kirkampong/miniconda3/envs/cs229/lib/python3.6/site-packages/sklearn/utils/validation.py:72: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().    
  return f(**kwargs)   
/Users/kirkampong/miniconda3/envs/cs229/lib/python3.6/site-packages/sklearn/linear_model/_logistic.py:764: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.     
    
Increase the number of iterations (max_iter) or scale the data as shown in:   
    https://scikit-learn.org/stable/modules/preprocessing.html   
Please also refer to the documentation for alternative solver options:    
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression    
  extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG).      
      
                 precision  recall    f1-score   support   
           0       0.80      0.86      0.83        14   
           1       0.99      0.98      0.99       198    

    accuracy                           0.98       212
    macro avg      0.89      0.92      0.91       212 
    weighet avg    0.98      0.98      0.98       212   

