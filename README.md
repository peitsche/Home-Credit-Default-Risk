# Home-Credit-Default-Risk
Kaggle challenge: [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk). 

## Description
Many people struggle to get loans due to insufficient or non-existent credit histories. And, unfortunately, this population is often taken advantage of by untrustworthy lenders.

Home Credit strives to broaden financial inclusion for the unbanked population by providing a positive and safe borrowing experience. In order to make sure this underserved population has a positive loan experience, Home Credit makes use of a variety of alternative data--including telco and transactional information--to predict their clients' repayment abilities.

While Home Credit is currently using various statistical and machine learning methods to make these predictions, they're challenging Kagglers to help them unlock the full potential of their data. Doing so will ensure that clients capable of repayment are not rejected and that loans are given with a principal, maturity, and repayment calendar that will empower their clients to be successful.

## Evaluation
Submissions are evaluated on area under the ROC curve between the predicted probability and the observed target.

## Approach
We have used both a deep neural network (Keras) and LightGBM to predict to predict the default risk. Our final public score was 0.79221, while the winning score achieved 0.80570. One of the main challenges was feature aggreagtion and engineering based (in total) seven different datasets.   
