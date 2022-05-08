# Neural Network Charity Analysis

## Overview

The purpose of this analysis is to build a model that will improve investment outcomes by predicting which investments will be impactful based on past experience. To make this determination we will be analyzing the data using neural networks.

## Results

### Data Preprocessing

* The target variable is the "IS_SUCCESSFUL" (1 or 0)
* Features for the model include:
  * APPLICATION_TYPE
  * AFFILIATION
  * CLASSIFICATION
  * USE_CASE
  * ORGANIZATION
  * STATUS
  * INCOME_AMT
  * SPECIAL_CONSIDERATIONS
  * ASK_AMT

  Category features were split and reduced by binning all categories with a value count below the threshold of 500 into a single "Other" category. The final input feature count was 43.
* Other input variables that are neither targets nor features are EID and NAME, which are non-linear and non-categorizational data
* Initially the NAME and EID columns were excluded, but during optimization it was determined that the frequency of applications from the same organization is useful in determining the impact of investments so a new column, frequency, was derived from the value count of the NAME column and added to the input feature set.

### Compiling, Training, and Evaluating the Model

* Initially the model was run with two layers using 64 and 16 neurons respecively, both with the "relu" activation function. These numbers were selected based on the heuristic of choosing the power of 2 nearest to double the input features, and a lower power of 2 for subsequent layers.
* By trial and error I determined that the model worked best with two layers of 128 and 32 neurons respectively, both with the "tanh" activation function. Extending the model to 3 layers did not meaningfully improve performance.
* The target model performance of 75% accuracy was attainable only after adding back information from the NAME/EID columns; specifically, the model was able to find a meaningful correlation between repeated names and the target variable, ultimately producing a model with about 76% accuracy.
  ![accuracy](<./accuracy.png>)
* Other steps to try increasing model performance included:
  * Running for more epochs: the accuracy seemed to stabilize before 100 epochs so additional training cycles did not improve performance
  * Adjusting the bins for categorizational data: observed performance was reduced or unchanged by changing the thresholds
  * Adding more layers: a third layer had no measurable impact on model performance
  * Adding more neurons to layers: there was a minor improvement by adding neurons to intermediate layers, although that might be attributable to random chance
  * Changing the activation function: in testing the results from using the "tanh" function were slightly better than "relu", however this too could be attributable to random chance

## Summary

Overall we were able to achieve the target performance of 75%. Due to the nature of neural networks it is not straightforward to understand how or why specific factors were taken into account when making predictions about future investments.

Before adopting this deep learning model as the determinant for future investment prospects, it would be best to consider whether any other machine learning models (such as Random Forest) would achieve comparable outcomes. The Random Forest classifier is appropriate for tabular data such as our input source and is appropriate for a single boolean output such as our target variable.