# 美股財報分析及預測
US Stock Market Analysis and Prediction

## Authors
蕭皓澤（資工二）、蔡明龍（資工二）、鄭益丞(財金三)、張念詠(財金三)、莫家希(財金三)

## Abstract 
This study explored the feasibility of predicting corporate growth or decline using natural language processing techniques applied to annual financial reports. 
Employing a dataset of 342 U.S. listed companies, we conducted experiments using a combination of traditional machine learning models (Support Vector Machines, Naive Bayes) and state-of-the-art deep learning models 
(Recurrent Neural Networks, BERT). To evaluate model performance from an investment perspective, we introduced a novel metric: the non-fail rate, which measures the accuracy of predicting either growth or decline.

Our findings indicate that ensemble methods, combining multiple well-performing models with dynamically adjusted weights, significantly improved prediction accuracy, achieving a non-fail rate of approximately 80%. 
However, the relatively small dataset size and the simplicity of the classification task limit the generalizability of our findings. Future research can address these limitations by expanding the dataset, 
exploring more granular classification schemes, and incorporating additional textual features and external factors.

## Raw Data 
https://github.com/AAAA-source/US-Stock-Market-Analysis-and-Prediction/blob/main/%E7%BE%8E%E8%82%A1%E5%85%AC%E5%8F%B8%E8%B2%A1%E5%A0%B1%E6%95%98%E8%BF%B0%26%E8%B2%A1%E5%8B%99%E6%8C%87%E6%A8%99.csv

## Paper and Reports 
https://github.com/AAAA-source/US-Stock-Market-Analysis-and-Prediction/blob/main/%E7%BE%8E%E8%82%A1%E5%88%86%E6%9E%90%E6%9C%9F%E6%9C%AB%E5%A0%B1%E5%91%8A.pdf
https://github.com/AAAA-source/US-Stock-Market-Analysis-and-Prediction/blob/main/%E7%BE%8E%E8%82%A1%E5%88%86%E6%9E%90slide.pdf

## Using Models 
### Single Model
- Naïve Bayes : Multinomial NB , Gaussian NB , Complement NB , Bernoulli NB , NBs with LDA , 
- SVM 
- Rocchio
- K-means
- KNN , KNN with SVD
- Neural Network , Simple RNN
### Merge Models
SVM + Bernoulli NB , SVM + NB
