# 2022 fall SNU NLP project

### Member
2019-13773 Kyungjin Kim  ([@jamesk617](https://github.com/jamesk617))  
2018-18574 Junyoung Park ([@engineerA314](https://github.com/engineerA314))  
2018-12018 Sungmin Song  ([@reverg](https://github.com/reverg))  

### Abstract
We worked on promotor gene classifying task by using machinelearning method. CNN+GRU, CNN, GRU, windowGRU, DNABERT+1-linear, DNABERT+2-linear models are implemented. GRU models showed lowest accuracy and BERT based model showed meaningful increment.

### Motivation
As natural language process technology has shown remarkable progress, now it is applied in not only the human language, but also for many other sequence data such as DNA. DNA is the key-element of the Central Dogma principle in biology and it is also a core for many medical science task like immune profiling.

One of important DNA target is promotor gene. Promotor gene is area that RNA transcription factors gets bonded into, so it is necessary to reveal the pattern of it for DNA study. As so, it is representative example for DNA-NLP task.

Therefore we chose promotor gene classifying task as our project subject. We believe that these kind of experiences can help us to understand more deeply in how machine learning on sequence datas are progressed, and to be more flexible researchers so that we can engage into various tasks in the world. 
  
  
### Idea
Baseline model of our project’s reference was CNN+GRU. We implemented CNN+GRU model and CNN only, GRU only model and BERT for comparison. Plus, we also used few variations for each models for broad practicing. 

Our first goal was to improve the accuracy of the task by finding other appropriate model and second goal was to practice various machine learning method by implementing codes.

### Experiments & Result

#### 1. Baseline (CNN + GRU)
How to run  
  1) Upload the elements in 'CNN+GRU' directory to Google Drive.  
  2) Run 'CNN+GRU.ipynb' with GPU mode in Colab  
  
Result
  1) Loss graph per epoch    
  ![CNN+GRU loss](https://user-images.githubusercontent.com/86403521/207677900-3f8455cb-fb55-4d6c-b4a1-56924e750698.png)
    
  2) Accuracy graph per epoch  
  ![CNN+GRU accuracy](https://user-images.githubusercontent.com/86403521/207678004-33de792e-58ca-462f-b8c2-0079653754bd.png)  
    
  3) Result for Test Set   
  
|Accuracy|Precision|Recall|  
|:---:|:---:|:---:|  
|0.876|0.864|0.893|
  
  
#### 2. only CNN
How to run  
  1) Run 'onlyCNN.ipynb'
  
Result
  1) Loss graph per epoch    
  ![only CNN loss](https://user-images.githubusercontent.com/107937875/207784872-86446e65-b872-4624-a2f0-bbddab2edecb.png)
    
  2) Accuracy graph per epoch  
  ![only CNN accuracy](https://user-images.githubusercontent.com/107937875/207784883-2732a530-dfd6-43f7-904b-bac7c459cdf6.png)

  3) Result for Test Set   
  
|Accuracy|Precision|Recall|  
|:---:|:---:|:---:|  
|0.871|0.896|0.839|
  
  

#### 3. only GRU
How to run  
  1) Upload the elements in 'onlyGRU' directory to Google Drive.  
  2) Run 'only_GRU_experiment.ipynb' with GPU mode in Colab  
  
Result
  1) Loss graph per epoch    
  ![only GRU loss](https://user-images.githubusercontent.com/86403521/207683388-e1a2d69e-3ac1-4baa-8208-e13862b55274.png)

    
  2) Accuracy graph per epoch  
  ![only GRU accuracy](https://user-images.githubusercontent.com/86403521/207683400-793d03ce-19c0-43d3-8a02-256e91611fca.png)

    
  3) Result for Test Set   
  
|Accuracy|Precision|Recall|  
|:---:|:---:|:---:|  
|0.824|0.793|0.876|
  
  
#### 4. window GRU
How to run  
  1) Upload the elements in 'windowGRU' directory to Google Drive.  
  2) Run 'window+GRU_experiment.ipynb' with GPU mode in Colab  
  
Result
  1) Loss graph per epoch    
  ![window GRU loss](https://user-images.githubusercontent.com/86403521/207683345-bde340b4-1f4a-4e37-a9f3-c9c9fbb9730b.png)

    
  2) Accuracy graph per epoch  
  ![window GRU accuracy](https://user-images.githubusercontent.com/86403521/207683329-4b73a419-100b-43b1-b99f-7f630f8c0694.png)

    
  3) Result for Test Set   
  
|Accuracy|Precision|Recall|  
|:---:|:---:|:---:|  
|0.731|0.716|0.765|
  
  
#### 5. DNABERT + 1 Linear layer
How to run  
  1) Upload the elements in 'DNABERT+1' directory to Google Drive.  
  2) Run 'DNABERT+1_layer.ipynb' with GPU mode in Colab  
  
Result
  1) Loss graph per epoch    
  ![dnabert+1classifier loss](https://user-images.githubusercontent.com/86403521/207683119-4f4800e5-a310-4260-a679-058279c98f7d.png)

  2) Accuracy graph per epoch  
  ![dnabert+1classifier metric](https://user-images.githubusercontent.com/86403521/207683219-46c1d4d3-275e-4611-a7b8-5964d5fe14ea.png)

  3) Result for Test Set   
  
|Accuracy|Precision|Recall|  
|:---:|:---:|:---:|  
|0.886|0.841|0.952|
  
  

#### 6. DNABERT + 2 Linear layer
How to run  
  1) Upload the elements in 'DNABERT+2' directory to Google Drive.  
  2) Run 'DNABERT_torch_2.ipynb' with GPU mode in Colab  
  
Result
  1) Loss graph per epoch    
  ![dnabert+2 loss](https://user-images.githubusercontent.com/48681924/207695532-71b4fd88-94ab-46e9-a5ea-f3dd37022569.png)

    
  2) Accuracy graph per epoch  
  ![dnabert+2 accuracy](https://user-images.githubusercontent.com/48681924/207695413-550fe549-15a1-48cd-a77d-4bd391a92a47.png)
    
  3) Result for Test Set   
  
|Accuracy|Precision|Recall|  
|:---:|:---:|:---:|  
|0.901|0.922|0.876|
  
  



### Conclusion

