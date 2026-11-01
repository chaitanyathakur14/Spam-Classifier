# Spam Classifier for MLOps Experiment 8
This is a SMS/Email Spam classifier that identifies if a given text message is a potential advert, fraud or scam and seperate it from actual text messages.
## Dataset Used:
The dataset used in this project was fetched from kaggle named:
### SMS Spam Collection Dataset
> Link to dataset: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

# About
## Identification and Description of problem
Big Tech Giants like Google put a spam classifier in their email system to detect whether a recieved email is an important one or a spam by some other company targeted for advertisement. <br>
Whenever a user logs into another site or uses a product using the same account for email then the company pushes promotion with or without consent. <br>
In order to deal with this massive problem, classification and detection is very crucial in order to provide a very good experience to the user and avoid any hassle. <br>

# Process
**We have to breakdown the MLA into following steps:**<br>
1. Data Cleaning
2. EDA
3. Text Preprocessing
4. Model Building
5. Evaluation
6. Improvement
7. Deployment

# Libraries Needed:
```shell
pip install nltk
pip install pandas
pip install sklearn
pip install numpy
pip install streamlit
pip install collection
```
# Conclusion
After every inspection, we can see that 
**Multinomial Naive Bayes** is the best performing algorithm with <br>

accuracy metrics of:<br>
--------------------------- <br>
Accuracy Score: 0.9691 <br>
Conusion Matrix: <br>
[[888   0] <br>
 [ 32 114]] <br>
Precision Score: 1.0 <br>

with hyperparameter of 
> max_features of tfidf set to 3000 <br>
> default parameters of MNB


## To run the website 
> Run the `main.ipynb` file from top to bottom<br>
> enter the following command in the terminal
```shell
streamlit run app.py
```

**Accuracy has been precisely calculated over different scenarios. However, we can further fine tune the model using other ensemble learning methods like VotingClasifier** <br>


*Note that this is merely a prototype and is not optimized*
