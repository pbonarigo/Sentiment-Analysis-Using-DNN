# Sentiment-Analysis-Using-DNN
US Airline Twitter-based Sentiment Analysis Using Tensorflow and Keras

Sharing the final project for HKUST MBA course ISOM 5240, titled "Deep Learning Business Applications with Python" developed with [@danieltsaicw](https://github.com/danieltsaicw).

We utilized several key resources to produce a program that could identify key hashtags and analyze the sentiment of the user, identifying the Tweet as either positive, negative, or neutral.  These included using Keras with Python running on Google's Tensorflow, as well as varios methods for building and training a Deep Neural Network (DNN), as well as using Twitter's API functionality to create a working live demo.

## The Dataset & Database

Natural language processing was achieved using TFIDF and the Python package [sklearn](https://pypi.org/project/scikit-learn/)

SQLite for data input and analysis with a dataset obtained from [Kaggle](https://www.kaggle.com/crowdflower/twitter-airline-sentiment) which contains over 14,000 US airline-related Tweets on which our model was trained.  These data points included the following attributes:

- tweet_id 
- airline_sentiment 
- airline_sentiment_confidence 
- negativereasonnegativereason_confidence 
- airline 
- airline_sentiment_gold 
- name 
- negativereason_gold 
- retweet_count 
- text 
- tweet_coord 
- tweet_created 
- tweet_locationuser_timezone 


## Analytics

Accuracy was measured using binary crossentropy.  Accuracy was better on the postive-leaning tweets, but ranged from 80% - 90%.  The PowerPoint presentation is included in the repo for full detail.

## Overhaul & Model

The model used the following parameters to achieve a satisfactory output.

- Baseline_model: Dense(16,64,32, 1),  
- Larger_model: Dense(512, 512, 1), epochs=20, batch_size=512, 
- Smaller_model = Dense(4, 4, 1), epochs=20, batch_size=512, 
-	L2_model = Dense(16, 16, 1), Regularizers(0.001,), epochs=20, batch_size=512, verbose=2, acc = 80.5% 
-	Dropout_model: Dense(16, 32, 1), Droput(0.5, 0.5), epochs=20, batch_size=512, verbose=2) 

We used dropout methodologies and L2 modeling to increase accuracy of the model and to prevent overfitting.

## Application

Twitter's API was used to perform real-time testing of the model in a live environment, in this case during the final class presentation.

# Future Work

- Create Multi-classification model, labeled as Negative, Neutral, and Positive
- Try existing embedding models, e.g. Word2Vec, FastText, Elmo, BERT
- Gather new data sources to improve the model
- Apply to international airlines or different geographies
- Develop customer segmentation / cohort analysis for analyzing sentiment changes over time
