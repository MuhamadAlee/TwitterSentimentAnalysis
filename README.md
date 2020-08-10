# TwitterSentimentAnalysis
 A machine Learning and Deep Learning Project based Twitter Sentiment Analysis for Personality Detection using Emotions.
 
 We have used Twitter Sentiment for Personality Detection of five personalities eg. Barack Obama,Hillary Clinton, Jeremy Corbyn, Boris Johnson and Bill Gates.
 
 Step 1 : We have Scrapped the tweets of these Personalities 2000 each using the official API of Twitter called Tweepy.
 
 
 Step 2 : Then we have cleaned these Tweets as removing Stopwords, retweets, hashTags urls and emojis for specific text classification.
 
 Step 3 : Afterwards we have detected the the emotions for each of the tweet using NRC Emotion Lexicon which having 10 classes for emotion Charcteristics and append it to each of            tweet in the dataframe. 
 
 Step 4 : Then we have simply added the writer of each of the tweets in front of each of the tweet eg. Barack Obama. This will be our Dependent Feature.
 
 Step 5 : Then we have Applied five Machine Learning and one Deep Learning Models on the Same dataset which are.
          
          ==> Decision tree
          ==> Random forest
          ==> Naive Bayes
          ==> Logistic Regression
          ==> Support Vector Machine
         
         ==> Recurrent Neural Network (Long Short Term Memory Units)
         
 Step 5 : We have got reasonable accuracies for all of these models having best of the accuracy for RNN about 90%.
 
 Step 6 : Then we have pickle up all these models and store them in the pickle format for later usage.
 
 Step 7 : All this remains good as long as we also try to estimate emotions using the same NRC Emotion Lexicon  for each of the Tweet.
 
 Step 8 : At last we have made a web applictaion using Python framework Flask for interaction for the user.
 
 
