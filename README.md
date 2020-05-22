# HateSpeechRecognition


*PythonNotebook for training your model is there in model folder.


## Introduction:
Our project comprises of a web application and a desktop application. 
The web application comes with two features which are followed as twitter user analysis and hashtag analysis and a desktop application which provides the feature of real time speech analysis. The feature of twitter user analysis helps to analyse previous tweets of a particular twitter handle from his/her timeline and determines the amount of negativity in the tweets. The feature of Hashtag analysis helps to get the hate percentage of tweets for that hashtag or topic by analysing previous 15000 tweets from current timestamp. 
Based on the same utilities as used by the web-app, our desktop application is designed  to provide real tiDevelop Apps for iOS and Android
Kotlin for Java Developers
JetBrains
COURSE
Free
Developing Android Apps with App Inventor
The Hong Kong University of Science and Technology
COURSE
Foundations of Objective-C App Development
University of California, Irvine
COURSE
Networking and Security in iOS Applications
University of California, Irvine
COURSE
Best Practices for iOS User Interface Design
University of California, Irvine
COURSE
Games, Sensors and Media
University of California, Irvine
COURSE
iOS Project: Transreality Game
University of California, Irvine
COURSE
Capstone MOOC for "Android App Development"
Vanderbilt University
COURSE
Java for Android
Vanderbilt University
COURSE
Engineering Maintainabl…
Vanderbilt University
COURSE
￼
Recently Launched Guided Projectsme hate analysis of input speech. This can be used to measure variation of hate sentiments present in the speaker’s speech with respect to time. The measure of hate sentiment is depicted using a 2-D plot. In addition to this, the desktop app can apply the real time speech analysis to audio and video source files too.




## Implementation








### Fetching:
In this project , we need real tweets from any particular twitter handler  account or from 		any hashtag using tweepy library		
We have converted everything into modules for the sake of simplicity:

. 



### Cleaning:
When we are done with feching tweets, we need to clean the tweets i.e removing any hyperlinks, unnecessary dates, special symbols, numerical values so that the tweets  contain only words from english corpus.	
Then we imported stopwords from  english corpus, and removes them from the tweets.	

Then  converting every word present in the tweets to its base form so that for every verb 
a particular word is present ( that two words with different tense and form aren’t treated as different words).


### Preprocessing:
We are using TF - IDF vectorization for converting textual data to numeric data that a  machine can understand. In TF-IDF, the words which occurs often are assigned a low tf-Idf value so they are of low importance.


    TF (Term Frequency) measures the frequency of a word in a document.
    TF = (Number of time the word occurs in the text) / (Total number of words in text)
    IDF (Inverse Document Frequency) measures the rank of the specific word for its relevancy within the text. Stop words which contain unnecessary information such as “a”, “into” and “and” carry less importance in spite of their occurrence.
    IDF = (Total number of documents / Number of documents with word t in it)

Thus, the TF-IDF is the product of TF and IDF: TF-IDF = TF * IDF

	
### Training
In this we have used logistic regeression, We can call a Logistic Regression a Linear Regression model but the Logistic Regression uses a more complex cost function, this cost function can be defined as the ‘Sigmoid function’ or also known as the ‘logistic function’ instead of a linear function. 
The hypothesis of logistic regression tends it to limit the cost function between 0 and 1.
				

 **Sigmoid Function:**
In order to map predicted values to probabilities, we use the Sigmoid function. The function maps any real value into another value between 0 and 1.

**Decision:**
We basically decide with a threshold value above which we classify values into Class 1 and of the value goes below the threshold then we classify it in Class 2.

**Cost Function:**
The cost function represents optimization objective i.e. we create a cost function and minimize it so that we can develop an accurate model with minimum error.


**Gradient Descent:**
Now we reduce the cost value by  using Gradient Descent. The main goal of Gradient descent is to minimize the cost value. i.e. min J(θ).
Now to minimize our cost function we need to run the gradient descent function on each parameter i.e.





### Prediction &  Accuracy:
In prediction , we input a csv file containing fetched tweets and the output we get a plot how much of it is hate-speech . We got the following plot for the test data and got accuracy of 96%.

   		  **Accuracy** : 0.961456638718558


### Saving trained model and vector:
Now we have successfully trained the model , and the vector with optimal values are saved along with the model, so that it can be directly imported and used .
We did this by using Jobutil library. 



