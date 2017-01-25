# Douban Conversation Corpus

## Data set
We release Douban Conversation Corpus, comprising a training data set, a development set and a test set for retrieval based chatbot. The statistics of Douban Conversation Corpus are shown in the following table. 
<center>
|      |Train|Val| Test         | 
| ------------- |:-------------:|:-------------:|:-------------:|
| session-response pairs  | 1m|50k| 10k |
| Avg. positive response per session     | 1|1| 1.18    | 
| Fless Kappa | N\A|N\A|0.41      | 
| Min turn per session | 3|3| 3      | 
| Max ture per session | 98|91|45    | 
| Average turn per session | 6.69|6.75|5.95    | 
| Average Word per utterance | 18.56|18.50|20.74   | 
</center>

The test data contains 1000 dialogue context, and for each context we create 10 responses as candidates. We recruited three labelers to judge if a candidate is a proper response to the session. A proper response means the response can naturally reply to the message given the context. Each pair received three labels and the majority of the labels was taken as the final decision.

<br>
As far as we known, this is the first human-labeled test set for retrieval-based chatbots. Anyone can download the test set here. We upload the training and dev set at https://www.dropbox.com/s/eo5zegg1k83uqaj/DoubanConversaionCorpus.zip?dl=0

## Source Code
We also release our source code to help others reproduce our result. The code has been tested under Ubuntu 14.04 with python 2.7. The authors are busy polishing the code style in order to change the parameters more convenient.
