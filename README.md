# MultiTurnResponseSelection

This is our Douban Conversation Corpus for retrieval-based chatbot under multi-turn conversation scenario. The test data contains 1000 dialogue context, and for each context we create 10 responses as candidates. We recruited three labelers to judge if a candidate is a proper response to the session. A proper response means the response can naturally reply to the message given the context. Each pair received three labels and the majority of the labels was taken as the final decision.

|      |Train|Val| Test         | 
| ------------- |:-------------:|:-------------:|:-------------:|
| session-response pairs  | 1m|50k| 10k |
| Avg. positive response per session     | 1|1| 1.18    | 
| Fless Kappa | N\A|N\A|0.41      | 
| Min turn per session | 3|3| 3      | 
| Max ture per session | 53|40|44    | 
| Average turn per session | 3.03|5.81|5.95    | 
| Average Word per utterance | 16.75|17.22|17.17    | 
