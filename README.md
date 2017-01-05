# MultiTurnResponseSelection

This is our test set for retrieval-based chatbot under multi-turn conversation scenario. The test data contains 1000 dialogue context, and for each context we create 10 responses as candidates. We recruited three labelers to judge 
	if a candidate is a proper response to the session. A proper response means the response can naturally reply to the message given the context. Each pair received three labels and the majority of the labels was taken as the final decision.

|      | Dataset         | 
| ------------- |:-------------:|
| session-response pairs   | 10,000 |
| Avg. positive response per session     |  1.18    | 
| Fless Kappa | 0.41      | 
| Min turn per session | 3      | 
| Max ture per session | 44    | 
| Average turn per session | 5.95    | 
| Average Word per utterance | 17.17    | 
