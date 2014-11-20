default: base

data.csv: ../dataset/*.csv
	cat ../dataset/*.csv > data.csv

data_preprocessed.csv: data.csv preprocessing.py
	python preprocessing.py

train.csv test.csv: data_preprocessed.csv resampling.py
	python resampling.py

.PHONY: base ensmble clean

# base model, pure naive bayes
base: train.csv test.csv model.py
	python -u model.py

# naive bayes with class specific decision boundary
boundary: train.csv test.csv model.py
	export set search=true; python -u model.py; unset search

# train two model on different feature sets, then combine them by MLE or entropy
ensemble: train.csv test.csv ensemble_model.py
	python ensemble_model.py

clean:
	-rm data.csv data_preprocessed.csv train.csv test.csv report*.txt \
		weights_of_words.json weights_of_classes.json
