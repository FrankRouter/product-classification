default: base

data.csv: ../dataset/*.csv
	cat ../dataset/*.csv > data.csv

data_preprocessed.csv: data.csv preprocessing.py
	python -u preprocessing.py

train.csv cv.csv: data_preprocessed.csv resampling.py
	python -u resampling.py

.PHONY: base boundary base_persistence boundary_persistence ensmble test clean 

# base model, pure naive bayes
base: train.csv cv.csv model.py
	python -u model.py -t train.csv -c cv.csv

base_persistence: train.csv cv.csv model.py
	python -u model.py -t train.csv -c cv.csv -p

# naive bayes with class specific decision boundary
boundary: train.csv cv.csv model.py
	python -u model.py -t train.csv -c cv.csv -s

boundary_persistence: train.csv cv.csv model.py
	python -u model.py -t train.csv -c cv.csv -s -p

# train two model on different feature sets, then combine them by MLE or entropy
ensemble: train.csv cv.csv ensemble_model.py
	python ensemble_model.py

test: test.py test.csv ../allcategories.csv boundary.json
	python -u test.py

clean:
	-rm -rf data.csv data_preprocessed.csv train.csv cv.csv report*.txt \
		weights_by_classes bin/ boundary.json data_resampled.csv
