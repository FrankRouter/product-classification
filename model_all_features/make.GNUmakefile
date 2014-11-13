default: clean data run

data:
	cat dataset/* > data.csv

run: data.csv
	python remove_punctuation.py
	python segmentation.py
	python resampling.py
	python split_dataset.py
	python model.py

clean:
	-mv report.txt report.txt.old