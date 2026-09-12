.PHONY: install data train evaluate figures pipeline test app clean

install:
	pip install -r requirements.txt

data:
	python scripts/prepare_data.py

train:
	python -m src.training.train_baseline
	python -m src.training.train_lstm

evaluate:
	python -m src.evaluation.evaluate

figures:
	python -m src.visualization.plots

pipeline: ## Full reproducible run: data -> train -> evaluate -> figures
	python scripts/train_all.py

test:
	pytest tests/ -v

app:
	streamlit run app/streamlit_app.py

clean:
	rm -rf data/processed/splits/* artifacts/metrics/* artifacts/figures/* models/*
