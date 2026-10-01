# Data

This project uses the **IMDB Dataset (Sentiment Analysis) in CSV format** — 50,000 English movie reviews with binary labels (1 = positive, 0 = negative).

1. Download it from Kaggle: https://www.kaggle.com/datasets/columbine/imdb-dataset-sentiment-analysis-in-csv-format
   - or with the Kaggle CLI: `kaggle datasets download -d columbine/imdb-dataset-sentiment-analysis-in-csv-format --unzip -p data/`
2. Make sure these files are in this folder:
   ```
   data/Train.csv   # 40,000 reviews
   data/Valid.csv   #  5,000 reviews
   data/Test.csv    #  5,000 reviews
   ```

The notebook reads them from `../data/` (relative to `notebooks/`). The CSV files are git-ignored.
