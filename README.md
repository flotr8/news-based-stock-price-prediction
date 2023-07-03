# news-based-stock-price-prediction
News based stock price prediction

This repository consists of all steps in order to create a stock price prediction model based on technical indicators and different news sources.

The overall modelling is done in ModelWorkbook.
The module helpFunctions.py includes different custom created functions used frequentlly over different workbooks.

All files with leading 0 are part of the data collection process:
- 0.1: Collects Links for Headlines from Reuters.com (or Reuters.com/markets)
- 0.2: Those Links collected in 0.1 are stored locally and then used to collect the content and also store them locally
- 0.21: This workbook collects headlines from BusinessInsider for a defined stock
- 0.3: Simple Workbook to collect historical stock price data from Yahoo Finance

All files with leading 1 are part of the data processing process:
- 1.0: Sentiment Analysis module which performs classification of preloaded news with different pre-trained models on different news types
- 1.2: Sentence Embedding module which generates sentence embeddings and also brings them in the shape of the matching stock price data (padding etc) and stores them in pkl files to load in Model Workbook
- 1.3: Workbook to generate Technical Stock indicators with ta-lib

