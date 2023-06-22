import glob
import pandas as pd
import os
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np


T_OPEN = dt.time(9,30,0)
T_CLOSE = dt.time(16,00,0)
ONE_DAY = dt.timedelta(1)
ONE_HOUR = dt.timedelta(hours = 1)
colours = ['r', 'b', 'g', 'c', 'y']


#Get Latest prepared Data
def getLatestData(type, ticker = None, model = 'ProsusAI/finbert'):
    if ticker is None: 
        if type == 'sentiments' or type == 'vectors':  
            folder_path = r'data/' + type + '/' + model 
        else:
            folder_path = r'data/' + type
    elif not ticker is None and (type == 'sentiments' or type =='vectors'):
            folder_path = r'data/' + type + '/' + model + '/' + ticker
    else:
         folder_path = r'data/' + type + '/' + ticker

    file_type = r'/*csv'
    files = glob.glob(folder_path + file_type)
    if not os.path.exists(folder_path):
        print("No directory/ data for " + type + " and " + ticker  + '.\nReturning empty Dataframe!')
        return pd.DataFrame(columns = ['Headline' ,'Datetime' ,'Positive', 'Negative' ,'Neutral'] )
    return pd.read_csv(max(files, key=os.path.getctime))

def parseDatetime(d):
    #Parse all Data on 30 minutes to aggregate on 60
    if 0 < d.time().minute < 30:
        d = d.replace(minute = 30, second = 0)
    else:
        if d.time().hour < 22:
            d = d.replace(hour = d.time().hour + 1, minute = 30, second = 0)
        else:
            d = d+ ONE_DAY
            d = d.replace(hour=9, minute=30)   
    #if not in trading times
    if not(T_OPEN < d.time() < T_CLOSE):
        #if after close, day plus one
        if d.time() > T_CLOSE:
            d = d + ONE_DAY
        d = d.replace(hour=9, minute=30)
    #if weekday not between 0 and 4, set to 0
    if not(0 <= d.weekday() <= 4):
        for i in range(0, 7- d.weekday()):
            d = d + ONE_DAY
        d = d.replace(hour=9, minute=30)
    return d

def plotResults(df, cols, y):
    plt.figure(figsize = (10,6))
    for i, col in enumerate(cols):
        plt.plot(df[col], df.index, label = col, c = colours[i])

    plt.legend()
    plt.ylabel(y)
    plt.xlabel('Time')
    plt.show()

# Compute the Bollinger Bands 
def BBANDS(data, window):
    MA = data.rolling(window).mean()
    SD = data.rolling(window).std()
    MiddleBand = MA
    UpperBand = MA + (2 * SD) 
    LowerBand = MA - (2 * SD)
    return MiddleBand, UpperBand, LowerBand

# Simple Moving Average 
def SMA(data, ndays): 
    SMA = pd.Series(data.rolling(ndays).mean(), name = 'SMA') 
    return SMA

# Exponentially-weighted Moving Average
def EWMA(data, ndays): 
    EMA = pd.Series(data.ewm(span = ndays, min_periods = ndays - 1).mean(), 
                 name = 'EWMA_' + str(ndays)) 
    return EMA

# Returns RSI values
def rsi(close, periods = 14):
    
    close_delta = close.diff()

    # Make two series: one for lower closes and one for higher closes
    up = close_delta.clip(lower=0)
    down = -1 * close_delta.clip(upper=0)
    
    ma_up = up.ewm(com = periods - 1, adjust=True, min_periods = periods).mean()
    ma_down = down.ewm(com = periods - 1, adjust=True, min_periods = periods).mean()

    rsi = ma_up / ma_down
    rsi = 100 - (100/(1 + rsi))
    return rsi

def gain(x):
    return ((x > 0) * x).sum()


def loss(x):
    return ((x < 0) * x).sum()


# Calculate money flow index
def mfi(high, low, close, volume, n=14):
    typical_price = (high + low + close)/3
    money_flow = typical_price * volume
    mf_sign = np.where(typical_price > typical_price.shift(1), 1, -1)
    signed_mf = money_flow * mf_sign
    mf_avg_gain = signed_mf.rolling(n).apply(gain, raw=True)
    mf_avg_loss = signed_mf.rolling(n).apply(loss, raw=True)
    return (100 - (100 / (1 + (mf_avg_gain / abs(mf_avg_loss))))).to_numpy()

# Returns the Force Index 
def ForceIndex(data, ndays): 
    FI = pd.Series(data['Close'].diff(ndays) * data['Volume'], name = 'ForceIndex') 
    return FI




def generateOutputFile():
    file_path = r'data/results/'                        # File Path for Results
    timestr = time.strftime("%Y%m%d-%H%M%S")
    file_path = file_path + 'results_' + timestr + '.csv'
    print("-" * 60)
    print("Generating File in " + file_path)
    print("-" * 60)
    header_row = ['Ticker','Features','n_past','Re-Train','RMSE','Accuracy','Actual Return abs', 'Actual Return %', 'Training Epochs','Start','End','Model Type', 'Train/Test']
    with open(file_path, 'w') as res:
        writer_object = writer(res)
        writer_object.writerow(header_row)
        res.close()
    
    return file_path

def configIndex(df, ticker, type):
    if type == "headlines" and (ticker is None or ticker =="marketsNews"):
        df['Datetime'] = df['Date'] + " " + df['Time']
        df.drop(columns = "Unnamed: 0", axis =0)
    elif type == "headlines":
        df['Datetime'] = df['Date']
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df['Datetime'] = df['Datetime'].dt.tz_localize(None)
    df = df.set_index('Datetime')
    return df

def transformHeadlines(df, s_d):
    df = df.reset_index()
    df['Datetime'] = df['Datetime'].apply(lambda row: parseDatetime(row))
    df['Counter'] = 1
    df.loc[df['Positive'] >= df['Negative'], 'Label'] = 1
    df.loc[df['Positive'] < df['Negative'], 'Label'] = -1
    df.drop_duplicates(subset = "Headline", keep =False, inplace = True)
    df_grouped = df.groupby('Datetime').agg({'Counter': 'sum' , 'Positive':'sum',
         'Negative': 'sum', 'Label': 'sum', 
         'Headline': lambda x: ' '.join(x)})
    df_grouped['length'] = df_grouped['Headline'].apply(lambda x: len(x))
    df_grouped = pd.merge(s_d, df_grouped, 'left' , left_index = True, right_index = True)
    df_grouped['Weighted_Score'] = df_grouped['Label'] / df_grouped['Counter']
    df_grouped['Score'] = df_grouped['Positive'] - df_grouped['Negative']
    df_grouped['Date'] = df_grouped.index.date
    df_grouped['Daily_Score'] = df_grouped['Score'].groupby(df_grouped['Date']).cumsum() 
    return df_grouped

def encode(titles):
    x = document_vector(model, titles)
    return x

def generate_encoding(sentence, use_model):
    sentence_embedding = use_model([sentence])[0].numpy()
    return sentence_embedding


def splitTrainTest(x_vector, x_prices, x_sentiments, y1, y2, stock_data, split_test, split_valid):
    train_length = len(stock_data[stock_data.index < split_valid]) - n_past
    valid_length = len(stock_data[stock_data.index < split_test]) - n_past
    x_vector_train, x_prices_train, x_sentiments_train, y_train1, y_train2 = x_vector[:train_length], x_prices[:train_length], x_sentiments[:train_length], y1[:train_length], y2[:train_length]
    x_vector_valid, x_prices_valid, x_sentiments_valid, y_valid1, y_valid2 = x_vector[train_length:valid_length], x_prices[train_length: valid_length], x_sentiments[train_length:valid_length],y1[train_length:valid_length],y2[train_length:valid_length]
    x_vector_test, x_prices_test, x_sentiments_test, y_test1, y_test2 = x_vector[valid_length:], x_prices[valid_length:], x_sentiments[valid_length:], y1[valid_length:], y2[valid_length:]
    return x_vector_train, x_prices_train,x_sentiments_train, y_train1, y_train2, x_vector_test, x_prices_test, x_sentiments_test, y_test1, y_test2, x_vector_valid, x_prices_valid, x_sentiments_valid, y_valid1, y_valid2

def splitTrainTest_noValid(x_vector, x_prices, x_sentiments, y1, y2, stock_data, split_test, split_valid):
    train_length = len(stock_data[stock_data.index < split_test]) - n_past
    x_vector_train, x_prices_train, x_sentiments_train, y_train1, y_train2 = x_vector[:train_length], x_prices[:train_length], x_sentiments[:train_length], y1[:train_length], y2[:train_length]
    x_vector_test, x_prices_test, x_sentiments_test, y_test1, y_test2 = x_vector[train_length:], x_prices[train_length:], x_sentiments[train_length:], y1[train_length:], y2[train_length:]
    return x_vector_train, x_prices_train,x_sentiments_train, y_train1, y_train2, x_vector_test, x_prices_test, x_sentiments_test, y_test1, y_test2


def createDatasets(df_complete, n_past):
#generate subset for different inputs
        x_vec1 = []
        x_vec2 = []
        x_vec3 = []
        x_vec = [x_vec1, x_vec2, x_vec3]
        x_price = []
        x_sentiment = []
        y = []

        for i, row in df_complete.iterrows():
            if 'GeneralNews' in cols_news:
                x_vec1.append(encode(row['GeneralNews']))
            if 'MarketsNews' in cols_news:
                x_vec2.append(encode(row['MarketsNews']))
            if 'StockNews' in cols_news:
                x_vec3.append(encode(row['StockNews']))
            x_price.append(row[cols_stock])
            x_sentiment.append(row[cols_sentiment])

            if Classification:
                y.append(row['Actual Direction'])
            else:
                y.append(row['Close'])

        del_c = 0 
        if not 'GeneralNews' in cols_news:
            del x_vec[0]
            del_c = del_c +1
        if not 'MarketsNews' in cols_news:
            del x_vec[1 - del_c]
            del_c = del_c +1
        if not 'StockNews' in cols_news:
            del x_vec[2 - del_c]   
        if len(cols_news) > 1:
            x_vec = np.reshape(x_vec, (len(x_vec[1]), len(cols_news) * 300))
        elif len(cols_news) > 0:
            x_vec = np.reshape(x_vec, (len(x_vec[0]), len(cols_news) * 300))

        sc = StandardScaler()
        sc_predict = StandardScaler()
        x_vec, x_price, x_sentiment, y = np.array(x_vec), np.array(x_price), np.array(x_sentiment), np.array(y)
        x_vec, x_price, x_sentiment, y = x_vec.astype(float), x_price.astype(float), x_sentiment.astype(float), y.astype(float)
        x_price = sc.fit_transform(x_price)
        x_sentiment = sc.fit_transform(x_sentiment)
        y = y.reshape(-1, 1)
        if not Classification:
            y = sc_predict.fit_transform(y)


        #Create dataset incl. lookback
        x_vector = []
        x_prices = []
        x_sentiments = []
        y_ = []

        for i in range(n_past, len(y)):
            if len(x_vec)> 0 :
                x_vector.append(x_vec[i - n_past:i+1,: x_vec.shape[1]])
            x_prices.append(x_price[i - n_past:i+1,: x_price.shape[1]])
            x_sentiments.append(x_sentiment[i-n_past:i+1,: x_sentiment.shape[1]])
            y_.append(y[i])


        x_vector, x_prices, x_sentiments, y_ = np.array(x_vector), np.array(x_prices), np.array(x_sentiments), np.array(y_)

        return x_vector, x_prices, x_sentiments, y_, sc, sc_predict

def preprocess_news(news_text):
    # Tokenize the text
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(news_text)
    sequences = tokenizer.texts_to_sequences(news_text)
    
    # Remove stopwords and punctuations
    stop_words = set(stopwords.words('english'))
    sequences = [[word for word in doc if word not in stop_words and word.isalpha()] for doc in sequences]
    
    # Perform lemmatization
    lemmatizer = WordNetLemmatizer()
    sequences = [[lemmatizer.lemmatize(word) for word in doc] for doc in sequences]
    
    # Pad the sequences to the max length
    max_length = max([len(s) for s in sequences])
    sequences = pad_sequences(sequences, maxlen=max_length)
    
    return sequences, tokenizer

def createPaddedNews(newstype, data, window_size, embedding_model = "word2vec"):
    news = data[newstype].values
    if embedding_model == "word2vec":
        # Preprocess the news data
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(news)
        word_index = tokenizer.word_index
        sequences = tokenizer.texts_to_sequences(news)
        max_len = max([len(news) for news in sequences])
        n_samples = len(sequences) - window_size + 1
        padded_news = np.zeros((n_samples, window_size, max_len))

        for i in range(n_samples):
            for j in range(window_size):
                padded_news[i, j, :len(sequences[i + j])] = sequences[i + j]

        # prepare embedding matrix
        embedding_matrix = np.zeros((len(word_index) + 1, w2v_model.vector_size))
        for word, i in word_index.items():
            if word in w2v_model:
                embedding_matrix[i] = w2v_model[word]

        return padded_news,embedding_matrix, max_len, word_index
    elif embedding_model == "bert":
        max_len = 512
        bert_tokens = bert_tokenizer.batch_encode_plus(news, max_length=max_len, padding='max_length', truncation=True, return_attention_mask=True)
        padded_bert_input_ids = np.zeros((len(bert_tokens['input_ids']) - window_size + 1, window_size, max_len))
        padded_bert_attention_masks = np.zeros((len(bert_tokens['attention_mask']) - window_size + 1, window_size, max_len))
        for i in range(len(padded_bert_input_ids)):
            for j in range(window_size):
                padded_bert_input_ids[i, j, :len(bert_tokens['input_ids'][i + j])] = bert_tokens['input_ids'][i + j]
                padded_bert_attention_masks[i, j, :len(bert_tokens['attention_mask'][i + j])] = bert_tokens['attention_mask'][i + j]
        return padded_bert_input_ids, padded_bert_attention_masks

def create_input_arrays(df, ws_price, ws_news):
    input_prices = df[cols_stock] # you can use any columns from your dataframe
    sentiment_scores = df[cols_sentiment]
    labels = df['Actual Direction']
    prices = df['Close']
    input_prices_array = []
    sentiment_scores_array = []
    labels_array = []
    prices_array = []
    for i in range(len(df) - ws_price+1):
        input_prices_array.append(input_prices.iloc[i:i+ws_price].values)
        sentiment_scores_array.append(sentiment_scores.iloc[i:i+ws_price].values)
        labels_array.append(labels.iloc[i+ws_price-1])
        prices_array.append(prices.iloc[i+ws_price-1])

    for i in range(len(df) - ws_news+1):
        sentiment_scores_array.append(sentiment_scores.iloc[i:i+ws_news].values)

    input_prices_array,sentiment_scores_array, labels_array, prices_array = np.array(input_prices_array), np.array(sentiment_scores_array), np.array(labels_array), np.array(prices_array)
    scalers = {}
    for i in range(input_prices_array.shape[1]):
        scalers[i] = StandardScaler()
        input_prices_array[:, i, :] = scalers[i].fit_transform(input_prices_array[:, i, :]) 

    for i in range(sentiment_scores_array.shape[1]):
        scalers[i] = StandardScaler()
        sentiment_scores_array[:, i, :] = scalers[i].fit_transform(sentiment_scores_array[:, i, :]) 



    prices_array =np.reshape(prices_array, (prices_array.shape[0], 1))
    labels_array = np.reshape(labels_array,(labels_array.shape[0],1) )

    sc_predict = StandardScaler()
    prices_array = sc_predict.fit_transform(prices_array)

    
    return input_prices_array, sentiment_scores_array, labels_array, prices_array, sc_predict


def create_input_arrays3(df, window_size):
    input_prices = df[cols_stock] # you can use any columns from your dataframe
    sentiment_scores = df[cols_sentiment]
    sc = StandardScaler()
    sc_predict = StandardScaler()
    input_prices = sc.fit_transform(input_prices)
    sentiment_scores = sc.fit_transform(sentiment_scores)
    labels = df['Actual Direction']
    prices = df['Close']
    labels = np.array(labels)
    prices = np.array(prices)
    prices = prices.reshape(-1, 1)
    prices = sc_predict.fit_transform(prices)
    input_prices_array = []
    sentiment_scores_array = []
    labels_array = []
    prices_array = []
    for i in range(len(df) - window_size+1):
        input_prices_array.append(input_prices[i:i+window_size])
        sentiment_scores_array.append(sentiment_scores[i:i+window_size])
        labels_array.append(labels[i+window_size-1])
        prices_array.append(prices[i+window_size-1])

    input_prices_array,sentiment_scores_array, labels_array, prices_array = np.array(input_prices_array), np.array(sentiment_scores_array), np.array(labels_array), np.array(prices_array)




    prices_array =np.reshape(prices_array, (prices_array.shape[0], 1))
    labels_array = np.reshape(labels_array,(labels_array.shape[0],1) )

    
    return input_prices_array, sentiment_scores_array, labels_array, prices_array, sc_predict


def create_input_arrays2(df, window_size):
    input_prices = df[cols_stock]
    sentiment_scores = df[cols_sentiment]
    labels = df['Actual Direction']
    prices = df['Close']
    input_prices_array = []
    sentiment_scores_array = []
    labels_array = []
    prices_array = []
    for i in range(len(df) - window_size+1):
        input_prices_array.append(input_prices.iloc[i:i+window_size].values)
        sentiment_scores_array.append(sentiment_scores.iloc[i:i+window_size].values)
        labels_array.append(labels.iloc[i+window_size-1])
        prices_array.append(prices.iloc[i+window_size-1])

    input_prices_array,sentiment_scores_array, labels_array, prices_array = np.array(input_prices_array), np.array(sentiment_scores_array), np.array(labels_array), np.array(prices_array)
    scalers = {}
    for i in range(input_prices_array.shape[1]):
        scalers[i] = MinMaxScaler()
        input_prices_array[:, i, :] = scalers[i].fit_transform(input_prices_array[:, i, :]) 

    for i in range(sentiment_scores_array.shape[1]):
        scalers[i] = MinMaxScaler()
        sentiment_scores_array[:, i, :] = scalers[i].fit_transform(sentiment_scores_array[:, i, :]) 

    prices_array = np.reshape(prices_array, (prices_array.shape[0], 1))
    labels_array = np.reshape(labels_array,(labels_array.shape[0],1) )

    sc_predict = MinMaxScaler()
    prices_array = sc_predict.fit_transform(prices_array)
    
    return input_prices_array, sentiment_scores_array, labels_array, prices_array, sc_predict


def modelClass(x_vector = None, x_prices = None, x_sentiments = None):

    if not x_vector is None and not x_prices is None and not x_sentiments is None:
        # Define the input layers for the embedded news, sentiment scores, and technical stock numbers
        news_input = Input(shape=(n_past, x_vector.shape[2]))
        stock_input = Input(shape=(n_past, x_prices.shape[2]))
        sentiment_input = Input(shape=(n_past, x_sentiments.shape[2]))

        # Combine the input layers into a single input layer
        input_layer = concatenate([news_input, sentiment_input, stock_input])

        # Define the LSTM layer
        lstm_layer = LSTM(units=64)(input_layer)

        # Define the output layer
        output_layer = Dense(units=1, activation='hard_sigmoid')(lstm_layer)

        # Create the model
        model = Model(inputs=[news_input, sentiment_input, stock_input], outputs=output_layer)

        # Compile the model
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    elif not x_vector is None and not x_prices is None:
        # Define the input layers for the news and technical stock numbers
        news_input = Input(shape=(n_past, x_vector.shape[2]))
        stock_input = Input(shape=(n_past, x_prices.shape[2]))

        # Combine the input layers into a single input layer
        input_layer = concatenate([news_input, stock_input])

        # Define the LSTM layer
        lstm_layer = LSTM(units=64)(input_layer)

        # Define the output layer
        output_layer = Dense(units=1, activation='sigmoid')(lstm_layer)

        # Create the model
        model = Model(inputs=[news_input, stock_input], outputs=output_layer)

        # Compile the model
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
 
    elif not x_sentiments is None and not x_prices is None:
        # Define the input layers for the news and technical stock numbers
        sentiment_input = Input(shape=(n_past, x_sentiments.shape[2]))
        stock_input = Input(shape=(n_past, x_prices.shape[2]))

        # Combine the input layers into a single input layer
        input_layer = concatenate([news_input, sentiment_input])

        # Define the LSTM layer
        lstm_layer = LSTM(units=64)(input_layer)

        # Define the output layer
        output_layer = Dense(units=1, activation='sigmoid')(lstm_layer)

        # Create the model
        model = Model(inputs=[news_input, stock_input], outputs=output_layer)

        # Compile the model
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    else:
        # Define the input layer for the technical stock prices
        stock_input = Input(shape=(n_past, x_prices.shape[2]))

        # Define the LSTM layer
        lstm_layer = LSTM(units=64)(stock_input)

        # Define the output layer
        output_layer = Dense(units=1, activation='sigmoid')(lstm_layer)

        # Create the model
        model = Model(inputs=stock_input, outputs=output_layer)

        # Compile the model
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model


def modelClass_incl_Embedding(window_size, embedding_model, max_len, word_index):

    # input layers
    stock_prices_input = Input(shape=(window_size, len(cols_stock)))
    sentiment_scores_input = Input(shape=(window_size, len(cols_sentiment)))
    
    # LSTM models
    stock_prices_lstm = LSTM(64)(stock_prices_input)
    sentiment_scores_lstm = LSTM(64)(sentiment_scores_input)

    # Embedding layer for news
    if embedding_model == 'word2vec':
        news_input = Input(shape=(window_size, max_len))
        news_embedding = Embedding(len(word_index) + 1, 300, weights=[embedding_matrix], input_length=max_len, trainable=False)(news_input)
        news_embedding = TimeDistributed(Conv1D(filters=32, kernel_size=3, activation='relu'))(news_embedding)
        news_embedding = TimeDistributed(MaxPooling1D(pool_size=2))(news_embedding)
        news_embedding = TimeDistributed(GlobalMaxPooling1D())(news_embedding)
    elif embedding_model == 'bert':
         # Define the input layer for news data
        # Define the input layer
        news_input = tf.keras.layers.Input(shape=(window_size, max_len), dtype=tf.int32)

        # Reshape the input to (batch_size, window_size*max_len)
        news_input_reshaped = tf.keras.layers.Reshape((window_size*max_len,), name="input_word_ids")(news_input)

        # Use the BERT model to get the word embeddings
        sequence_output = bert_model(news_input_reshaped)[0]

        # Reshape the output to (batch_size, window_size, max_len, embedding_size)
        news_embedding = tf.keras.layers.Reshape((window_size *  max_len, -1), name="sequence_output")(sequence_output)



    news_lstm = LSTM(64)(news_embedding)


    # merge the output
    merged = Concatenate()([ news_lstm, sentiment_scores_lstm,stock_prices_lstm])

    # output layer
    direction_output = Dense(1, activation='sigmoid', name='direction_output')(merged)
    price_output = Dense(1, activation='linear', name='price_output')(merged)

    
    # create the model
    model = Model(inputs=[news_input,sentiment_scores_input, stock_prices_input], outputs=[direction_output, price_output])
    #model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.compile(optimizer='adam', 
              loss={'direction_output': 'binary_crossentropy', 'price_output': 'mean_squared_error'},
                loss_weights={'direction_output': 0.8, 'price_output': 0.2},
              metrics={'direction_output': ['accuracy', 'binary_crossentropy'], 'price_output': ['mae']})

    return model

def modelClass_no_Embedding(window_size):
    # input layers
    stock_prices_input = Input(shape=(window_size, len(cols_stock)))
    sentiment_scores_input = Input(shape=(window_size, len(cols_sentiment)))
    
    # LSTM models
    stock_prices_lstm = LSTM(64, return_sequences= True)(stock_prices_input)
    sentiment_scores_lstm = LSTM(64, return_sequences= True)(sentiment_scores_input)

    # merge the output
    merged = Concatenate()([sentiment_scores_lstm, stock_prices_lstm])

    merged = LSTM(64)(merged)

    # output layer
    direction_output = Dense(1, activation='hard_sigmoid', name='direction_output')(merged)
    price_output = Dense(1, activation='linear', name='price_output')(merged)

    # create the model
    model = Model(inputs=[sentiment_scores_input, stock_prices_input], outputs=[direction_output, price_output])
    #model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.compile(optimizer='adam', 
              loss={'direction_output': 'binary_crossentropy', 'price_output': 'mean_squared_error'},
               # loss_weights={'direction_output': 0.8, 'price_output': 0.2},
              metrics={'direction_output': ['accuracy', 'binary_crossentropy'], 'price_output': ['mae']})

    return model

def modelClass_onlyEmbedd(window_size):
    # Embedding layer for news
    if embedding_model == 'word2vec':
        news_input = Input(shape=(window_size, max_len))
        news_embedding = Embedding(len(word_index) + 1, 300, weights=[embedding_matrix], input_length=max_len, trainable=False)(news_input)
    elif embedding_model == 'bert':
        # Define the input layer for your news data
        max_len = 1601
        news_input = Input(shape=(window_size, max_len))
        news_embedding = bert(news_input)

    # Embedding layer for news
    #news_embedding = Embedding(len(word_index) + 1, 300, weights=[embedding_matrix], input_length=max_len, trainable=False)(news_input)
    news_embedding = TimeDistributed(Conv1D(filters=32, kernel_size=3, activation='relu'))(news_embedding)
    news_embedding = TimeDistributed(MaxPooling1D(pool_size=2))(news_embedding)
    news_embedding = TimeDistributed(GlobalMaxPooling1D())(news_embedding)
    news_lstm = LSTM(64)(news_embedding)


    # output layer
    direction_output = Dense(1, activation='sigmoid', name='direction_output')(news_lstm)
    price_output = Dense(1, activation='linear', name='price_output')(news_lstm)

    
    # create the model
    model = Model(inputs= news_input, outputs=[direction_output, price_output])
    #model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.compile(optimizer='adam', 
              loss={'direction_output': 'binary_crossentropy', 'price_output': 'mean_squared_error'},
              loss_weights={'direction_output': 0.8, 'price_output': 0.2},
              metrics={'direction_output': ['accuracy', 'binary_crossentropy'], 'price_output': ['mae']})

    return model

def modelClass_only_stock(window_size):
    # input layer
    stock_prices_input = Input(shape=(window_size, len(cols_stock)))

    # LSTM model
    stock_prices_lstm = LSTM(64)(stock_prices_input)

    # output layer
    direction_output = Dense(1, activation='sigmoid', name='direction_output')(stock_prices_lstm)
    price_output = Dense(1, activation='linear', name='price_output')(stock_prices_lstm)

    # create the model
    model = Model(inputs=stock_prices_input, outputs=[direction_output, price_output])
    #model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.compile(optimizer='adam', 
              loss={'direction_output': 'binary_crossentropy', 'price_output': 'mean_squared_error'},
              loss_weights={'direction_output': 0.8, 'price_output': 0.2},
              metrics={'direction_output': ['accuracy', 'binary_crossentropy'], 'price_output': ['mae']})

    return model



def modelRegression(layer_amount):

    if layer_amount == 2:
        input_words = Input(shape=(n_past+1, x_vector.shape[2]))
        #input_words = Input(shape = (len(cols_news) * 300))
        #e = Embedding(3,1,input_length = 300) (input_words)
        e = LSTM(64)(input_words)
        #e = LSTM(64) (e)
        e = Dropout(0.25)(e)
        e = Dense(1, activation='linear')(e)
        e  = Model(inputs = input_words, outputs = e)

        input_stocks = Input(shape=(n_past+1, x_prices_train.shape[2]))
        s = LSTM(64)(input_stocks)
        #s = LSTM(64)(s)
        s = Dropout(0.25)(s)
        s = Dense(units = 1, activation = 'linear') (s)
        s = Model(inputs = input_stocks, outputs = s)

        combined = layers.concatenate([e.output, s.output])

        z = Dense(1,activation ='linear') (combined)

        stacked_model = Model(inputs = [e.input, s.input], outputs = z)
        stacked_model.compile(optimizer = 'adam' , loss='mean_squared_error')

        return stacked_model

    else:
        input_stocks = Input(shape=(n_past+1, x_prices.shape[2]))
        s = LSTM(128, return_sequences = True)(input_stocks)
        s = LSTM(64)(s)
        s = Dropout(0.25)(s)
        s = Dense(units = 1, activation = 'linear') (s)
        s = Model(inputs = input_stocks, outputs = s)
        s.compile(optimizer = 'adam' , loss='mean_squared_error')

        return s

def evalModel(preds, data, evalType, sc_predict):
    result = data.copy()
    result.loc[result['Close'] > result['Open'], 'Actual Direction'] = 1
    result.loc[result['Close'] <= result['Open'], 'Actual Direction'] = 0
    if not Classification:
        result['Prediction'] = sc_predict.inverse_transform(preds)
        MSE = np.square(np.subtract(result['Prediction'], result['Close'])).mean() 
        RMSE = math.sqrt(MSE)
        result.loc[result['Prediction'] > result['Open'], 'Decision'] = 1
        result.loc[result['Prediction'] <= result['Open'], 'Decision'] = 0
    else:
        result['Prediction']= preds[0]
        result['Price_Prediction'] = sc_predict.inverse_transform(preds[1])
        #result['Price_Prediction'] = preds[1]
        result.loc[result['Prediction'] > 0.5, 'Decision']  = 1
        result.loc[result['Prediction'] <= 0.5, 'Decision']  = 0
        MSE = np.square(np.subtract(result['Price_Prediction'], result['Close'])).mean() 
        RMSE = math.sqrt(MSE)



    start_credit = 10000
    credit = start_credit
    # Return Calculation based on actual trades
    for j in result.index:
        amount = credit / result.loc[j, 'Open']              #replace credit with start_credit to ignore Zinseszins
        #amount = 10                    
        #if abs(result.loc[j, 'predicted'] - result.loc[j, 'open']) > 0.5: #Only trade "Big Changes"
        if result.loc[j, 'Decision']:
            result.loc[j, 'return'] = result.loc[j, 'Close'] * amount - result.loc[j, 'Open'] * amount
        else:
            result.loc[j, 'return'] = result.loc[j, 'Open'] * amount - result.loc[j, 'Close'] * amount
        #else:
         #   result.loc[j, 'return'] = 0
        credit = credit + result.loc[j, 'return']
        

    return_abs = credit - start_credit
    return_per = ((credit / start_credit) -1) *100

    print('-' * 60)
    print("Root Mean Square Error:")
    print(RMSE)
    print('-' * 60)
    print("\nAccuracy: ")
    print(accuracy_score(result['Actual Direction'], result['Decision']))
    print(confusion_matrix(result['Actual Direction'], result['Decision']))
    print('-' * 60)
    print("\nActual Return Absolut:")
    print(return_abs)
    print('-' * 60)
    print('-' * 60)
    print("\nActual Return %:")
    print(return_per)
    print('-' * 60)


    timestr = time.strftime("%Y%m%d-%H%M%S")
    result.to_csv('data/results/' + evalType +  "/Results_Evaluation" + timestr +'.csv')
    return RMSE, accuracy_score(result['Actual Direction'], result['Decision']), return_abs, return_per
