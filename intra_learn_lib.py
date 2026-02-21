import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

#sys.path.append("E:\\ML\\SERVER_DATA\\MAIN\\my_pyth\\LIB")
from intra_lib import *

# Make numpy values easier to read.
np.set_printoptions(precision=4, suppress=True)

THRESHOLD = 0
VERBOSE_MODE = 1
MODEL_NAME_LONG = "model_min_full"
MODEL_NAME_SHORT = "model_max_full"
HARDWARE_MODE = 'PDF'

METRICS = [
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'), 
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
      keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
]

ACCURACY_THRESHOLD = 1

class myCallback(tf.keras.callbacks.Callback):
	def on_epoch_end(self, epoch, logs={}):
		if(float(logs.get('accuracy')) >= ACCURACY_THRESHOLD):
			print("\nReached %2.2f%% accuracy, so stopping training!!" %(ACCURACY_THRESHOLD*100))
			self.model.stop_training = True


def plot_loss(history, label, n, pdf):
  if HARDWARE_MODE != 'PC':
    fig = plt.figure(figsize=(12, 8))
    # Use a log scale on y-axis to show the wide range of values.
    plt.semilogy(history.epoch, history.history['loss'],
                 color=n, label='Train ' + label)
    plt.semilogy(history.epoch, history.history['val_loss'],
                 color=n, label='Val ' + label,
                 linestyle="--")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(label)
    pdf.savefig()
    plt.close(fig)

def make_model(metrics, NEURO_NUMBER, train_features, output_bias=None):
  LEARNING_RATE=1e-5
  print("LEARNING_RATE=", LEARNING_RATE) 
  if output_bias is not None:
    output_bias = tf.keras.initializers.Constant(output_bias)
  model = keras.Sequential([
          keras.layers.Dense( NEURO_NUMBER, activation='relu', input_shape=(train_features.shape[-1],)),
          keras.layers.Dropout(0.5),
          #keras.layers.Dense( 1764, activation='relu', input_shape=(train_features.shape[-1],)),  # added
          #keras.layers.Dropout(0.5),                                                              # added
          keras.layers.Dense(1, activation='sigmoid', bias_initializer=output_bias),
          ])

  model.compile(
      optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
      loss=keras.losses.BinaryCrossentropy(),
      metrics=metrics)

  return model


def get_best_epoch(history):
  epoch_list  = history.epoch  
  pre_list    = history.history['precision']
  recall_list = history.history['recall']
  max_pre = max(pre_list)
  # Calculate max threshold for presicion in order to show after training
  PRE_THRESHOLD = 1
  pre_thresh_list = [0.5, 0.6, 0.7, 0.8, 0.9, 1]
  for i in pre_thresh_list:
    if max_pre >= i:
      PRE_THRESHOLD = i
  max_index = pre_list.index(max_pre)
  print(max_pre, "id: ", max_index)
  print("Precision threshold achived: ", PRE_THRESHOLD, "precision: ", pre_list[max_index], 
        "recall: ", recall_list[max_index], "EPOCH: ", epoch_list[max_index]+1)

  counter = 0
  recal_thresh_list = [0.5, 0.6, 0.7, 0.8, 0.9, 1]

  # Serching the best case from tail. For EPOCH=6000 start gets started from id 4800  
  start_id = int(len(epoch_list)*0.8)
  max_precision = 0
  max_recall = 0
  max_id  = 0
  #print("range ", epoch_list[0], "-", epoch_list[len(epoch_list)-1])
  for i in range(0, len(epoch_list)):
    # Searching the best result only after specific EPOCH iteration
    if i >= start_id:
      if pre_list[i] > max_precision:
        #print(max_id, "is changed into", i)
        max_precision = pre_list[i]
        max_id  = i
        max_recall = recall_list[i]
      elif pre_list[i] == max_precision and recall_list[i] >= max_recall:
        max_precision = pre_list[i]
        max_id  = i
        max_recall = recall_list[i]
    
    if counter <= len(recal_thresh_list) - 1:
       if pre_list[i] >= PRE_THRESHOLD and recall_list[i] >= recal_thresh_list[counter]:
          print("precision = ", PRE_THRESHOLD, "; racall = ", recal_thresh_list[counter], "achived at EPOCH ", (i+1))
          counter += 1        

  # Range starts from 0 For example for EPOCH 300 it is 0...299. So real epoch number should be (max_id+1)
  real_epoch = max_id+1 
  print("Best case: EPOCH:", real_epoch, "precision:", max_precision, "recall:", max_recall)
  #print("Best case: EPOCH:", epoch_list[max_id], "precision:", pre_list[max_id], "recall:", recall_list[max_id])

  return real_epoch
 
####### Show metrix #######

def plot_metrics(history, pdf, best_epoch):
  metrics = ['loss', 'prc', 'precision', 'recall']
  fig = plt.figure(figsize=(12, 8))
  for n, metric in enumerate(metrics):
    name = metric.replace("_"," ").capitalize()
    plt.subplot(2,2,n+1)
    plt.plot(history.epoch, history.history[metric], color='r', label='Train')
    plt.plot(history.epoch, history.history['val_'+metric],
             color='g', linestyle="--", label='Val')
    plt.xlabel('Epoch')
    plt.ylabel(name)
    if metric == 'loss':
      plt.ylim([0, plt.ylim()[1]])
    elif metric == 'auc':
      plt.ylim([0.8,1])
    else:
      plt.ylim([0,1])

    plt.legend()
  pdf.savefig()
  plt.close(fig)

  best_weight_ttl = "Best epoch " +str(best_epoch)
  # Show tp (True Positive) and fp (False Positive) and fn (False Negative)
  start_epoch = int(len(history.epoch)*0.8)
  fig = plt.figure(figsize=(12, 8))
  plt.title("Cases: tp (True Positive) and fp (False Positive) and fn (False Negative)")
  plt.plot(history.epoch[start_epoch:], history.history['tp'][start_epoch:], color='g', label='True Positive')
  plt.plot(history.epoch[start_epoch:], history.history['fp'][start_epoch:], color='r', label='False Positive')
  plt.plot(history.epoch[start_epoch:], history.history['fn'][start_epoch:], color='k', label='False Negative')
  plt.vlines(best_epoch, 0, max(history.history['tp'][start_epoch:]), linestyles ="dotted", colors ="k", label=best_weight_ttl)
  plt.xlabel('Epoch')
  plt.legend()
  pdf.savefig()
  plt.close(fig)

  fig = plt.figure(figsize=(12, 8))
  plt.title("Cases (part): precision and recall")
  plt.plot(history.epoch[start_epoch:], history.history['precision'][start_epoch:], color='g', label='Precision')
  plt.plot(history.epoch[start_epoch:], history.history['recall'][start_epoch:], color='k', label='Recall')
  plt.vlines(best_epoch, 0, max(history.history['precision'][start_epoch:]), linestyles ="dotted", colors ="k", label=best_weight_ttl)
  plt.xlabel('Epoch')
  plt.legend()
  pdf.savefig()
  plt.close(fig)


def plot_cm(labels, predictions, p, pdf):
  cm = confusion_matrix(labels, predictions > p)
  if len(cm[0]) < 1:
    print("WARNING: cm is NOT ok")
    return
 
  if p == 0.5:
    plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title('Confusion matrix @{:.2f}'.format(p))
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    pdf.savefig()
    plt.close()
  print('Looses Cases Detected (True Negatives): ', cm[0][0])
  print('Looses Cases Incorrectly Detected (False Positives): ', cm[0][1])
  print('Successful Cases Missed (False Negatives): ', cm[1][0])
  print('Successful Cases Detected (True Positives): ', cm[1][1])
  print('Total Successful Transactions: ', np.sum(cm[1]))
  if (cm[1][1]+cm[0][1]) > 0:
    print('PRED', p, 'RESULT: ', cm[1][1]/(cm[1][1]+cm[0][1]))
  else:
    print('PRED', p, 'RESULT: ', 0)


def tune_ds(df, TUNE_DS_FLAG, exclude_file):
  if TUNE_DS_FLAG == False:
    return df
  print("tune_ds() is called\nType = 1" )
  print("THRESHOLD=", THRESHOLD, " all such tickers dropped")
  in_len = len(df)
  to_remove_list = []
  to_remove_list2 = [] # Stores full names with path
  previous_ticker=""
  counter = 0
  for index, row in df.iterrows():
    if previous_ticker != df.at[index, 'File']:
      if previous_ticker != "":
        print("  ", previous_ticker, counter)
        if counter <= THRESHOLD:
          to_remove_list2.append(previous_ticker)
          # Save file name only
          k_id = previous_ticker.find("/")          
          to_remove_list.append(previous_ticker[k_id+1:])

      counter = 0  
      previous_ticker = df.at[index, 'File']

    if df.at[index, 'Type'] == 1:
      counter+=1
  print("  ", previous_ticker, counter) #print last one 

  print("Tickers to be removed:", to_remove_list2)
  print("Number:", len(to_remove_list2))
  for elem in to_remove_list2:
    df = df[df['File'] != elem]

  print("Input df size:", in_len, "Output df size:", len(df))
  df_tmp = pd.DataFrame(to_remove_list)
  df_tmp.to_csv(exclude_file, index=False, header=False)
  return df


def get_df_by_dataset_file(input_file, MODEL, pdf, TUNE_DS_FLAG):
  print(MODEL)
  DATASET_COLUMNS = get_dataset_columns_full(MODEL)
  print("DATASET_COLUMNS:\n", DATASET_COLUMNS)
  exclude_file = get_base_name(input_file) + "_exclude.csv"
  print("Ticker exclude file:", exclude_file)
  
  print("Reading file...")
  
  df = pd.read_csv(input_file, names=DATASET_COLUMNS, index_col=False, skiprows = 1)
  previous_ticker=""
  print(df.head().to_string())

  if TUNE_DS_FLAG == True:
    df = tune_ds(df, TUNE_DS_FLAG, exclude_file)
  
  # Entry index from the dataset
  TYPE_POSITIVE = 1
  TYPE_NEGATIVE = 0
  CASES_TO_SHOW = 3
  TAIL_COLUMNS_LIST = get_close_target_columns_entry(MODEL)
  print("TAIL_COLUMNS_LIST:", TAIL_COLUMNS_LIST)
  for type_class in (TYPE_POSITIVE, TYPE_NEGATIVE):
    show_counter = 0
    print("\n##### EXEMPLES with classification type:", type_class)
    for index, row in df.iterrows():
      if show_counter > CASES_TO_SHOW:
        break;
      #if df.at[index, 'Type'] == type_class and previous_ticker != df.at[index, 'File']:
      if df.at[index, 'Type'] == type_class:
        i = index
        previous_ticker = df.at[index, 'File']
  
        # First elem is entry and others are target     
        ENTRY_CLMN = TAIL_COLUMNS_LIST[0]
        e_price = df.at[index, ENTRY_CLMN]
        str_out = str(1) + " - "
        for tail_id in range(len(TAIL_COLUMNS_LIST)-1):
          CLMN = TAIL_COLUMNS_LIST[tail_id+1]
          t_price = df.at[index, CLMN]
          targ_i = round(t_price/e_price,3)
          str_out+=str(targ_i)+ ", "
        print("ID", i, ":", "CLOSE: ", str_out) 

        x = []
        for ind in range(MODEL.p_days + MODEL.t_days):
          x.append(ind)
        column = 'Open'
        open_c = []
        for j in DATASET_COLUMNS:
          if j.find(column) != -1:
            real_column = j
            elem = df.at[i, real_column]
            open_c.append(elem)
        #print(column, open_c)
   
        column = 'Low'
        low_c = []
        for j in DATASET_COLUMNS:
          if j.find(column) != -1:
            real_column = j
            elem = df.at[i, real_column]
            low_c.append(elem)
        #print(column, low_c)
   
        column = 'High'
        high_c = []
        for j in DATASET_COLUMNS:
          if j.find(column) != -1:
            real_column = j
            elem = df.at[i, real_column]
            high_c.append(elem)
        #print(column, high_c)
   
        column = 'Close'
        close_c = []
        for j in DATASET_COLUMNS:
          if j.find(column) != -1:
            real_column = j
            elem = df.at[i, real_column]
            close_c.append(elem)
        #print(column, close_c)
   
        column = 'Volume'
        vol_c = []
        for j in DATASET_COLUMNS:
          if j.find(column) != -1:
            real_column = j
            elem = df.at[i, real_column]
            vol_c.append(elem)
        #print(column, vol_c)
   
        column = 'Index'
        index_c = []
        for j in DATASET_COLUMNS:
          if j.find(column) != -1:
            real_column = j
            elem = df.at[i, real_column]
            index_c.append(elem)
        #print(column, index_c)
  
        print("    V:", vol_c)
        print("Index:", index_c)
        print("    H:", high_c)
        print("    C:", close_c, "Len: ", len(close_c))
  
        if HARDWARE_MODE != 'PC':
          # plot 1
          fig = plt.figure(figsize=(12, 8))
          plt.plot(x,open_c, color='y', label = 'Open')
          plt.plot(x,high_c, color='g', label = 'High')
          plt.plot(x,low_c, color='r', label = 'Low')
          plt.plot(x,close_c, color='b', label = 'Close')
          plt.plot(x,index_c, color='k', label = 'Index')
          
          # beautify the x-labels
          diff = str(round(close_c[len(close_c)-1]/close_c[len(close_c)-2], 4))
          label_str = df.at[i, 'File'] + " Datetime: " + str(df.at[i, 'Datetime']) + " Classification flag:" + str(df.at[i, 'Type']) + ". " +str(round( close_c[len(close_c)-1], 4 )) + "/" + str(round( close_c[len(close_c)-2], 4 )) + "=" + diff 
          plt.gcf().autofmt_xdate()
          plt.title(label_str)
          plt.legend()
          plt.hlines([1], min(x), max(x), linestyle='--', color='r')
          #plt.show()
          pdf.savefig()
          plt.close(fig)
      
          #plot 2
          fig = plt.figure(figsize=(12, 8))
          plt.plot(x,vol_c, color='b', label = 'Volume chart')
          plt.gcf().autofmt_xdate()
          plt.legend()
          #ax = plt2.gca()
          #ax.set_ylim([0, 1])
          plt.hlines([1], min(x), max(x), linestyle='--', color='r')
          #plt.show()
          pdf.savefig()
          plt.close(fig)
      
        show_counter = show_counter + 1
  
        # Check normalization - max value should be in [0-6], but if it is in items [7,8,9] the warning should be raised
        flag = 0
        for i in range(0, MODEL.p_days):
          if index_c[i] == 1:
            flag = flag + 1
          if vol_c[i] == 1:
            flag = flag + 1
          if high_c[i] == 1:
            flag = flag + 1
        if flag < 3:
          print("!!!Warning: NORMALIZATION is NOT OK!!!")
  

  neg, pos = np.bincount(df['Type'])
  total = neg + pos

  brif_info = 'Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(total, pos, 100 * pos / total)
  print(brif_info)
  brif_info = df.head(10).to_string() + "\n "+ brif_info
  only_txt_into_page(pdf, "Dataset example", brif_info, 10, 8)

  return df

def preliminary_training(NEURO_NUMBER, df, train_features, train_labels, val_features, val_labels, pdf):
  model = make_model(METRICS, NEURO_NUMBER, train_features)
  model.summary()
  
  model.predict(train_features[:10])
  
  results = model.evaluate(train_features, train_labels, batch_size=BATCH_SIZE, verbose=0)
  print("Loss: {:0.4f}".format(results[0]))

  neg, pos = np.bincount(df['Type'])  
  initial_bias = np.log([pos/neg])
  initial_bias
  
  model = make_model(METRICS, NEURO_NUMBER, train_features, output_bias=initial_bias)
  model.predict(train_features[:10])
  results = model.evaluate(train_features, train_labels, batch_size=BATCH_SIZE, verbose=0)
  print("Loss: {:0.4f}".format(results[0]))
  
  initial_weights = os.path.join(tempfile.mkdtemp(), 'initial_weights')
  model.save_weights(initial_weights)
  
  print("Before model.load_weights")
  
  model = make_model(METRICS, NEURO_NUMBER, train_features)
  model.load_weights(initial_weights)
  model.layers[-1].bias.assign([0.0])
  zero_bias_history = model.fit(
      train_features,
      train_labels,
      batch_size=BATCH_SIZE,
      epochs=20,
      validation_data=(val_features, val_labels), 
      verbose=0)
  
  print("Preliminary 20 EPOCHs to be done")
  
  model = make_model(METRICS, NEURO_NUMBER, train_features)
  model.load_weights(initial_weights)
  careful_bias_history = model.fit(
      train_features,
      train_labels,
      batch_size=BATCH_SIZE,
      epochs=20,
      validation_data=(val_features, val_labels), 
      verbose=0) 
  
  plot_loss(zero_bias_history, "Zero Bias", 'g', pdf)
  plot_loss(careful_bias_history, "Careful Bias", 'b', pdf)
  
  model = make_model(METRICS, NEURO_NUMBER, train_features) 
  model.load_weights(initial_weights)
  return model


def get_model_summary(model):
    stream = StringIO()
    model.summary(print_fn=lambda x: stream.write(x + '\n'))
    summary_string = stream.getvalue()
    stream.close()
    return summary_string

def load_model(model_name, MODEL):
  print("load_model(", model_name, ", ", MODEL, ") called")
  shutil.rmtree("model_max_full", ignore_errors=True)
  shutil.rmtree("model_min_full", ignore_errors=True)
               
  shutil.unpack_archive(model_name, './', 'zip')

  if MODEL.mode == "L":
    #LONG models use this name
    load_model = "model_min_full"
  else:
    #SHORT models use this name
    load_model = "model_max_full"

  model = tf.keras.models.load_model(load_model)
  model.summary()
  return model 

# Return model file name
def intra_learn(input_file, MODEL, EPOCHS, TUNE_DS_FLAG, model_name=None):
  if TUNE_DS_FLAG == True:
    t_flag = "_T"
  else:
    t_flag = "_F"

  PRELIM_LEARN = True
  if model_name is not None:
    PRELIM_LEARN = False
    output_file = model_name[:-4] + "_E" + str(EPOCHS)
  else:
    output_file = "model_" + input_file[0:-4] + "_E" + str(EPOCHS) + t_flag
  print("PRELIM_LEARN=", PRELIM_LEARN)

  OUTPUT_PDF_FILE = output_file + ".pdf"
  pdf = PdfPages(OUTPUT_PDF_FILE)

  LEARN_COLUMNS   = get_learn_columns_full(MODEL)
  NEURO_NUMBER = (len(LEARN_COLUMNS) - 1)*(len(LEARN_COLUMNS) - 1)
  print("LEARN_COLUMNS:\n", LEARN_COLUMNS, "\nLen=", len(LEARN_COLUMNS))
  print("NEURO_NUMBER=", NEURO_NUMBER)

  print('\nInput file', input_file)
  print('Output file', output_file)
  print("OUTPUT_PDF_FILE:", OUTPUT_PDF_FILE)

  pdf_page_str = "Dataset file: " + input_file + "\n" + "LEARN_COLUMNS:\n" + str(LEARN_COLUMNS) + "\nSize:" + str(len(LEARN_COLUMNS)) + "\nNumber of neuronos: " + str(NEURO_NUMBER) + "\n"
  pdf_page_str+= "Parameters: " + str(MODEL) + "\nEPOCHs:" + str(EPOCHS)
  str_title = "Model: " + output_file
  only_txt_into_page(pdf, str_title, pdf_page_str, 10, 8)

  df = get_df_by_dataset_file(input_file, MODEL, pdf, TUNE_DS_FLAG) 

  stat_df = collect_stat_ds(df)
  time_str = "Datetime range: " + str(min(df['Datetime'])) + "-" + str(max(df['Datetime']))
  pdf_page_str = stat_df.to_string() + "\nNum of tickers:" + str(len(stat_df)) + "\nNum of all cases:" + str(len(df)) + "\n" + time_str
  only_txt_into_page(pdf, "Statistics by tickers", pdf_page_str, 10, 6)

  cleaned_df  = df[LEARN_COLUMNS].copy()

  # Use a utility from sklearn to split and shuffle your dataset.
  TEST_SIZE = 0.1 # Like it is empty. Should be like 0.2
  train_df, test_df = train_test_split(cleaned_df, test_size=TEST_SIZE)
  train_df, val_df = train_test_split(train_df, test_size=0.15)
  
  # Form np arrays of labels and features.
  train_labels = np.array(train_df.pop('Type'))
  bool_train_labels = train_labels != 0
  val_labels = np.array(val_df.pop('Type'))
  test_labels = np.array(test_df.pop('Type'))
  
  train_features = np.array(train_df)
  val_features = np.array(val_df)
  test_features = np.array(test_df)
  
  cleaned_df.describe() 

  output = StringIO()  
  print('Training labels shape:', train_labels.shape, file=output)
  print('Validation labels shape:', val_labels.shape, file=output)
  print('Test labels shape:', test_labels.shape, file=output)
  print('Training features shape:', train_features.shape, file=output)
  print('Validation features shape:', val_features.shape, file=output)
  print('Test features shape:', test_features.shape, file=output)
  result_str = output.getvalue()
  output.close()
  print(result_str)
  #print(test_features)
  
  pos_df = pd.DataFrame(train_features[ bool_train_labels], columns=train_df.columns)
  neg_df = pd.DataFrame(train_features[~bool_train_labels], columns=train_df.columns)
  
  if HARDWARE_MODE != 'PC':
    # Entry Close
    fig = plt.figure(figsize=(12, 12))
    clmn = get_entry_column(MODEL)
    sns.jointplot(pos_df[clmn],
                  kind='hex', xlim=(0.92,1), ylim=(0.92,1))
    plt.suptitle("Positive distribution")
    pdf.savefig()
    plt.close(fig) 

    fig = plt.figure(figsize=(12, 12))
    sns.jointplot(neg_df[clmn],
                  kind='hex', xlim=(0.92,1), ylim=(0.92,1))
    _ = plt.suptitle("Negative distribution")
    pdf.savefig()
    plt.close(fig) 

    # Entry Volume
    fig = plt.figure(figsize=(12, 12))
    clmn = get_entry_column_volume(MODEL)
    sns.jointplot(pos_df[clmn],
                  kind='hex', xlim=(0,1), ylim=(0,1))
    plt.suptitle("Positive distribution")
    pdf.savefig()
    plt.close(fig) 

    fig = plt.figure(figsize=(12, 12))
    sns.jointplot(neg_df[clmn],
                  kind='hex', xlim=(0,1), ylim=(0,1))
    _ = plt.suptitle("Negative distribution")
    pdf.savefig()
    plt.close(fig) 

  print("Before PRELIM_LEARN")  
  # Preliminary learning for debugging
  if model_name is not None:
    model = load_model(model_name, MODEL)
  elif PRELIM_LEARN == True:
    model = preliminary_training(NEURO_NUMBER, df, train_features, train_labels, val_features, val_labels, pdf)
  else:
    model = make_model(METRICS, NEURO_NUMBER, train_features)
    model.summary()

  result_str = result_str + "\n" + get_model_summary(model) 
  only_txt_into_page(pdf, "Model summary", result_str, 10, 8)
  # END - Preliminary learning for debugging - END #

  #START_FROM_EPOCHS = EPOCHS*0.9
  #print("If early stopping enabled START_FROM_EPOCHS=", START_FROM_EPOCHS, "of", EPOCHS)
  early_stopping = tf.keras.callbacks.EarlyStopping(
      monitor='val_prc',
      #monitor='fp', 
      verbose=1,
      patience=10,
      mode='max',
      restore_best_weights=True
      #, start_from_epoch = START_FROM_EPOCHS # Not supported now
      )

  stop_on_accuracy = myCallback()

  ### BEST weght section ###
  checkpoint_dir = "CHECKPOINTS"
  shutil.rmtree(checkpoint_dir, ignore_errors=True)
  if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir) 
  model_check_point = tf.keras.callbacks.ModelCheckpoint(
                      #filepath='CHECKPOINTS/model.{epoch:05d}-pr{precision:.5f}-re{recall:.5f}.h5',
                      filepath='CHECKPOINTS/weights_E{epoch:05d}.h5',
                      monitor='precision',
                      save_freq='epoch', verbose=0, 
                      save_weights_only=True, save_best_only=False )         
  #####
  print("model.fit() before")
  baseline_history = model.fit(
      train_features,
      train_labels,
      batch_size=BATCH_SIZE,
      epochs=EPOCHS,
      verbose=VERBOSE_MODE,
      #callbacks=[early_stopping],
      callbacks=[model_check_point, stop_on_accuracy],
      validation_data=(val_features, val_labels))

  print("model.fit() done")

  print("Restore the best weights")
  best_epoch = get_best_epoch(baseline_history)
  plot_metrics(baseline_history,pdf, best_epoch)

  best_weights = 'CHECKPOINTS/weights_E' + str(best_epoch).zfill(5) + '.h5'
  model.load_weights(best_weights)
  print(best_weights, "restored")
  
  train_predictions_baseline = model.predict(train_features, batch_size=BATCH_SIZE)
  print("model.predict(train_features, batch_size=BATCH_SIZE) done")
  test_predictions_baseline = model.predict(test_features, batch_size=BATCH_SIZE)
  print("model.predict(test_features, batch_size=BATCH_SIZE) done")

  
  for p in [0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    baseline_results = model.evaluate(test_features, test_labels,
                                      batch_size=BATCH_SIZE, verbose=0)
      
    baseline_names = model.metrics_names
    
    print('= PRED', p, '=')
    if p == 0.5:
      print(baseline_names[0], ": ", baseline_results[0])
      print(baseline_names[1], ": ", baseline_results[1])
      print(baseline_names[2], ": ", baseline_results[2])
      print(baseline_names[3], ": ", baseline_results[3])
      print(baseline_names[4], ": ", baseline_results[4])
      print(baseline_names[5], ": ", baseline_results[5])
      print(baseline_names[6], ": ", baseline_results[6])
      print(baseline_names[7], ": ", baseline_results[7])
      print(baseline_names[8], ": ", baseline_results[8])
      print(baseline_names[9], ": ", baseline_results[9])
      print()

    # TODO FIX error during drawing  
    #plot_cm(test_labels, test_predictions_baseline, p, pdf)

  ##############################
  
  if MODEL.mode == 'L':
    model_name = MODEL_NAME_LONG
  else:
    model_name = MODEL_NAME_SHORT
  model.save(model_name)
  shutil.make_archive(output_file, 'zip', ".", model_name)
  shutil.rmtree(model_name)

  pdf.close()  
  output_file = output_file + ".zip"
  print('Model done: ', output_file)
  #shutil.copy(output_file, OUT_DIR)
  return output_file


def collect_stat_ds(df):
  out_df = pd.DataFrame(columns=["File", "all_cases", "pos", "neg"])
  counter = 0
  files_list = df.File.unique()
  for ticker_file in files_list:
    ticker_df = df[(df['File'] == ticker_file)]
    ticker_pos_df = ticker_df[(ticker_df['Type'] == 1)]
    ticker_neg_df = ticker_df[(ticker_df['Type'] == 0)]
    all_len = len(ticker_df)
    pos_len = len(ticker_pos_df)
    neg_len = len(ticker_neg_df)
    out_df.loc[len(out_df)] = [ticker_file, all_len, pos_len, neg_len]

  return out_df
