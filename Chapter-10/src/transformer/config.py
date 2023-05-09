import transformers

MAX_LEN = 512

TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
EPOCHS = 10
BERT_PATH = 'Chapter-10/input/bert_base_uncased/'
MODEL_PATH = "model.bin"
TRAINING_FILE = "Chapter-10/input/imdb.csv"
TOKENIZER = transformers.BertTokenizer.from_pretrained(
     BERT_PATH, do_lower_case=True)
