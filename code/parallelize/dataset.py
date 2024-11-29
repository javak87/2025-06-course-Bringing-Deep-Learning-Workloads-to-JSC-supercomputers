import re
from torch.utils.data import Dataset
from datasets import load_dataset

class Xsum(Dataset):
    def __init__(self, tokenizer, type_path, num_samples, input_length, output_length):
        self.dataset = load_dataset("EdinburghNLP/xsum", split=type_path)
        if num_samples:
            self.dataset = self.dataset.select(list(range(0, num_samples)))
        self.input_length = input_length
        self.tokenizer = tokenizer
        self.output_length = output_length

    def __len__(self):
        return self.dataset.shape[0]

    def clean_text(self, text):
        text = text.replace('Example of text:', '')
        text = text.replace('Example of Summary:', '')
        text = text.replace('\n','')
        text = text.replace('``', '')
        text = text.replace('"', '')

        return text


    def convert_to_features(self, example_batch):
        # Tokenize contexts and questions (as pairs of inputs)

        input_ = self.clean_text(example_batch['document'])
        target_ = self.clean_text(example_batch['summary'])

        source = self.tokenizer.batch_encode_plus([input_], max_length=self.input_length,
                                                     padding='max_length', truncation=True, return_tensors="pt")

        targets = self.tokenizer.batch_encode_plus([target_], max_length=self.output_length,
                                                     padding='max_length', truncation=True, return_tensors="pt")


        return source, targets

    def __getitem__(self, index):
        source, targets = self.convert_to_features(self.dataset[index])

        source_ids = source["input_ids"].squeeze()
        target_ids = targets["input_ids"].squeeze()

        src_mask    = source["attention_mask"].squeeze()
        target_mask = targets["attention_mask"].squeeze()

        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask}

