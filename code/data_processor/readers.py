import json
import numpy
import random
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, IterableDataset





class AspectDetectionDataset(IterableDataset):
    def __init__(self, file, tokenizer, shuffle=True):
        self.file = file
        self.shuffle = shuffle
        self.tokenizer = tokenizer
        self.buffer_size = 5211

    def process(self, inst):
        '''
        This function is just turn the silver label to int
        '''
        if type(inst) is str:
            inst = json.loads(inst)
        inp = inst['review'] # list of sentences
#         print('current encoding review is %s' % str(inp))
        inp = [self.tokenizer.encode(sentence) for sentence in inp]
        out = [] # list of classes
        aspects = list(inst['aspects'].keys())
        for aspect in aspects:
            if inst['aspects'][aspect] == 'yes':
                out.append(1)
            else:
                out.append(0)
        # print(out)
#         print('In the AspectDetection Dataset, Process output the inp is: %s' % str(inp), 'In the AspectDetection Dataset, Process output the inp shape is: %s' % str(numpy.shape(inp)))
#         print('In the AspectDetection Dataset, Process output the out shape is: %s' %str(out), 'In the AspectDetection Dataset, Process output the out shape is: %s' % str(numpy.shape(out)))
        return inp, out # return the tokens and aspects len list with 1 and -1
    def __iter__(self):
        if self.shuffle:
            shufbuf = []
            try:
                dataset_iter = open(self.file, 'r')
                for i in range(self.buffer_size):
                    shufbuf.append(next(dataset_iter))
#                     print('shufbuf is what %s' % str(shufbuf))
            except:
                self.buffer_size = len(shufbuf)

            try:
                while True:
                    try:
#                         dataset_iter = open(self.file, 'r')
                        item = next(dataset_iter)
#                         print('what is this item %s' % str(item))
                        evict_idx = random.randint(0, self.buffer_size-1)
#                         print('what is this evict_idx %s' % str(evict_idx))
                        yield self.process(shufbuf[evict_idx])
                        shufbuf[evict_idx] = item
                    except StopIteration:
                        break
                while len(shufbuf) > 0:
                    yield self.process(shufbuf.pop())
            except GeneratorExit:
                pass

        else:
            f = open(self.file, 'r')
            for line in f:
                yield self.process(line)
            f.close()


def aspect_detection_collate(batch, mask_id=0):
    text = [inst[0] for inst in batch] # B, S, T

#     print('what is the BST ? %s' % str(type(text)), text[:2])
    max_sentence_len = max([len(sentences) for sentences in text])
#     print('what is the max sentence length ? %s' % str(max_sentence_len))
    max_token_len = min(100, max([max([len(tokens) for tokens in sentences]) for sentences in text]))
#     print('what is the max token length ? %s' % str(max_token_len))
    padded_text = []
    for sentences in text:
        padded_sentences = []
        for tokens in sentences:
            if len(tokens) < max_token_len:
                tokens = tokens + [mask_id] * (max_token_len - len(tokens))
            tokens = tokens[:max_token_len]
            assert len(tokens) == max_token_len
            padded_sentences.append(tokens)

        if len(padded_sentences) < max_sentence_len:
            zeros = [mask_id] * max_token_len
            padded_sentences = padded_sentences + [zeros] * (max_sentence_len - len(padded_sentences))
        assert len(padded_sentences) == max_sentence_len
        padded_text.append(padded_sentences)

    padded_text = torch.tensor(padded_text)
    label = [inst[1] for inst in batch]
#     print('what is inst? inst size is %s' % str(numpy.shape(batch[0])), 'what is inst? label size is %s' % str(numpy.shape(label)), label)
    label = torch.tensor(label)

#     print('In the aspect_detection_collate, Process output the padded_text shape is: the three dimension is batch size, the number of sentences that a review
# contains most sentences, the longest sentences that contain token number,
# Thus B, S, T. the second was actually sentence number instead of sentence length %s' % str(numpy.shape(padded_text)))
#     print('In the aspect_detection_collate, Process output the label shape is: %s' % str(numpy.shape(label)))
    return padded_text, label




class SummarizationDataset(IterableDataset):
    def __init__(self, file, use_keywords='input', use_switch='input', shuffle=True, shuffle_sentences=False):
        if type(file) is str:
            self.files = [file]
        elif type(file) is list:
            self.files = file
        self.use_keywords = use_keywords
        self.use_switch = use_switch
        self.shuffle = shuffle
        self.shuffle_sentences = shuffle_sentences
        self.buffer_size = 4500

    def process(self, inst, file_idx):
        tmp_dic = {7:'Pos', 8:'Neu', 9:'Neg'}
        if type(inst) is str:
            try:
                inst = json.loads(inst)
            except:
                print(inst)
                exit()
        reviews = inst['reviews']
        if self.shuffle_sentences:
            random.shuffle(reviews)
        inp = ' '.join(['<rev> ' + review for review in reviews]).lower()
        # print(inp)
        if type(inst['summary']) is list:
            inst['summary'] = inst['summary'][0]
            # print(len(inst['summary']))
        out = inst['summary'].lower()
        # print(out)
        try:
            keywords = inst['keywords'][:10]
            switch = inst['switch']
            # print(keywords)
            # print(switch)
        except:
            switch = 0
            pass
        if self.use_switch == 'input':
            switch_prompt = ['<switch>']
            for i, asp in enumerate(switch):
                switch_idx = i + file_idx*len(switch)
                if asp > 0:
                    if i < 7:
                        switch_prompt.append('<asp_%d>' % switch_idx)
                    else:
                        tmp_sent = '<' + tmp_dic[i] + '>'
                        switch_prompt.append(tmp_sent)
            inp = ' '.join(switch_prompt) + ' ' + inp
        if self.use_keywords == 'input':
            inp = '<key> ' + ' '.join(keywords) + ' ' + inp
        elif self.use_keywords == 'output':
            out = '<key> ' + ' '.join(keywords) + ' ' + out
        return inp, out, switch


    def __iter__(self):
        if self.shuffle:
            dataset_iters = [open(file, 'r') for file in self.files]
            # print(self.files)
            shufbuf = []
            file_indices = []
            # print(self.files)
            try:
                for i in range(self.buffer_size // len(self.files)):
                    for file_idx, dataset_iter in enumerate(dataset_iters):
                        item = json.loads(next(dataset_iter).strip())
                        shufbuf.append(item)
                        file_indices.append(file_idx)
                self.buffer_size = len(shufbuf)
            except:
                self.buffer_size = len(shufbuf)
            try:
                while True:
                    for i, dataset_iter in enumerate(dataset_iters):
                        try:
                            item = json.loads(next(dataset_iter).strip())
                            evict_idx = random.randint(0, self.buffer_size-1)
                            yield self.process(shufbuf[evict_idx], file_indices[evict_idx])
                            shufbuf[evict_idx] = item
                            file_indices[evict_idx] = i
                        except StopIteration:
                            dataset_iters[i].close()
                            dataset_iters[i] = open(self.files[i], 'r')
            except GeneratorExit:
                pass

            for dataset_iter in dataset_iters:
                dataset_iter.close()

        else:
            for file_idx, file in enumerate(self.files):
                f = open(file, 'r')
                for line in f:
                    yield self.process(line, file_idx)
                f.close()
