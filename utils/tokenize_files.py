import spacy
import os
import logging
nlp = spacy.load("en")

def generalfiletokinzer(fpath, fout):
    global nlp

    with open(fout,'w',encoding='utf-8') as fout:
        logging.info(f"Working on {fpath}")
        for line in open(fpath, 'r',encoding='utf-8'):
            sent = line.strip()
            doc = nlp.tokenizer(sent)
            tokens = [token.text for token in doc]
            fout.write(" ".join(tokens) + "\n")


def local_tokenizer(doc):
    #doc = ''.join(x.rstrip('\n') for x in doc)
    #doc = sent_tokenize(doc)
    tok_doc = []
    #for sent in doc:
    sent = nlp.tokenizer(doc)
    tokens = [token.text for token in sent]
    sent = " ".join(tokens) + "\n"
    tok_doc.append(sent)
    return tok_doc

#src_file = "/data_local/src/gector-master/new_data/wi_nl_fc_en_clang8_split/wi.nl.fc.easynote.clang8.train.ori"
#cor_file = "/data_local/src/gector-master/new_data/wi_nl_fc_en_clang8_split/wi.nl.fc.easynote.clang8.train.cor"
#/data_local/src/gector-master/new_data/wi_nl_fc_en_clang8_split/

new_raw_data_path = "/data_local/src/gector-master/new_data/2021_11_19/raw/"
new_tok_data_path = "/data_local/src/gector-master/new_data/2021_11_19/new_tok"

raw_files = os.listdir(new_raw_data_path)
raw_files = [f for f in raw_files if f.endswith(('ori','cor'))]
os.makedirs(new_tok_data_path, exist_ok=True)


for f in raw_files:
    f1= f.split(".")
    ext = f1[-1]
    f1 = f1[:-1]
    new_f = '.'.join(f1) + '.tok.' + 'ori'
    f_tok = os.path.join(new_tok_data_path, new_f)

    if os.path.exists(f_tok):
        logging.info(f"skip this step as {f_tok} already exists")
    else:
        generalfiletokinzer(os.path.join(new_raw_data_path,f), f_tok)
    print(f"Done the file {f}")

