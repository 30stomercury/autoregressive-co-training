import numpy as np
from dataflow.kaldiark import parse_feat_matrix

class get_fbank_scp:
    """
    LibriSpeech:
        - data/librispeech/ext-data/train-clean-{100/360/960}.scp
    
    WSJ:
        - data/wsj/ext-data/si284-0.9-{train/dev}.fbank.scp
        - data/wsj/ext-data/dev93.fbank.scp
        - data/wsj/ext-data/eval92.fbank.scp
    """

    def __init__(self, file):
        # For memeory issue, we do not load all mat here
        scps, names = self.parser(file)
        self.scps = scps
        self.names = names

    def __getitem__(self, index):
        scp_id = self.scps[index]
        name, ark_offset = scp_id.split()
        ark, offset = ark_offset.split(':')
        # read features
        f = open(ark, 'rb')
        f.seek(int(offset))
        mat = parse_feat_matrix(f)
        f.close()

        return name, mat

    def parser(self, file):
        scps = []
        names = []
        mats = []
        # For memeory issue, we do not load all mat here
        with open(file) as f:
            for scp_id in f:
                name, ark_offset = scp_id.split()
                ark, offset = ark_offset.split(':')
                # read features
                f_ = open(ark, 'rb')
                f_.seek(int(offset))
                # collect elements
                scps.append(scp_id)
                names.append(name)
        f.close()

        return scps, names


class get_bstring:
    """Used in phn classification.    

    WSJ 
        - data/wsj/ext-data/dev93.bpali
        - data/wsj/ext-data/eval92.bpali
    """

    def __init__(self, file, tok_file):
        f = open(file)
        self.lines = f.readlines()
        f.close()

        self.tok_2_int, self.int_2_tok = token_list(tok_file)

        self.seqs = {}
        for i, line in enumerate(self.lines):
            if i % 3 == 0:
                curr = line.split()[0]

            if i % 3 == 1:
                orig_seq = [self.tok_2_int[x] for x in line.split()]
                self.seqs[curr] = orig_seq

            if i % 3 == 2 and line.split()[0] != '.':
                print('Align fault')

    def __getitem__(self, name):
        return self.seqs[name]


class get_bstring_scp:
    """Used in phn classification.
    WSJ:
        - data/wsj/ext-data/si284-0.9-{train/dev}.bpali.scp
        - data/wsj/ext-data/dev93.bpali
        - data/wsj/ext-data/eval92.bpali
    """

    def __init__(self, file, tok_file):
        self.f_prefix = file.rsplit('/', 2)[0] + '/'
        self.tok_2_int, self.int_2_tok = token_list(tok_file)
        scps, seqs = self.parser(file)
        self.seqs = seqs

    def __getitem__(self, name):
        seq = self.seqs[name]

        return seq

    def parser(self, file):
        scps = []
        seqs = {}
        with open(file) as f:
            for scp_id in f:
                name, ark_offset = scp_id.split()
                ark, offset = ark_offset.split(':')
                ark = self.f_prefix + ark
                # read features
                f_ = open(ark, 'r')
                f_.seek(int(offset))
                mat = f_.readline()
                seq = [self.tok_2_int[x] for x in mat.split()]
                seqs[name] = seq
                scps.append(scp_id)
        f.close()

        return scps, seqs


def token_list(file):
    f = open(file)
    lines = f.readlines()
    f.close()
    
    tok2int = {}
    int2tok = {}
    for i, l in enumerate(lines):
        int2tok[i] = l.split()[0].lower()
    int2tok[i+1] = "sil"
    int2tok[i+2] = "<spn>"
    int2tok[i+3] = "<noise>"
    for i in range(len(int2tok)):
        tok2int[int2tok[i]] = i
    return tok2int, int2tok
