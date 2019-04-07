import tempfile
import os
from fastai.text import *
import sentencepiece as spm

special_cases = [BOS, PAD, '@anonymized_account', '@url']
input_sentence_size: int = 1E7

model_filename_base = 'spm'
# the two files, describing a tokenizer, residing in a given directory
vocab_filename = f'{model_filename_base}.vocab'
model_filename = f'{model_filename_base}.model'


class SentencePieceTokenizerFunc(BaseTokenizer):
    def __init__(self, lang: str, model_path: PathOrStr):
        super().__init__(lang)
        self.lang = lang
        self.proc = spm.SentencePieceProcessor()
        self.proc.load(model_path)

    def add_special_cases(self, toks: Collection[str]):
        pass

    def tokenizer(self, t: str) -> List[str]:
        return self.proc.EncodeAsPieces(t)


def train_tokenizer(unsup_path, output_model_dir, model_type='bpe', vocab_size=30000):
    if not os.path.exists(output_model_dir):
        os.makedirs(output_model_dir)
    df = pd.read_csv(unsup_path, names=['labels', 'texts'], engine='python')

    df = df.loc[df.texts.notnull()]

    with tempfile.NamedTemporaryFile(mode='w') as raw_file:
        lines = list(df.texts)
        raw_file.write('\n'.join(lines))
        model_prefix = os.path.join(output_model_dir, model_filename_base)

        sp_params = [
            f"--input={raw_file.name}",
            f"--character_coverage=1.0",
            f"--unk_id=0",
            f"--pad_id=-1",
            f"--bos_id=-1",
            f"--eos_id=-1",
            f"--max_sentence_length=20480",
            f"--input_sentence_size={int(input_sentence_size)}",
            f"--user_defined_symbols={','.join(special_cases)}",
            f"--model_prefix={model_prefix}",
            f"--vocab_size={vocab_size} --model_type={model_type}"]
        spm.SentencePieceTrainer.Train(" ".join(sp_params))
    vocab_path = os.path.join(output_model_dir, vocab_filename)
    with open(vocab_path, 'r') as f:
        vocab = [line.split('\t')[0] for line in f.readlines()]
    pickle.dump(vocab, open(vocab_path, 'wb'))


def load_tokenizer(model_dir: PathOrStr) -> Tuple[Tokenizer, Vocab]:
    model_path = os.path.join(model_dir, model_filename)
    vocab_path = os.path.join(model_dir, vocab_filename)
    tok_factory_func = partial(SentencePieceTokenizerFunc, model_path=model_path)
    tok = Tokenizer(tok_factory_func, pre_rules=[], post_rules=[])
    vocab = Vocab(pickle.load(open(vocab_path, 'rb')))
    return tok, vocab
