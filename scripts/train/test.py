from streaming import StreamingDataset, MDSWriter
import numpy as np
from transformers import AutoTokenizer
from llmfoundry import ConcatTokensDataset


# class MyDataset:
#     def __iter__(self):
#         for i in range(5):
#             yield {'text': 'HELLO WORLD'}



# dataset = ConcatTokensDataset(
#     hf_dataset=MyDataset(),
#     tokenizer=tokenizer,
#     max_length=1,
#     bos_text='',
#     eos_text='',
#     no_wrap=False
# )

from streaming import StreamingDataset, MDSWriter
import numpy as np
from transformers import AutoTokenizer

columns = {'tokens': 'bytes'}
tokenizer = AutoTokenizer.from_pretrained('mosaicml/mpt-7b')

print('Converting to MDS format...')
total_tokens = 0
with MDSWriter(out='out',
                columns=columns,
                compression=None) as out:
    tokens = tokenizer('HELLO WORLD',truncation=False,padding=False)['input_ids']
    print(tokens)
    for i in range(5): 
        out.write({'tokens': np.asarray(tokens).tobytes() })
        total_tokens += len(tokens)
print('mds writer tokens', total_tokens) # prints 25 tokens (each 'HELLO WORLD' is 5 tokens)

print('Iterating through samples')

dataset = StreamingDataset(
    local='out',
    shuffle=False)

print('dataset size', dataset.num_samples)
tokens_bytes = 0
for sample in dataset:
    tokens_bytes += len(sample['tokens'])
    print('sample', tokenizer.decode(np.frombuffer(sample['tokens'], dtype=int)))
print('streaming dataset tokens', tokens_bytes/8) # prints 260 tokens


    # for sample in dataset:
    #     print(sample, len(sample['tokens']))
    #     out.write(sample)
    #     total_tokens += len(sample['tokens'])