# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import copy
import math
import os
import tempfile
from argparse import ArgumentParser, Namespace
from concurrent.futures import ProcessPoolExecutor
from glob import glob
from typing import Iterable, List, Tuple, cast

from composer.utils import (ObjectStore, maybe_create_object_store_from_uri,
                            parse_uri)
from streaming import MDSWriter
from tqdm import tqdm
from transformers import AutoTokenizer

from llmfoundry.data import ConcatTokensDataset
from llmfoundry.utils.data_prep_utils import (DownloadingIterable,
                                              merge_shard_groups)


def parse_args() -> Namespace:
    """Parse commandline arguments."""
    parser = ArgumentParser(
        description=
        'Convert text files into MDS format, optionally concatenating and tokenizing'
    )
    parser.add_argument(
        '--max_mds_writer_workers',
        type=int,
        default=64,
        help='The maximum number of workers to use for MDS writing')
    parser.add_argument('--output_folder',
                        type=str,
                        required=True,
                        help='The folder to write output to')
    parser.add_argument('--input_folder',
                        type=str,
                        required=True,
                        help='The folder with text files to convert to mds')
    parser.add_argument('--compression',
                        type=str,
                        default='zstd',
                        help='The compression algorithm to use for MDS writing')

    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        '--concat_tokens',
        type=int,
        help='Convert text to tokens and concatenate up to this many tokens')

    parser.add_argument('--tokenizer',
                        type=str,
                        help='The name of the tokenizer to use')
    parser.add_argument(
        '--bos_text',
        type=str,
        required=False,
        default=None,
        help=
        'The text to prepend to each example to separate concatenated examples')
    parser.add_argument(
        '--eos_text',
        type=str,
        required=False,
        default=None,
        help=
        'The text to append to each example to separate concatenated examples')
    parser.add_argument(
        '--no_wrap',
        default=False,
        action='store_true',
        help=
        'Whether to let text examples wrap across multiple training examples')
    parser.add_argument(
        '--processes',
        type=int,
        required=False,
        default=1,
        help='The number of processes to use to download and convert the dataset'
    )
    parser.add_argument(
        '--reprocess',
        type=bool,
        required=False,
        default=False,
        help=
        'If true, reprocess the input_folder to mds format. Otherwise, only reprocess upon changes to the input folder.'
    )

    parsed = parser.parse_args()

    # Make sure we have needed concat options
    if (parsed.concat_tokens is not None and
            isinstance(parsed.concat_tokens, int) and parsed.tokenizer is None):
        parser.error(
            'When setting --concat_tokens, you must specify a --tokenizer')

    # now that we have validated them, change BOS/EOS to strings
    if parsed.bos_text is None:
        parsed.bos_text = ''
    if parsed.eos_text is None:
        parsed.eos_text = ''
    return parsed


def get_object_names(input_folder: str) -> List[str]:
    object_store = maybe_create_object_store_from_uri(input_folder)
    if object_store is not None:
        _, _, folder_prefix = parse_uri(input_folder)
        names = [
            name for name in object_store.list_objects(folder_prefix)
            if name.endswith('.txt')
        ]
    else:
        # input_folder is a local folder
        names = [
            text_file for dirpath, _, _ in os.walk(input_folder)
            for text_file in glob(os.path.join(dirpath, '*.txt'))
        ]
    # return names, sizes
    print(f'Found {len(names)} text files at {input_folder}')

    return names


def get_task_args(
    object_names: List[str],
    output_root: str,
    input_folder: str,
    n_groups: int,
    tokenizer_name: str,
    concat_tokens: int,
    eos_text: str,
    bos_text: str,
    no_wrap: bool,
    compression: str,
    max_mds_writer_workers: int,
) -> Iterable:
    num_objects = len(object_names)
    objs_per_group = math.ceil(num_objects / n_groups)
    for group, i in enumerate(range(0, num_objects, objs_per_group)):
        output_subdir = os.path.join(output_root, str(group))
        yield (
            object_names[i:min(i + objs_per_group, num_objects)],
            output_subdir,
            input_folder,
            tokenizer_name,
            concat_tokens,
            eos_text,
            bos_text,
            no_wrap,
            compression,
            max_mds_writer_workers,
        )


def download_and_convert_starargs(args: Tuple):
    return download_and_convert(*args)


def download_and_convert(
    file_names: List[str],
    output_folder: str,
    input_folder: str,
    tokenizer_name: str,
    concat_tokens: int,
    eos_text: str,
    bos_text: str,
    no_wrap: bool,
    compression: str,
    max_mds_writer_workers: int,
):
    object_store = maybe_create_object_store_from_uri(input_folder)

    # Download file_names
    with tempfile.TemporaryDirectory() as tmp_dir:
        downloading_iter = DownloadingIterable(object_names=file_names,
                                               output_folder=tmp_dir,
                                               object_store=object_store)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        tokenizer.model_max_length = 5000000000  # Hack to prevent warnings from HuggingFace

        # Use the ConcatTokensDataset from LLM-foundry to concatenate sequences of tokens up to the maximum sequence length
        dataset = ConcatTokensDataset(
            hf_dataset=downloading_iter,
            max_length=concat_tokens,
            tokenizer=tokenizer,
            eos_text=eos_text,
            bos_text=bos_text,
            no_wrap=no_wrap,
        )

        columns = {'tokens': 'bytes'}

        print(f'Converting to MDS format...')
        total_tokens_bytes = 0
        with MDSWriter(out=output_folder,
                       columns=columns,
                       max_mds_writer_workers=max_mds_writer_workers,
                       compression=compression) as out:
            for sample in tqdm(dataset):
                total_tokens_bytes += len(sample['tokens'])
                out.write(sample)
        total_tokens = total_tokens_bytes / 8
        print('tokens', total_tokens_bytes, total_tokens)
        return total_tokens


def is_remote_path(path: str) -> bool:
    backend, bucket, _ = parse_uri(path)
    return backend != '' and bucket != ''


def is_already_processed(output_root: str, done_file_name: str, args_str: str,
                         object_names: List[str]) -> bool:
    # Retrieve the done file contents
    output_object_store = maybe_create_object_store_from_uri(output_root)
    if output_object_store is not None:
        # Download and read the done file from the remote object store
        _, _, output_folder_prefix = parse_uri(output_root)
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                done_file = os.path.join(tmp_dir, done_file_name)
                output_object_store.download_object(
                    os.path.join(output_folder_prefix, done_file_name),
                    done_file)
                with open(done_file) as df:
                    done_file_contents = df.read().splitlines()
        except FileNotFoundError:
            return False
    else:
        # Read the local done file
        done_file = os.path.join(output_root, done_file_name)
        if not os.path.isfile(done_file):
            return False
        with open(done_file) as df:
            done_file_contents = df.read().splitlines()
    # Compare the arguments
    prev_args_str = done_file_contents[0]
    if prev_args_str != args_str:
        return False

    # Compare file names
    prev_names = done_file_contents[1:]
    if len(prev_names) != len(object_names):
        return False
    for idx, prev_name in enumerate(prev_names):
        if object_names[idx] != prev_name:
            return False
    return True


def write_done_file(folder: str, file_name: str, args_str: str,
                    object_names: List[str]):
    with open(os.path.join(folder, file_name), 'w') as done_file:
        done_file.write('\n'.join([args_str] + object_names) + '\n')


def get_done_file_name() -> str:
    return '.text_to_mds_conversion_done'


def main(
    tokenizer_name: str,
    output_folder: str,
    input_folder: str,
    concat_tokens: int,
    eos_text: str,
    bos_text: str,
    no_wrap: bool,
    max_mds_writer_workers: int,
    compression: str,
    processes: int,
    args_str: str,
    reprocess: bool,
):
    done_file_name = get_done_file_name()
    is_remote_output = is_remote_path(output_folder)

    object_names = get_object_names(input_folder)

    # Check if the text files in the bucket have already been processed.
    if not reprocess and is_already_processed(output_folder, done_file_name,
                                              args_str, object_names):
        print(
            f'Input folder {input_folder} is already processed at {output_folder} and reprocess is set to False. Set reprocess to True if you would like to force reprocessing.'
        )
        return

    # Use a temporary local directory if the output is remote and there are more than 1 processes
    local_output_folder = tempfile.TemporaryDirectory(
    ).name if is_remote_output else output_folder

    if processes > 1:
        # Download and convert the text files in parallel
        args = get_task_args(object_names, local_output_folder, input_folder,
                             processes, tokenizer_name, concat_tokens, eos_text,
                             bos_text, no_wrap, compression,
                             max_mds_writer_workers)
        with ProcessPoolExecutor(max_workers=processes) as executor:
            print('all tokens',
                  sum(executor.map(download_and_convert_starargs, list(args))))

        # Merge the mds shards from each of the processes into a single folder
        merge_shard_groups(local_output_folder)
    else:
        download_and_convert(object_names, local_output_folder, input_folder,
                             tokenizer_name, concat_tokens, eos_text, bos_text,
                             no_wrap, compression, max_mds_writer_workers)

    # Write a done file with the args and object names
    write_done_file(local_output_folder, done_file_name, args_str, object_names)

    if is_remote_output:
        # Upload the local output to the remote location
        output_object_store = cast(
            ObjectStore, maybe_create_object_store_from_uri(output_folder))
        _, _, output_folder_prefix = parse_uri(output_folder)
        files_to_upload = os.listdir(local_output_folder)

        for file in files_to_upload:
            assert not os.path.isdir(file)
            remote_path = os.path.join(output_folder_prefix, file)
            output_object_store.upload_object(
                remote_path, os.path.join(local_output_folder, file))


def _args_str(original_args: Namespace) -> str:
    """Create a string from the args to determine whether to reprocess.

    Args:
        original_args (Namespace): args to transform
    """
    args = copy.deepcopy(original_args)

    # Remove args that do not affect the final result.
    delattr(args, 'max_mds_writer_workers')
    delattr(args, 'reprocess')
    return str(args)


if __name__ == '__main__':
    args = parse_args()
    main(tokenizer_name=args.tokenizer,
         output_folder=args.output_folder,
         input_folder=args.input_folder,
         concat_tokens=args.concat_tokens,
         eos_text=args.eos_text,
         bos_text=args.bos_text,
         no_wrap=args.no_wrap,
         max_mds_writer_workers=args.max_mds_writer_workers,
         compression=args.compression,
         processes=args.processes,
         reprocess=args.reprocess,
         args_str=_args_str(args))
