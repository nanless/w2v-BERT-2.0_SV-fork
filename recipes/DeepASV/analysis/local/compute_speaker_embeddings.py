import os
import sys
import argparse
import pickle
import glob
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


def find_speaker_dirs(utterances_dir):
    dirs = sorted([
        d for d in glob.glob(os.path.join(utterances_dir, '*'))
        if os.path.isdir(d)
    ])
    return dirs


def count_utterances(speaker_dir, exclude_prefix=None, exclude_pattern=None):
    files = glob.glob(os.path.join(speaker_dir, '*.pkl'))
    basenames = [os.path.basename(f) for f in files]

    if exclude_prefix:
        basenames = [b for b in basenames if not b.startswith(exclude_prefix)]
    if exclude_pattern:
        import fnmatch
        basenames = [b for b in basenames if not fnmatch.fnmatch(b, exclude_pattern)]

    return len(basenames)


def compute_speaker_embedding(args_pack):
    speaker_dir, speakers_dir, min_utterances, skip_existing, exclude_prefix, exclude_pattern = args_pack

    speaker_id = os.path.basename(speaker_dir)
    out_path = os.path.join(speakers_dir, speaker_id, f'{speaker_id}.pkl')

    if skip_existing and os.path.exists(out_path):
        return {'speaker_id': speaker_id, 'status': 'skipped', 'num_utterances': 0}

    import fnmatch
    embedding_files = sorted(glob.glob(os.path.join(speaker_dir, '*.pkl')))
    basenames = [os.path.basename(f) for f in embedding_files]

    if exclude_prefix:
        embedding_files = [f for f, b in zip(embedding_files, basenames) if not b.startswith(exclude_prefix)]
        basenames = [b for b in basenames if not b.startswith(exclude_prefix)]
    if exclude_pattern:
        embedding_files = [f for f, b in zip(embedding_files, basenames) if not fnmatch.fnmatch(b, exclude_pattern)]

    num_utterances = len(embedding_files)
    if num_utterances < min_utterances:
        return {'speaker_id': speaker_id, 'status': 'too_few', 'num_utterances': num_utterances}

    embeddings = []
    for emb_file in embedding_files:
        try:
            with open(emb_file, 'rb') as f:
                emb = pickle.load(f)
            embeddings.append(emb)
        except Exception as e:
            tqdm.write(f'Error loading {emb_file}: {e}')

    if len(embeddings) == 0:
        return {'speaker_id': speaker_id, 'status': 'load_error', 'num_utterances': 0}

    # average all utterance embeddings for this speaker
    avg_embedding = np.mean(np.stack(embeddings, axis=0), axis=0)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'wb') as f:
        pickle.dump(avg_embedding.astype(np.float32), f)

    return {'speaker_id': speaker_id, 'status': 'ok', 'num_utterances': num_utterances}


def main():
    parser = argparse.ArgumentParser(description='Compute speaker-level embeddings by averaging utterance embeddings')
    parser.add_argument('--utterances_dir', required=True, help='Directory of utterance embeddings (output of step1)')
    parser.add_argument('--speakers_dir', required=True, help='Output directory for speaker embeddings')
    parser.add_argument('--min_utterances', type=int, default=1, help='Minimum utterances required per speaker')
    parser.add_argument('--num_processes', type=int, default=0, help='Number of processes (0 = auto)')
    parser.add_argument('--chunk_size', type=int, default=10, help='Chunk size for parallel processing')
    parser.add_argument('--skip_existing', action='store_true', help='Skip speakers that already have embeddings')
    parser.add_argument('--exclude_filename_prefix', default=None, help='Exclude files whose basename starts with this prefix')
    parser.add_argument('--exclude_filename_pattern', default=None, help='Exclude files matching this glob pattern')
    args = parser.parse_args()

    if args.num_processes <= 0:
        args.num_processes = cpu_count()
    print(f'Using {args.num_processes} processes')

    os.makedirs(args.speakers_dir, exist_ok=True)

    speaker_dirs = find_speaker_dirs(args.utterances_dir)
    print(f'Found {len(speaker_dirs)} speaker directories')

    total_utterances = sum(
        count_utterances(d, args.exclude_filename_prefix, args.exclude_filename_pattern)
        for d in speaker_dirs
    )
    print(f'Total utterance files: {total_utterances}')

    task_args = [
        (sd, args.speakers_dir, args.min_utterances, args.skip_existing,
         args.exclude_filename_prefix, args.exclude_filename_pattern)
        for sd in speaker_dirs
    ]

    results = []
    with Pool(processes=args.num_processes) as pool:
        for result in tqdm(pool.imap_unordered(compute_speaker_embedding, task_args, chunksize=args.chunk_size),
                           total=len(task_args), desc='Computing speaker embeddings', ncols=100):
            results.append(result)

    ok = [r for r in results if r['status'] == 'ok']
    skipped = [r for r in results if r['status'] == 'skipped']
    too_few = [r for r in results if r['status'] == 'too_few']
    errors = [r for r in results if r['status'] == 'load_error']

    print(f'\nDone. OK: {len(ok)}, Skipped: {len(skipped)}, Too few utts: {len(too_few)}, Errors: {len(errors)}')

    if ok:
        total_utts_used = sum(r['num_utterances'] for r in ok)
        print(f'Average utterances per speaker: {total_utts_used / len(ok):.1f}')


if __name__ == '__main__':
    main()
