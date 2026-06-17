import os, sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, '../../../..'))
_DEEPASV_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, '../..'))

sys.path.insert(0, _DEEPASV_DIR)
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, 'deeplab/pretrained/audio2vector/module/transformers/src'))

import argparse
import pickle
import numpy as np
import torch
import torch.utils.data
from tqdm import tqdm

from deeplab.utils.fileio import read_hyperyaml, load_audio


def find_audio_files(data_root, exts=None):
    if exts is None:
        exts = ('.wav', '.flac', '.mp3')
    files = []
    for dirpath, _, filenames in os.walk(data_root):
        for fn in filenames:
            if fn.lower().endswith(exts):
                files.append(os.path.join(dirpath, fn))
    return sorted(files)


def parse_speaker_utt(file_path, data_root):
    rel = os.path.relpath(file_path, data_root)
    parts = rel.replace('\\', '/').split('/')
    speaker_id = parts[-2]
    utt_name = os.path.splitext(parts[-1])[0]
    return speaker_id, utt_name


class AudioInferenceDataset(torch.utils.data.Dataset):
    def __init__(self, file_paths, data_root, sr, max_len):
        self.file_paths = file_paths
        self.data_root = data_root
        self.sr = sr
        self.max_len = max_len

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        audio_path = self.file_paths[idx]
        signal = load_audio(audio_path, self.sr)[0][:self.max_len]
        signal = torch.from_numpy(signal.astype(np.float32))
        speaker_id, utt_name = parse_speaker_utt(audio_path, self.data_root)
        return audio_path, speaker_id, utt_name, signal


def load_model_and_config(ckpt_path, yaml_path, device):
    ckpt_data = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    hparams = read_hyperyaml(path=yaml_path)
    modules = hparams['modules']

    for key, module in modules.items():
        if key == 'classifier':
            continue
        if key not in ckpt_data['modules']:
            print(f'      {key}: <Not found in checkpoint, keeping init weights>')
            module = module.eval().to(device)
            continue
        curr_state_dict = module.state_dict()
        ckpt_state_dict = ckpt_data['modules'][key]
        mismatched = False
        for k in curr_state_dict.keys():
            if k in ckpt_state_dict and curr_state_dict[k].shape == ckpt_state_dict[k].shape:
                curr_state_dict[k] = ckpt_state_dict[k]
            else:
                mismatched = True
        module.load_state_dict(curr_state_dict)
        module = module.eval().to(device)
        if mismatched:
            print(f'      {key}: <Partial weights matched>')
        else:
            print(f'      {key}: <All weights matched>')
    return modules, hparams


def main():
    parser = argparse.ArgumentParser(description='Extract w2v-bert speaker embeddings from audio files')
    parser.add_argument('--data_root', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--train_yaml', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--skip_existing', action='store_true')
    parser.add_argument('--random_shuffle', action='store_true')
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--max_files', type=int, default=0)
    parser.add_argument('--file_stride', type=int, default=1)
    parser.add_argument('--file_offset', type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f'Loading model from: {args.checkpoint}')
    print(f'Loading config from: {args.train_yaml}')
    modules, hparams = load_model_and_config(args.checkpoint, args.train_yaml, args.device)

    sr = hparams['sample_rate']
    max_len = int(hparams['max_valid_dur'] * sr) if 'max_valid_dur' in hparams else sr * 60
    dtype = torch.bfloat16 if hparams.get('use_amp', True) else torch.float32

    spk_model = modules['spk_model']

    audio_files = find_audio_files(args.data_root)
    print(f'Found {len(audio_files)} total audio files')

    if args.random_shuffle:
        rng = np.random.RandomState(args.random_seed)
        rng.shuffle(audio_files)

    if args.max_files > 0:
        audio_files = audio_files[:args.max_files]

    audio_files = audio_files[args.file_offset::args.file_stride]
    print(f'GPU shard [{args.file_offset}/{args.file_stride}]: {len(audio_files)} files '
          f'(num_workers={args.num_workers})')

    dataset = AudioInferenceDataset(audio_files, args.data_root, sr, max_len)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    processed = 0
    skipped = 0

    for audio_path, speaker_id, utt_name, aud_inputs in tqdm(loader, desc=f'GPU{args.file_offset}', ncols=100):
        audio_path = audio_path[0]
        speaker_id = speaker_id[0]
        utt_name = utt_name[0]
        aud_inputs = aud_inputs.to(args.device, non_blocking=True)

        out_path = os.path.join(args.output_dir, speaker_id, f'{utt_name}.pkl')
        if args.skip_existing and os.path.exists(out_path):
            skipped += 1
            continue

        with torch.autocast('cuda', dtype=dtype):
            with torch.no_grad():
                embedding = spk_model(aud_inputs)
                if len(embedding.shape) == 3:
                    embedding = embedding[:, -1, :]
                embedding = embedding.float().detach().cpu().numpy().squeeze(0)

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, 'wb') as f:
            pickle.dump(embedding, f)
        processed += 1

    print(f'\nDone. Processed: {processed}, Skipped: {skipped}, Total files: {len(audio_files)}')
    print(f'Embeddings saved to: {args.output_dir}')


if __name__ == '__main__':
    main()
