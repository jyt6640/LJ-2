import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
import sys
sys.path.append('./waveglow/')
sys.path.append('./tacotron2/')
import numpy as np
import argparse
import torch
import librosa
import soundfile as sf

print(sys.path)

from tacotron2.hparams import create_hparams
from tacotron2.model import Tacotron2gi
from tacotron2.layers import TacotronSTFT, STFT
from tacotron2.audio_processing import griffin_lim
from tacotron2.text import text_to_sequence
from tacotron2.utils import load_filepaths_and_text
from waveglow.denoiser import Denoiser


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--tacotron2_ckpt_path', type=str, help='directory to save checkpoints')
    parser.add_argument('-w', '--waveglow_ckpt_path', type=str, help='directory to save checkpoints')
    args = parser.parse_args()

    hparams = create_hparams()

    checkpoint_path = args.tacotron2_ckpt_path
    model = Tacotron2(hparams)
    model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu'))['state_dict'])
    model.eval()
    
    waveglow_path = args.waveglow_ckpt_path

    waveglow = torch.load(waveglow_path, map_location=torch.device('cpu'))['model']
    waveglow.eval()
    for k in waveglow.convinv:
        k.float()
    denoiser = Denoiser(waveglow)


    abs_tc2_path = os.path.abspath(checkpoint_path)
    abs_wg_path = os.path.abspath(waveglow_path)
    tc2_num = abs_tc2_path.split('_')[-1]
    wg_num = abs_wg_path.split('_')[-1]
    audio_prefix = tc2_num + "_" + wg_num

    texts = [
        "실종 당시 회색 후드티, 남색 바지, 검은색 가방 등을 착용하고 있었다.",
        "프란치스코 교황이 방북 의사를 거듭 명확히 표명, 기대감이 향하는 모습이다.",
        "고객님, 박병호 님의 성함으로 예약 확인이 되지 않았습니다.",
        "말레이시아가 세계 삼대 반딧불 서식지인 만큼 반딧불 투어를 갈겁니다.",
    ]

    dir_name = "./inference_output"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    for i in range(len(texts)):
        sequence = np.array(text_to_sequence(texts[i], 
                                            ['hangul_cleaners']))[None, :]
        sequence = torch.autograd.Variable(
            torch.from_numpy(sequence)).cpu().long()

        mel, mel_postnet, _, alignment = model.inference(sequence)

        with torch.no_grad():
            audio = waveglow.infer(mel_postnet, sigma=0.666)

        audio_denoised = denoiser(audio, strength=0.01)[:, 0]

        sf.write(
            '{}/{}_{}.wav'.format(dir_name, audio_prefix, i),
            audio_denoised.cpu().numpy().T,
            hparams.sampling_rate
        )
