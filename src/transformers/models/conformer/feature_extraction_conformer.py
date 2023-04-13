# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Feature extractor class for Conformer
"""

from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torchaudio.functional as F

from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...utils import TensorType, logging


logger = logging.get_logger(__name__)


class ConformerFeatureExtractor(SequenceFeatureExtractor):
    model_input_names = ["input_values", "attention_mask"]

    def __init__(
        self,
        feature_size: int = 1,
        sampling_rate: int = 16000,
        n_fft: int = 400, 
        win_length: int = 400, 
        hop_length: int = 200,
        f_min: float = 0.0,
        f_max: float = None,
        pad: float = 0,
        n_mels: int = 128,
        power: float = 2,
        normalized: bool = False,
        wkwargs: Dict = None,
        center: bool = True,
        pad_mode: Optional[str] = "reflect",
        norm: Optional[str] = None,
        mel_scale: str = "htk",
        max_length: int = 1024,
        padding_value: float = 0.0,
        return_attention_mask: bool = True,
        do_normalize: bool = True,
        **kwargs
    ):
        super().__init__(feature_size=feature_size, sampling_rate=sampling_rate, padding_value=padding_value, **kwargs)
        self.max_length = max_length
        self.return_attention_mask = return_attention_mask
        self.do_normalize = do_normalize
        self.mel_spec_args = dict(
            sample_rate=sampling_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            f_min=f_min,
            f_max=f_max,
            pad=pad,
            n_mels=n_mels,
            power=power,
            normalized=normalized,
            wkwargs=wkwargs,
            center=center,
            pad_mode=pad_mode,
            norm=norm,
            mel_scale=mel_scale,
        )
    
    @staticmethod
    def mel_spectrogram(
        waveform: torch.Tensor,
        sample_rate: int = 16000,
        n_fft: int = 400, 
        win_length: int = 400, 
        hop_length: int = 200,
        f_min: float = 0.0,
        f_max: float = None,
        pad: float = 0,
        n_mels: int = 128,
        window_fn = torch.hann_window,
        power: float = 2,
        normalized: bool = False,
        wkwargs: Dict = None,
        center: bool = True,
        pad_mode: Optional[str] = "reflect",
        norm: Optional[str] = None,
        mel_scale: str = "htk",
    ) -> torch.Tensor:
        window = window_fn(win_length) if wkwargs is None else window_fn(win_length, **wkwargs)
        spectrogram = F.spectrogram(
            waveform=waveform,
            pad=pad,
            window=window,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            power=power,
            normalized=normalized,
            center=center,
            pad_mode=pad_mode,
        )
        fb = F.melscale_fbanks(
            n_freqs=n_fft // 2 + 1,
            f_min=f_min,
            f_max=f_max if f_max is not None else float(sample_rate // 2),
            n_mels=n_mels, 
            sample_rate=sample_rate,
            norm=norm,
            mel_scale=mel_scale,
        )
        mel_spectrogram = torch.matmul(spectrogram.transpose(-1, -2), fb).transpose(-1, -2)
        return mel_spectrogram

    def _extract_mel_spec_fbank_features(
        self,
        waveform: np.ndarray,
        max_length: int,
    ) -> np.ndarray:
        """
        Get mel-spectrogram features using TorchAudio. Note that TorchAudio requires 16-bit signed integers as inputs
        and hence the waveform should not be normalized before feature extraction.
        """
        waveform = torch.from_numpy(waveform).unsqueeze(0)
        specs = self.mel_spectrogram(waveform, **self.mel_spec_args)
        specs = specs.permute(0, 2, 1)
        specs = specs.squeeze()

        n_frames = specs.shape[0]
        difference = max_length - n_frames

        # pad or truncate, depending on difference
        if difference > 0:
            pad_module = torch.nn.ZeroPad2d((0, 0, 0, difference))
            specs = pad_module(specs)
            attention_mask = np.ones(n_frames, dtype=np.int32)
            attention_mask = np.pad(attention_mask, (0, difference))
        elif difference < 0:
            specs = specs[0:max_length, :]
            attention_mask = np.ones(max_length, dtype=np.int32)

        specs = specs.numpy()

        return {
            "input_values": specs,
            "attention_mask": attention_mask,
            "input_length": attention_mask.sum(-1)
        }

    @staticmethod
    def zero_mean_unit_var_norm(
        input_values: List[np.ndarray], attention_mask: List[np.ndarray], padding_value: float = 0.0
    ) -> List[np.ndarray]:
        """
        Every array in the list is normalized to have zero mean and unit variance
        """
        if attention_mask is not None:
            attention_mask = np.array(attention_mask, np.int32)
            normed_input_values = []

            for vector, length in zip(input_values, attention_mask.sum(-1)):
                normed_slice = (vector - vector[:length].mean()) / np.sqrt(vector[:length].var() + 1e-7)
                if length < normed_slice.shape[0]:
                    normed_slice[length:] = padding_value

                normed_input_values.append(normed_slice)
        else:
            normed_input_values = [(x - x.mean()) / np.sqrt(x.var() + 1e-7) for x in input_values]

        return normed_input_values

    def __call__(
        self,
        raw_speech: Union[np.ndarray, List[float], List[np.ndarray], List[List[float]]],
        sampling_rate: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ):
        if sampling_rate is not None:
            if sampling_rate != self.sampling_rate:
                raise ValueError(
                    f"The model corresponding to this feature extractor: {self} was trained using a sampling rate of"
                    f" {self.sampling_rate}. Please make sure that the provided `raw_speech` input was sampled with"
                    f" {self.sampling_rate} and not {sampling_rate}."
                )
        else:
            logger.warning(
                "It is strongly recommended to pass the `sampling_rate` argument to this function. "
                "Failing to do so can result in silent errors that might be hard to debug."
            )

        is_batched = bool(
            isinstance(raw_speech, (list, tuple))
            and (isinstance(raw_speech[0], np.ndarray) or isinstance(raw_speech[0], (tuple, list)))
        )

        if is_batched:
            raw_speech = [np.asarray(speech, dtype=np.float32) for speech in raw_speech]
        elif not is_batched and not isinstance(raw_speech, np.ndarray):
            raw_speech = np.asarray(raw_speech, dtype=np.float32)
        elif isinstance(raw_speech, np.ndarray) and raw_speech.dtype is np.dtype(np.float64):
            raw_speech = raw_speech.astype(np.float32)

        # always return batch
        if not is_batched:
            raw_speech = [raw_speech]
        
        # extract fbank features and pad/truncate to max_length
        features = [self._extract_mel_spec_fbank_features(waveform, max_length=self.max_length) for waveform in raw_speech]
        input_values = [feature["input_values"] for feature in features]
        attention_mask = [feature["attention_mask"] for feature in features]
        input_length = [feature["input_length"] for feature in features]

        # convert into BatchFeature
        padded_inputs = BatchFeature({"input_values": input_values, "attention_mask": attention_mask, "input_length": input_length})

        # make sure list is in array format
        input_values = padded_inputs.get("input_values")
        if isinstance(input_values[0], list):
            padded_inputs["input_values"] = [np.asarray(feature, dtype=np.float32) for feature in input_values]

        # convert attention_mask to correct format
        attention_mask = padded_inputs.get("attention_mask")
        if attention_mask is not None:
            padded_inputs["attention_mask"] = [np.asarray(array, dtype=np.int32) for array in attention_mask]
        
        # zero-mean and unit-variance normalization
        if self.do_normalize:
            padded_inputs["input_values"] = self.zero_mean_unit_var_norm(
                padded_inputs["input_values"], attention_mask=attention_mask, padding_value=self.padding_value
            )

        if return_tensors is not None:
            padded_inputs = padded_inputs.convert_to_tensors(return_tensors)

        return padded_inputs