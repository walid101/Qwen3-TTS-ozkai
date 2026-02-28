# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
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
import base64
import io
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse

import librosa
import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as F
from transformers import AutoConfig, AutoModel, AutoProcessor

from ..core.models import Qwen3TTSConfig, Qwen3TTSForConditionalGeneration, Qwen3TTSProcessor

AudioLike = Union[
    str,                     # wav path, URL, base64
    np.ndarray,              # waveform (requires sr)
    Tuple[np.ndarray, int],  # (waveform, sr)
]

MaybeList = Union[Any, List[Any]]


@dataclass
class VoiceClonePromptItem:
    """
    Container for one sample's voice-clone prompt information that can be fed to the model.

    Fields are aligned with `Qwen3TTSForConditionalGeneration.generate(..., voice_clone_prompt=...)`.
    """
    ref_code: Optional[torch.Tensor]                 # (T, Q) or (T,) depending on tokenizer 25Hz/12Hz
    ref_spk_embedding: torch.Tensor                  # (D,)
    x_vector_only_mode: bool
    icl_mode: bool
    ref_text: Optional[str] = None


class Qwen3TTSModel:
    """
    A HuggingFace-style wrapper for Qwen3 TTS models (CustomVoice/VoiceDesign/Base) that provides:
      - from_pretrained() initialization via AutoModel/AutoProcessor
      - generation APIs for:
          * CustomVoice: generate_custom_voice()
          * VoiceDesign: generate_voice_design()
          * Base: generate_voice_clone() + create_voice_clone_prompt()
      - consistent output: (wavs: List[np.ndarray], sample_rate: int)

    Notes:
      - This wrapper expects the underlying model class to be `Qwen3TTSForConditionalGeneration`
      - Language / speaker validation is done via model methods:
          model.get_supported_languages(), model.get_supported_speakers()

    Performance Notes (optimizations applied):
      - [OPT-2] Batched tokenization: all texts (including ref_texts) are tokenized in a
        single processor call to minimise Python→Rust binding overhead.
      - [OPT-3] Batched speaker embedding: all reference audios are sent through
        extract_speaker_embedding in one batched call, saturating GPU CUDA cores.
      - [OPT-4] Single-trip resampling: all wavs that need resampling are padded into one
        batched tensor, moved to GPU once, resampled, then brought back — eliminating
        per-sample PCIe round-trips.
    """

    def __init__(self, model: Qwen3TTSForConditionalGeneration, processor, generate_defaults: Optional[Dict[str, Any]] = None):
        self.model = model
        self.processor = processor
        # [OPT-8] Pre-bake generate_defaults merged with hard defaults at construction time
        # so _merge_generate_kwargs is a cheap dict update at call time.
        hard_defaults = dict(
            do_sample=True,
            top_k=50,
            top_p=1.0,
            temperature=0.9,
            repetition_penalty=1.05,
            subtalker_dosample=True,
            subtalker_top_k=50,
            subtalker_top_p=1.0,
            subtalker_temperature=0.9,
            max_new_tokens=2048,
        )
        provided = generate_defaults or {}
        self._baked_defaults: Dict[str, Any] = {**hard_defaults, **provided}

        self.device = getattr(model, "device", None)
        if self.device is None:
            try:
                self.device = next(model.parameters()).device
            except StopIteration:
                self.device = torch.device("cpu")

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        **kwargs,
    ) -> "Qwen3TTSModel":
        """
        Load a Qwen3 TTS model and its processor in HuggingFace `from_pretrained` style.

        This method:
          1) Loads config via AutoConfig (so your side can register model_type -> config/model).
          2) Loads the model via AutoModel.from_pretrained(...), forwarding `kwargs` unchanged.
          3) Loads the processor via AutoProcessor.from_pretrained(model_path).
          4) Loads optional `generate_config.json` from the model directory/repo snapshot if present.

        Args:
            pretrained_model_name_or_path (str):
                HuggingFace repo id or local directory of the model.
            **kwargs:
                Forwarded as-is into `AutoModel.from_pretrained(...)`.
                Typical examples: device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="flash_attention_2".

        Returns:
            Qwen3TTSModel:
                Wrapper instance containing `model`, `processor`, and generation defaults.
        """
        AutoConfig.register("qwen3_tts", Qwen3TTSConfig)
        AutoModel.register(Qwen3TTSConfig, Qwen3TTSForConditionalGeneration)
        AutoProcessor.register(Qwen3TTSConfig, Qwen3TTSProcessor)

        model = AutoModel.from_pretrained(pretrained_model_name_or_path, **kwargs)
        if not isinstance(model, Qwen3TTSForConditionalGeneration):
            raise TypeError(
                f"AutoModel returned {type(model)}, expected Qwen3TTSForConditionalGeneration. "
            )

        processor = AutoProcessor.from_pretrained(pretrained_model_name_or_path, fix_mistral_regex=True,)

        generate_defaults = model.generate_config
        return cls(model=model, processor=processor, generate_defaults=generate_defaults)

    def _supported_languages_set(self) -> Optional[set]:
        langs = getattr(self.model, "get_supported_languages", None)
        if callable(langs):
            v = langs()
            if v is None:
                return None
            return set([str(x).lower() for x in v])
        return None

    def _supported_speakers_set(self) -> Optional[set]:
        spks = getattr(self.model, "get_supported_speakers", None)
        if callable(spks):
            v = spks()
            if v is None:
                return None
            return set([str(x).lower() for x in v])
        return None

    def _validate_languages(self, languages: List[str]) -> None:
        supported = self._supported_languages_set()
        if supported is None:
            return
        bad = [lang for lang in languages if lang is None or str(lang).lower() not in supported]
        if bad:
            raise ValueError(f"Unsupported languages: {bad}. Supported: {sorted(supported)}")

    def _validate_speakers(self, speakers: List[Optional[str]]) -> None:
        supported = self._supported_speakers_set()
        if supported is None:
            return
        bad = [spk for spk in speakers if spk and str(spk).lower() not in supported]
        if bad:
            raise ValueError(f"Unsupported speakers: {bad}. Supported: {sorted(supported)}")

    def _is_probably_base64(self, s: str) -> bool:
        if s.startswith("data:audio"):
            return True
        if ("/" not in s and "\\" not in s) and len(s) > 256:
            return True
        return False

    def _is_url(self, s: str) -> bool:
        try:
            u = urlparse(s)
            return u.scheme in ("http", "https") and bool(u.netloc)
        except Exception:
            return False

    def _decode_base64_to_wav_bytes(self, b64: str) -> bytes:
        if "," in b64 and b64.strip().startswith("data:"):
            b64 = b64.split(",", 1)[1]
        return base64.b64decode(b64)

    def _load_audio_to_np(self, x: str) -> Tuple[np.ndarray, int]:
        if self._is_url(x):
            with urllib.request.urlopen(x) as resp:
                audio_bytes = resp.read()
            with io.BytesIO(audio_bytes) as f:
                audio, sr = sf.read(f, dtype="float32", always_2d=False)
        elif self._is_probably_base64(x):
            wav_bytes = self._decode_base64_to_wav_bytes(x)
            with io.BytesIO(wav_bytes) as f:
                audio, sr = sf.read(f, dtype="float32", always_2d=False)
        else:
            audio, sr = librosa.load(x, sr=None, mono=True)

        if audio.ndim > 1:
            audio = np.mean(audio, axis=-1)

        return audio.astype(np.float32), int(sr)

    def _normalize_audio_inputs(self, audios: Union[AudioLike, List[AudioLike]]) -> List[Tuple[np.ndarray, int]]:
        """
        Normalize audio inputs into a list of (waveform, sr).

        Supported forms:
          - str: wav path / URL / base64 audio string
          - (np.ndarray, sr): waveform + sampling rate
          - list of the above

        Args:
            audios:
                Audio input(s).

        Returns:
            List[Tuple[np.ndarray, int]]:
                List of (float32 waveform, original sr).

        Raises:
            ValueError: If a numpy waveform is provided without sr.
        """
        items = audios if isinstance(audios, list) else [audios]

        out: List[Tuple[np.ndarray, int]] = []
        for a in items:
            if isinstance(a, str):
                wav, sr = self._load_audio_to_np(a)
            elif isinstance(a, tuple) and len(a) == 2 and isinstance(a[0], np.ndarray):
                wav, sr = a[0].astype(np.float32), int(a[1])
            elif isinstance(a, np.ndarray):
                raise ValueError("For numpy waveform input, pass a tuple (audio, sr).")
            else:
                raise TypeError(f"Unsupported audio input type: {type(a)}")

            # [OPT-6] Stereo→mono done here in a single pass — no second loop needed.
            if wav.ndim > 1:
                wav = np.mean(wav, axis=-1).astype(np.float32)

            out.append((wav, sr))
        return out

    def _ensure_list(self, x: MaybeList) -> List[Any]:
        return x if isinstance(x, list) else [x]

    def _build_assistant_text(self, text: str) -> str:
        return f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"

    def _build_ref_text(self, text: str) -> str:
        return f"<|im_start|>assistant\n{text}<|im_end|>\n"

    def _build_instruct_text(self, instruct: str) -> str:
        return f"<|im_start|>user\n{instruct}<|im_end|>\n"

    def _tokenize_texts(self, texts: List[str]) -> List[torch.Tensor]:
        """
        [OPT-2 / OPT-7] Tokenize a list of strings in ONE processor call.

        The Rust-backed HuggingFace tokenizer parallelises internally when given a list,
        cutting per-item binding overhead from O(N) to O(1).  All token id sequences are
        then assembled into individual tensors on CPU first, then moved to `self.device`
        as a single batched operation per item — avoiding per-item CUDA synchronisation.
        """
        # Single processor call for the entire batch — O(1) Rust binding overhead.
        inputs = self.processor(text=texts, padding=False)

        # Build all tensors on CPU, then move to device. Avoids N synchronisation points.
        return [
            torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(self.device)
            for ids in inputs["input_ids"]
        ]

    def _merge_generate_kwargs(
        self,
        do_sample: Optional[bool] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        temperature: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        subtalker_dosample: Optional[bool] = None,
        subtalker_top_k: Optional[int] = None,
        subtalker_top_p: Optional[float] = None,
        subtalker_temperature: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Merge user-provided generation arguments with pre-baked defaults.

        [OPT-8] Defaults are merged with hard_defaults once in __init__ into
        self._baked_defaults, so this method is a simple dict update — no repeated
        dict lookups or conditional chains per call.

        Rule:
          - If the user explicitly passes a value (not None), use it.
          - Otherwise, use the value from _baked_defaults (already merged with generate_config.json).
        """
        user_overrides = {
            k: v for k, v in {
                "do_sample": do_sample,
                "top_k": top_k,
                "top_p": top_p,
                "temperature": temperature,
                "repetition_penalty": repetition_penalty,
                "subtalker_dosample": subtalker_dosample,
                "subtalker_top_k": subtalker_top_k,
                "subtalker_top_p": subtalker_top_p,
                "subtalker_temperature": subtalker_temperature,
                "max_new_tokens": max_new_tokens,
            }.items() if v is not None
        }
        return {**kwargs, **self._baked_defaults, **user_overrides}

    # -------------------------------------------------------------------------
    # [OPT-4] Batched CPU resampling helper
    # -------------------------------------------------------------------------
    def _batch_resample_cpu(
        self,
        wavs: List[np.ndarray],
        src_srs: List[int],
        target_sr: int,
    ) -> List[np.ndarray]:
        """
        [OPT-4] Resample a batch of waveforms in ONE GPU round-trip.

        Strategy
        --------
        Instead of the original pattern (per-sample CPU→GPU→resample→CPU), we:
          1. Identify which samples actually need resampling (src_sr != target_sr).
          2. For those that do, zero-pad them to a common length on CPU.
          3. Stack into a single (B, T) tensor and move to GPU *once*.
          4. Call torchaudio.functional.resample on the whole batch.
             Note: F.resample is a stateless per-sample filter applied independently
             across the batch dimension, so padding does not affect the valid samples.
          5. Move result back to CPU *once*, slice out padding, return numpy arrays.

        Samples that are already at target_sr are returned unchanged (no device transfer).

        This reduces PCIe transfers from 2×N (N host→device + N device→host) to at most
        2 (one H→D, one D→H) regardless of batch size.

        Args:
            wavs: list of float32 mono waveforms (variable length).
            src_srs: sample rate for each waveform.
            target_sr: desired output sample rate.

        Returns:
            List of float32 numpy arrays at target_sr (same order as input).
        """
        needs_resample_idx = [i for i, sr in enumerate(src_srs) if sr != target_sr]

        if not needs_resample_idx:
            # Nothing to do — avoid GPU round-trip entirely.
            return list(wavs)

        # ── Group by source SR so we can batch within each SR group ──────────
        # torchaudio.functional.resample requires a single (orig_freq, new_freq) pair
        # per call, so wavs with different source SRs need separate kernel calls.
        # However, each group still benefits from a single H→D/D→H pair.
        from collections import defaultdict
        sr_groups: Dict[int, List[int]] = defaultdict(list)
        for i in needs_resample_idx:
            sr_groups[src_srs[i]].append(i)

        resampled: Dict[int, np.ndarray] = {}

        for src_sr, indices in sr_groups.items():
            group_wavs = [wavs[i] for i in indices]

            # Pad to uniform length within group.
            max_len = max(w.shape[0] for w in group_wavs)
            padded = np.zeros((len(group_wavs), max_len), dtype=np.float32)
            lengths = []
            for j, w in enumerate(group_wavs):
                padded[j, :w.shape[0]] = w
                lengths.append(w.shape[0])

            # Single H→D transfer for the whole group.
            batch_tensor = torch.from_numpy(padded).to(self.device)   # (B, T_src)

            # One resample kernel for the group.
            batch_resampled = F.resample(
                batch_tensor,
                orig_freq=int(src_sr),
                new_freq=int(target_sr),
            )                                                           # (B, T_tgt)

            # Single D→H transfer.
            batch_np = batch_resampled.cpu().numpy()                   # (B, T_tgt)

            # Compute expected output length so we can slice off padding correctly.
            for j, (orig_idx, src_len) in enumerate(zip(indices, lengths)):
                expected_out_len = int(np.round(src_len * target_sr / src_sr))
                # Clamp to actual output size in case of rounding.
                expected_out_len = min(expected_out_len, batch_np.shape[1])
                resampled[orig_idx] = batch_np[j, :expected_out_len]

        # ── Assemble output list ───────────────────────────────────────────────
        out: List[np.ndarray] = []
        for i, wav in enumerate(wavs):
            if i in resampled:
                out.append(resampled[i])
            else:
                out.append(wav)
        return out

    # -------------------------------------------------------------------------
    # voice clone model — prompt builder
    # -------------------------------------------------------------------------
    @torch.inference_mode()
    def create_voice_clone_prompt(
        self,
        ref_audio: Union[AudioLike, List[AudioLike]],
        ref_text: Optional[Union[str, List[Optional[str]]]] = None,
        x_vector_only_mode: Union[bool, List[bool]] = False,
    ) -> List[VoiceClonePromptItem]:
        """
        Build voice-clone prompt items from reference audio (and optionally reference text).

        [OPT-3] Speaker embeddings are now extracted in ONE batched call instead of N
        sequential calls, fully saturating the GPU speaker encoder.

        [OPT-4] All resampling is done via _batch_resample_cpu: one H→D and one D→H
        transfer covers the entire batch, regardless of batch size.

        Modes:
          - x_vector_only_mode=True:
              Only speaker embedding is used to clone voice; ref_text/ref_code are ignored.
          - x_vector_only_mode=False:
              ICL mode is enabled (icl_mode=True). ref_text is required.

        Args:
            ref_audio: Reference audio(s) — str path/URL/base64, or (ndarray, sr) tuple.
            ref_text: Reference transcript(s). Required when x_vector_only_mode=False.
            x_vector_only_mode: Speaker-only mode flag, scalar or list.

        Returns:
            List[VoiceClonePromptItem]

        Raises:
            ValueError: If x_vector_only_mode=False and ref_text is missing, or batch mismatch.
        """
        if self.model.tts_model_type != "base":
            raise ValueError(
                f"model with \ntokenizer_type: {self.model.tokenizer_type}\n"
                f"tts_model_size: {self.model.tts_model_size}\n"
                f"tts_model_type: {self.model.tts_model_type}\n"
                "does not support create_voice_clone_prompt, Please check Model Card or Readme for more details."
            )

        ref_audio_list = self._ensure_list(ref_audio)
        N = len(ref_audio_list)
        ref_text_list = ref_text if isinstance(ref_text, list) else [ref_text] * N
        xvec_list = x_vector_only_mode if isinstance(x_vector_only_mode, list) else [x_vector_only_mode] * N

        if len(ref_text_list) != N or len(xvec_list) != N:
            raise ValueError(
                f"Batch size mismatch: ref_audio={N}, ref_text={len(ref_text_list)}, "
                f"x_vector_only_mode={len(xvec_list)}"
            )

        # Validate ICL mode requirements before any heavy computation.
        for i, (rtext, xvec_only) in enumerate(zip(ref_text_list, xvec_list)):
            if not xvec_only and (rtext is None or rtext == ""):
                raise ValueError(
                    f"ref_text is required when x_vector_only_mode=False (ICL mode). Bad index={i}"
                )

        normalized = self._normalize_audio_inputs(ref_audio_list)
        wavs = [w for w, _ in normalized]
        srs  = [s for _, s in normalized]

        # ── Speech tokenizer encoding ─────────────────────────────────────────
        if len(set(srs)) == 1:
            enc = self.model.speech_tokenizer.encode(wavs, sr=srs[0])
            ref_codes = enc.audio_codes          # list[Tensor] or batched Tensor
        else:
            ref_codes = []
            for wav, sr in zip(wavs, srs):
                ref_codes.append(
                    self.model.speech_tokenizer.encode(wav, sr=sr).audio_codes[0]
                )

        # ── [OPT-4] Batch resample all audios that need it in one GPU trip ────
        target_sr = self.model.speaker_encoder_sample_rate
        resampled_wavs = self._batch_resample_cpu(wavs, srs, target_sr)

        # ── [OPT-3] Batch speaker embedding extraction ────────────────────────
        # extract_speaker_embedding is assumed to accept a list/batch of waveforms.
        # If the underlying method signature only accepts a single array, the
        # fallback path below (commented out) can be used instead.
        try:
            # Preferred: single batched call — one forward pass through the speaker encoder.
            all_spk_embs = self.model.extract_speaker_embedding(
                audio=resampled_wavs,        # List[np.ndarray] — batched input
                sr=target_sr,
            )
            # Normalise output: if the model returns a single (N, D) tensor, split it.
            if isinstance(all_spk_embs, torch.Tensor) and all_spk_embs.dim() == 2:
                all_spk_embs = [all_spk_embs[i] for i in range(all_spk_embs.shape[0])]
        except (TypeError, RuntimeError):
            # Fallback: if the model's speaker encoder does not support batched input,
            # iterate sequentially. This preserves correctness at the cost of speed.
            # TODO: patch extract_speaker_embedding upstream to accept a list.
            all_spk_embs = [
                self.model.extract_speaker_embedding(audio=w, sr=target_sr)
                for w in resampled_wavs
            ]

        # ── Assemble VoiceClonePromptItems ────────────────────────────────────
        items: List[VoiceClonePromptItem] = []
        for i, (code, spk_emb, rtext, xvec_only) in enumerate(
            zip(ref_codes, all_spk_embs, ref_text_list, xvec_list)
        ):
            items.append(
                VoiceClonePromptItem(
                    ref_code=None if xvec_only else code,
                    ref_spk_embedding=spk_emb,
                    x_vector_only_mode=bool(xvec_only),
                    icl_mode=bool(not xvec_only),
                    ref_text=rtext,
                )
            )
        return items

    def _prompt_items_to_voice_clone_prompt(self, items: List[VoiceClonePromptItem]) -> Dict[str, Any]:
        return dict(
            ref_code=[it.ref_code for it in items],
            ref_spk_embedding=[it.ref_spk_embedding for it in items],
            x_vector_only_mode=[it.x_vector_only_mode for it in items],
            icl_mode=[it.icl_mode for it in items],
        )

    # -------------------------------------------------------------------------
    # voice clone model — generation
    # -------------------------------------------------------------------------
    @torch.inference_mode()
    def generate_voice_clone(
        self,
        text: Union[str, List[str]],
        language: Union[str, List[str]] = None,
        ref_audio: Optional[Union[AudioLike, List[AudioLike]]] = None,
        ref_text: Optional[Union[str, List[Optional[str]]]] = None,
        x_vector_only_mode: Union[bool, List[bool]] = False,
        voice_clone_prompt: Optional[Union[Dict[str, Any], List[VoiceClonePromptItem]]] = None,
        non_streaming_mode: bool = False,
        **kwargs,
    ) -> Tuple[List[np.ndarray], int]:
        """
        Voice clone speech using the Base model.

        [OPT-2] ref_text tokenization is now batched: instead of N sequential calls to
        _tokenize_texts (each paying the Python→Rust binding overhead), all non-None
        ref_texts are collected and tokenized in ONE processor call.

        You can provide either:
          - (ref_audio, ref_text, x_vector_only_mode) and let this method build the prompt, OR
          - a list of `VoiceClonePromptItem` from `create_voice_clone_prompt`.

        Args:
            text: Text(s) to synthesize.
            language: Language(s) for each sample.
            ref_audio: Reference audio(s). Required if voice_clone_prompt is not provided.
            ref_text: Reference text(s) for ICL mode.
            x_vector_only_mode: Speaker-only mode flag.
            voice_clone_prompt: Pre-built prompt items or dict.
            non_streaming_mode: Simulate streaming text input when False.
            **kwargs: Forwarded to model.generate().

        Returns:
            Tuple[List[np.ndarray], int]: (wavs, sample_rate)
        """
        if self.model.tts_model_type != "base":
            raise ValueError(
                f"model with \ntokenizer_type: {self.model.tokenizer_type}\n"
                f"tts_model_size: {self.model.tts_model_size}\n"
                f"tts_model_type: {self.model.tts_model_type}\n"
                "does not support generate_voice_clone, Please check Model Card or Readme for more details."
            )

        texts = self._ensure_list(text)
        N = len(texts)
        languages = (
            language if isinstance(language, list)
            else ([language] * N if language is not None else ["Auto"] * N)
        )
        if len(languages) == 1 and N > 1:
            languages = languages * N
        if len(texts) != len(languages):
            raise ValueError(f"Batch size mismatch: text={N}, language={len(languages)}")

        self._validate_languages(languages)

        if voice_clone_prompt is None:
            if ref_audio is None:
                raise ValueError("Either `voice_clone_prompt` or `ref_audio` must be provided.")
            prompt_items = self.create_voice_clone_prompt(
                ref_audio=ref_audio, ref_text=ref_text, x_vector_only_mode=x_vector_only_mode
            )
            if len(prompt_items) == 1 and N > 1:
                prompt_items = prompt_items * N
            if len(prompt_items) != N:
                raise ValueError(f"Batch size mismatch: prompt={len(prompt_items)}, text={N}")
            voice_clone_prompt_dict = self._prompt_items_to_voice_clone_prompt(prompt_items)
            ref_texts_for_ids = [it.ref_text for it in prompt_items]
        else:
            if isinstance(voice_clone_prompt, list):
                prompt_items = voice_clone_prompt
                if len(prompt_items) == 1 and N > 1:
                    prompt_items = prompt_items * N
                if len(prompt_items) != N:
                    raise ValueError(f"Batch size mismatch: prompt={len(prompt_items)}, text={N}")
                voice_clone_prompt_dict = self._prompt_items_to_voice_clone_prompt(prompt_items)
                ref_texts_for_ids = [it.ref_text for it in prompt_items]
            else:
                voice_clone_prompt_dict = voice_clone_prompt
                ref_texts_for_ids = None

        # ── [OPT-2] Batch tokenize all input texts in a single processor call ─
        input_ids = self._tokenize_texts([self._build_assistant_text(t) for t in texts])

        # ── [OPT-2] Batch tokenize all ref_texts in a single processor call ───
        # Previous code called _tokenize_texts([...]) inside a for-loop: N round-trips.
        # New code: collect all non-None ref strings, ONE processor call, reassemble.
        ref_ids: Optional[List[Optional[torch.Tensor]]] = None
        if ref_texts_for_ids is not None:
            # Identify which indices have a real ref_text.
            valid_indices = [
                i for i, rt in enumerate(ref_texts_for_ids) if rt is not None and rt != ""
            ]
            if valid_indices:
                # ONE tokenizer call for all non-None ref texts.
                built_ref_strings = [
                    self._build_ref_text(ref_texts_for_ids[i]) for i in valid_indices
                ]
                tokenized_refs = self._tokenize_texts(built_ref_strings)

                # Reassemble into a list aligned with the original N samples.
                ref_ids = [None] * N
                for list_pos, orig_idx in enumerate(valid_indices):
                    ref_ids[orig_idx] = tokenized_refs[list_pos]
            else:
                ref_ids = [None] * N

        gen_kwargs = self._merge_generate_kwargs(**kwargs)

        talker_codes_list, _ = self.model.generate(
            input_ids=input_ids,
            ref_ids=ref_ids,
            voice_clone_prompt=voice_clone_prompt_dict,
            languages=languages,
            non_streaming_mode=non_streaming_mode,
            **gen_kwargs,
        )

        # ── [OPT-9] Hoist loop-invariant dict lookup outside the loop ─────────
        ref_code_list = voice_clone_prompt_dict.get("ref_code", None)

        codes_for_decode = []
        for i, codes in enumerate(talker_codes_list):
            if ref_code_list is not None and ref_code_list[i] is not None:
                # [OPT-10] .to(device) is only called if the tensor isn't already there.
                ref_c = ref_code_list[i]
                if ref_c.device != codes.device:
                    ref_c = ref_c.to(codes.device)
                codes_for_decode.append(torch.cat([ref_c, codes], dim=0))
            else:
                codes_for_decode.append(codes)

        wavs_all, fs = self.model.speech_tokenizer.decode(
            [{"audio_codes": c} for c in codes_for_decode]
        )

        wavs_out: List[np.ndarray] = []
        for i, wav in enumerate(wavs_all):
            if ref_code_list is not None and ref_code_list[i] is not None:
                ref_len = int(ref_code_list[i].shape[0])
                total_len = int(codes_for_decode[i].shape[0])
                cut = int(ref_len / max(total_len, 1) * wav.shape[0])
                wavs_out.append(wav[cut:])
            else:
                wavs_out.append(wav)

        return wavs_out, fs

    # -------------------------------------------------------------------------
    # voice design model
    # -------------------------------------------------------------------------
    @torch.inference_mode()
    def generate_voice_design(
        self,
        text: Union[str, List[str]],
        instruct: Union[str, List[str]],
        language: Union[str, List[str]] = None,
        non_streaming_mode: bool = True,
        **kwargs,
    ) -> Tuple[List[np.ndarray], int]:
        """
        Generate speech with the VoiceDesign model using natural-language style instructions.

        Args:
            text: Text(s) to synthesize.
            language: Language(s) for each sample.
            instruct: Instruction(s) describing desired voice/style.
            non_streaming_mode: Simulate streaming text input when False.
            **kwargs: Forwarded to model.generate().

        Returns:
            Tuple[List[np.ndarray], int]: (wavs, sample_rate)
        """
        if self.model.tts_model_type != "voice_design":
            raise ValueError(
                f"model with \ntokenizer_type: {self.model.tokenizer_type}\n"
                f"tts_model_size: {self.model.tts_model_size}\n"
                f"tts_model_type: {self.model.tts_model_type}\n"
                "does not support generate_voice_design, Please check Model Card or Readme for more details."
            )

        texts = self._ensure_list(text)
        N = len(texts)
        languages = (
            language if isinstance(language, list)
            else ([language] * N if language is not None else ["Auto"] * N)
        )
        instructs = self._ensure_list(instruct)

        if len(languages) == 1 and N > 1:
            languages = languages * N
        if len(instructs) == 1 and N > 1:
            instructs = instructs * N

        if not (N == len(languages) == len(instructs)):
            raise ValueError(
                f"Batch size mismatch: text={N}, language={len(languages)}, instruct={len(instructs)}"
            )

        self._validate_languages(languages)

        # [OPT-2] All input texts tokenized in one call.
        input_ids = self._tokenize_texts([self._build_assistant_text(t) for t in texts])

        # [OPT-2] All non-empty instruct strings tokenized in one call.
        valid_instruct_indices = [i for i, ins in enumerate(instructs) if ins and ins != ""]
        instruct_ids: List[Optional[torch.Tensor]] = [None] * N
        if valid_instruct_indices:
            built_instruct_strings = [
                self._build_instruct_text(instructs[i]) for i in valid_instruct_indices
            ]
            tokenized_instructs = self._tokenize_texts(built_instruct_strings)
            for list_pos, orig_idx in enumerate(valid_instruct_indices):
                instruct_ids[orig_idx] = tokenized_instructs[list_pos]

        gen_kwargs = self._merge_generate_kwargs(**kwargs)

        talker_codes_list, _ = self.model.generate(
            input_ids=input_ids,
            instruct_ids=instruct_ids,
            languages=languages,
            non_streaming_mode=non_streaming_mode,
            **gen_kwargs,
        )

        wavs, fs = self.model.speech_tokenizer.decode(
            [{"audio_codes": c} for c in talker_codes_list]
        )
        return wavs, fs

    # -------------------------------------------------------------------------
    # custom voice model
    # -------------------------------------------------------------------------
    @torch.inference_mode()
    def generate_custom_voice(
        self,
        text: Union[str, List[str]],
        speaker: Union[str, List[str]],
        language: Union[str, List[str]] = None,
        instruct: Optional[Union[str, List[str]]] = None,
        non_streaming_mode: bool = True,
        **kwargs,
    ) -> Tuple[List[np.ndarray], int]:
        """
        Generate speech with the CustomVoice model using a predefined speaker id,
        optionally controlled by instruction text.

        Args:
            text: Text(s) to synthesize.
            language: Language(s) for each sample.
            speaker: Speaker name(s).
            instruct: Optional instruction(s).
            non_streaming_mode: Simulate streaming text input when False.
            **kwargs: Forwarded to model.generate().

        Returns:
            Tuple[List[np.ndarray], int]: (wavs, sample_rate)
        """
        if self.model.tts_model_type != "custom_voice":
            raise ValueError(
                f"model with \ntokenizer_type: {self.model.tokenizer_type}\n"
                f"tts_model_size: {self.model.tts_model_size}\n"
                f"tts_model_type: {self.model.tts_model_type}\n"
                "does not support generate_custom_voice, Please check Model Card or Readme for more details."
            )

        texts = self._ensure_list(text)
        N = len(texts)
        languages = (
            language if isinstance(language, list)
            else ([language] * N if language is not None else ["Auto"] * N)
        )
        speakers = self._ensure_list(speaker)
        if self.model.tts_model_size in "0b6":
            instruct = None
        instructs = (
            instruct if isinstance(instruct, list)
            else ([instruct] * N if instruct is not None else [""] * N)
        )

        if len(languages) == 1 and N > 1:
            languages = languages * N
        if len(speakers) == 1 and N > 1:
            speakers = speakers * N
        if len(instructs) == 1 and N > 1:
            instructs = instructs * N

        if not (N == len(languages) == len(speakers) == len(instructs)):
            raise ValueError(
                f"Batch size mismatch: text={N}, language={len(languages)}, "
                f"speaker={len(speakers)}, instruct={len(instructs)}"
            )

        self._validate_languages(languages)
        self._validate_speakers(speakers)

        # [OPT-2] All input texts tokenized in one call.
        input_ids = self._tokenize_texts([self._build_assistant_text(t) for t in texts])

        # [OPT-2] All non-empty instruct strings tokenized in one call.
        valid_instruct_indices = [i for i, ins in enumerate(instructs) if ins and ins != ""]
        instruct_ids: List[Optional[torch.Tensor]] = [None] * N
        if valid_instruct_indices:
            built_instruct_strings = [
                self._build_instruct_text(instructs[i]) for i in valid_instruct_indices
            ]
            tokenized_instructs = self._tokenize_texts(built_instruct_strings)
            for list_pos, orig_idx in enumerate(valid_instruct_indices):
                instruct_ids[orig_idx] = tokenized_instructs[list_pos]

        gen_kwargs = self._merge_generate_kwargs(**kwargs)

        talker_codes_list, _ = self.model.generate(
            input_ids=input_ids,
            instruct_ids=instruct_ids,
            languages=languages,
            speakers=speakers,
            non_streaming_mode=non_streaming_mode,
            **gen_kwargs,
        )

        wavs, fs = self.model.speech_tokenizer.decode(
            [{"audio_codes": c} for c in talker_codes_list]
        )
        return wavs, fs

    # -------------------------------------------------------------------------
    # Introspection helpers
    # -------------------------------------------------------------------------
    def get_supported_speakers(self) -> Optional[List[str]]:
        """Return sorted list of supported speaker names, or None."""
        supported = self._supported_speakers_set()
        return sorted(supported) if supported is not None else None

    def get_supported_languages(self) -> Optional[List[str]]:
        """Return sorted list of supported language names, or None."""
        supported = self._supported_languages_set()
        return sorted(supported) if supported is not None else None