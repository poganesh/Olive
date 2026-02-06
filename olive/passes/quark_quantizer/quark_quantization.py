#
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import logging
import os
import sys
from pathlib import Path
from typing import Union

import torch

from olive.model import HfModelHandler, ONNXModelHandler
from olive.passes import Pass
from olive.passes.pass_config import BasePassConfig, PassConfigParam

logger = logging.getLogger(__name__)


class QuarkQuantization(Pass):
    """Quark 0.11 Quantization Pass for Olive."""

    @classmethod
    def _default_config(cls, accelerator_spec=None):
        return {
            "quant_scheme": PassConfigParam(
                type_=str,
                default_value="uint4_wo_128",
                description="Quantization scheme: uint4_wo_128, int4_wo_128, int8, fp8, mxfp4",
            ),
            "quant_algo": PassConfigParam(
                type_=Union[str, list],
                default_value="awq",
                description="Quantization algorithm(s): awq, gptq, smoothquant, rotation",
            ),
            "dataset": PassConfigParam(
                type_=str,
                default_value="pileval_for_awq_benchmark",
                description="Calibration dataset",
            ),
            "data_type": PassConfigParam(
                type_=str,
                default_value="bfloat16",
                description="Model data type: auto, float16, bfloat16, float32",
            ),
            "num_calib_data": PassConfigParam(
                type_=int,
                default_value=128,
                description="Number of calibration samples",
            ),
            "model_export": PassConfigParam(
                type_=Union[str, list],
                default_value="hf_format",
                description="Export format(s): hf_format, onnx, gguf",
            ),
            "exclude_layers": PassConfigParam(
                type_=list,
                default_value=None,
                description="Layers to exclude from quantization",
            ),
            "layer_quant_scheme": PassConfigParam(
                type_=list,
                default_value=None,
                description="Layer-specific schemes: [['lm_head', 'int8']]",
            ),
            "kv_cache_dtype": PassConfigParam(
                type_=str,
                default_value=None,
                description="KV cache dtype: fp8 or None",
            ),
            "min_kv_scale": PassConfigParam(
                type_=float,
                default_value=0.0,
                description="Minimum scale for KV cache quantization",
            ),
            "kv_cache_post_rope": PassConfigParam(
                type_=bool,
                default_value=False,
                description="Quantize KV cache after RoPE",
            ),
            "attention_dtype": PassConfigParam(
                type_=str,
                default_value=None,
                description="Attention quantization dtype: fp8 or None",
            ),
            "seq_len": PassConfigParam(
                type_=int,
                default_value=512,
                description="Sequence length for calibration",
            ),
            "batch_size": PassConfigParam(
                type_=int,
                default_value=1,
                description="Batch size for calibration",
            ),
            "pack_method": PassConfigParam(
                type_=str,
                default_value="reorder",
                description="Pack method: order or reorder",
            ),
            "export_weight_format": PassConfigParam(
                type_=str,
                default_value="real_quantized",
                description="Weight format: fake_quantized or real_quantized",
            ),
            "custom_mode": PassConfigParam(
                type_=str,
                default_value="quark",
                description="Export mode: quark, awq, fp8",
            ),
            "trust_remote_code": PassConfigParam(
                type_=bool,
                default_value=True,
                description="Trust remote code from HuggingFace",
            ),
        }

    def _run_for_config(
        self,
        model: Union[HfModelHandler, ONNXModelHandler],
        config: BasePassConfig,
        output_model_path: str,
    ) -> Union[HfModelHandler, ONNXModelHandler]:
        if isinstance(model, ONNXModelHandler):
            raise ValueError("ONNX model quantization is not supported. Use HfModelHandler.")

        logger.info("[INFO] Running QuarkQuantization using Quark 0.11 API")
        return self._run_quark_torch(model, config, output_model_path)

    def _run_quark_torch(
        self,
        model: HfModelHandler,
        config: BasePassConfig,
        output_model_path: str,
    ) -> HfModelHandler:
        """Run Quark 0.11 torch quantization."""
        # Disable torch dynamo on Windows (Triton not available)
        if sys.platform == "win32":
            os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

        from quark.torch import (
            LLMTemplate,
            ModelQuantizer,
            export_gguf,
            export_onnx,
            export_safetensors,
        )
        from quark.torch.utils.llm import (
            get_calib_dataloader,
            get_model,
            get_tokenizer,
            prepare_for_moe_quant,
            revert_model_patching,
        )

        output_dir = Path(output_model_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # 1. Load model
        logger.info("[INFO] Loading model from: %s", model.model_path)
        torch_model, _ = get_model(
            str(model.model_path),
            config.data_type,
            device,
            multi_gpu=False,
            multi_device=False,
            attn_implementation="eager",
            trust_remote_code=config.trust_remote_code,
        )

        prepare_for_moe_quant(torch_model)

        model_type = (
            torch_model.config.model_type
            if hasattr(torch_model.config, "model_type")
            else torch_model.config.architectures[0]
        )

        tokenizer = get_tokenizer(
            str(model.model_path),
            max_seq_len=config.seq_len,
            model_type=model_type,
            trust_remote_code=config.trust_remote_code,
        )

        # 2. Prepare calibration data
        logger.info("[INFO] Loading calibration dataset: %s", config.dataset)
        main_device = torch_model.device
        calib_dataloader = get_calib_dataloader(
            dataset_name=config.dataset,
            processor=None,
            tokenizer=tokenizer,
            batch_size=config.batch_size,
            num_calib_data=config.num_calib_data,
            seqlen=config.seq_len,
            device=main_device,
        )

        # 3. Setup quantization config
        logger.info("[INFO] Setting up quantization with scheme: %s", config.quant_scheme)

        if model_type not in LLMTemplate.list_available():
            raise ValueError(f"Model type '{model_type}' is not supported. Available: {LLMTemplate.list_available()}")

        template = LLMTemplate.get(model_type)

        # Handle quant_algo as string or list
        quant_algo_list = None
        if config.quant_algo:
            if isinstance(config.quant_algo, str):
                quant_algo_list = config.quant_algo.split(",") if "," in config.quant_algo else [config.quant_algo]
            elif isinstance(config.quant_algo, list):
                quant_algo_list = config.quant_algo

        # Build layer-specific config
        layer_config = dict(config.layer_quant_scheme) if config.layer_quant_scheme else {}

        # Get quantization config from template
        quant_config = template.get_config(
            scheme=config.quant_scheme,
            algorithm=quant_algo_list,
            kv_cache_scheme=config.kv_cache_dtype,
            min_kv_scale=config.min_kv_scale,
            layer_config=layer_config if layer_config else None,
            attention_scheme=config.attention_dtype,
            exclude_layers=config.exclude_layers,
        )

        # Handle kv_cache_post_rope flag
        if config.kv_cache_post_rope:
            if hasattr(quant_config, "kv_cache_post_rope"):
                quant_config.kv_cache_post_rope = True
            else:
                logger.warning("kv_cache_post_rope not supported by quant_config")

        # 4. Quantize model
        logger.info("[INFO] Starting model quantization")
        quantizer = ModelQuantizer(quant_config, multi_device=False)
        torch_model = quantizer.quantize_model(torch_model, calib_dataloader)

        # 5. Freeze model
        logger.info("[INFO] Freezing quantized model")
        torch_model = quantizer.freeze(torch_model)

        # 6. Revert model patching
        logger.info("[INFO] Reverting model patching")
        revert_model_patching(torch_model)

        # 7. Validate export configuration
        if config.custom_mode != "quark" and config.export_weight_format == "fake_quantized":
            raise ValueError("'fake_quantized' only supports custom_mode='quark'")

        # 8. Export model
        logger.info("[INFO] Exporting quantized model to: %s", output_dir)

        export_formats = config.model_export
        if isinstance(export_formats, str):
            export_formats = [export_formats]
        elif export_formats is None:
            export_formats = ["hf_format"]

        for export_format in export_formats:
            if export_format == "hf_format":
                with torch.no_grad():
                    export_safetensors(
                        model=torch_model,
                        output_dir=str(output_dir),
                        custom_mode=config.custom_mode,
                        weight_format=config.export_weight_format,
                        pack_method=config.pack_method,
                    )
                    tokenizer.save_pretrained(str(output_dir))
                logger.info("[INFO] Exported HF format to: %s", output_dir)

            elif export_format == "onnx":
                with torch.inference_mode():
                    batch_iter = iter(calib_dataloader)
                    input_args = next(batch_iter)
                    uint4_int4_flag = "uint4" in config.quant_scheme or "int4" in config.quant_scheme

                    onnx_output_dir = output_dir / "onnx"
                    onnx_output_dir.mkdir(exist_ok=True)
                    export_onnx(
                        model=torch_model,
                        output_dir=str(onnx_output_dir),
                        input_args=input_args,
                        uint4_int4_flag=uint4_int4_flag,
                    )
                logger.info("[INFO] Exported ONNX format to: %s", onnx_output_dir)

            elif export_format == "gguf":
                with torch.inference_mode():
                    export_gguf(
                        torch_model,
                        output_dir=str(output_dir),
                        model_type=model_type,
                        tokenizer_path=str(model.model_path),
                    )
                logger.info("[INFO] Exported GGUF format to: %s", output_dir)

        return HfModelHandler(str(output_dir))
