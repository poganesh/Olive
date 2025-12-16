#
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import json
import logging
import tempfile
from argparse import Namespace
from pathlib import Path
from typing import Optional, Union

import onnx
import torch
from packaging import version

from olive.common.config_utils import validate_config
from olive.common.utils import exclude_keys
from olive.data.config import DataConfig
from olive.model import HfModelHandler, ONNXModelHandler
from olive.model.utils import resolve_onnx_path
from olive.passes import Pass
from olive.passes.onnx.common import get_external_data_config, model_proto_to_olive_model
from olive.passes.pass_config import BasePassConfig, PassConfigParam
from olive.search.search_parameter import Categorical

logger = logging.getLogger(__name__)


class QuarkQuantization(Pass):
    """Quark 0.10 Quantization Pass for Olive."""

    @classmethod
    def _default_config(cls, accelerator_spec=None):
        return {
            # Core quantization parameters
            "quant_scheme": PassConfigParam(
                type_=str,
                default_value="w_uint4_per_group_asym",
                description="Quantization scheme. Examples: w_uint4_per_group_asym, w_int4_per_group_sym, w_int8_a_int8_per_tensor_sym, w_fp8_a_fp8",
            ),
            "quant_algo": PassConfigParam(
                type_=Union[str, list],
                default_value="awq",
                description="Quantization algorithm(s). Single string 'awq' or list ['awq', 'gptq']. Options: awq, gptq, smoothquant, rotation",
            ),
            "dataset": PassConfigParam(
                type_=str,
                default_value="pileval_for_awq_benchmark",
                description="Calibration dataset. Options: pileval_for_awq_benchmark, wikitext_for_gptq_benchmark, cnn_dailymail",
            ),
            "data_type": PassConfigParam(
                type_=str,
                default_value="bfloat16",
                description="Model data type. Use 'bfloat16' for bf16 models, 'float16' for fp16, 'auto' for automatic.",
            ),
            "num_calib_data": PassConfigParam(
                type_=int, default_value=128, description="Number of calibration samples. Typical: 128-512"
            ),
            "model_export": PassConfigParam(
                type_=Union[str, list],
                default_value="hf_format",
                description="Export format(s). Single string 'hf_format' or list ['hf_format', 'onnx']. Options: hf_format, onnx, gguf",
            ),
            "exclude_layers": PassConfigParam(
                type_=list,
                default_value=None,
                description="List of layers to exclude. Set to [] to exclude nothing explicitly.",
            ),
            # Quantization configuration
            "group_size": PassConfigParam(
                type_=int, default_value=128, description="Group size for quantization. Options: 32, 64, 128"
            ),
            "group_size_per_layer": PassConfigParam(
                type_=list,
                default_value=None,
                description="List of [layer_pattern, group_size] pairs. Example: [['*attention*', 64], ['*mlp*', 128]]",
            ),
            "kv_cache_dtype": PassConfigParam(
                type_=str, default_value=None, description="KV cache quantization dtype. Options: 'fp8', None"
            ),
            "min_kv_scale": PassConfigParam(
                type_=float, default_value=0.0, description="Minimum scale for KV cache quantization."
            ),
            "attention_dtype": PassConfigParam(
                type_=str, default_value=None, description="Attention quantization dtype. Options: 'fp8', None"
            ),
            # Calibration parameters
            "seq_len": PassConfigParam(type_=int, default_value=512, description="Sequence length for calibration."),
            "batch_size": PassConfigParam(type_=int, default_value=1, description="Batch size for calibration."),
            # Export parameters
            "pack_method": PassConfigParam(
                type_=str, default_value="reorder", description="Pack method. Options: 'order', 'reorder'"
            ),
            "export_weight_format": PassConfigParam(
                type_=str,
                default_value="real_quantized",
                description="Weight format. Options: 'fake_quantized', 'real_quantized'",
            ),
            "custom_mode": PassConfigParam(
                type_=str,
                default_value="quark",
                description="Export mode. Options: 'quark', 'awq', 'fp8' (legacy)",
            ),
            "torch_compile": PassConfigParam(
                type_=bool, default_value=False, description="Enable torch.compile (post-quantization)."
            ),
            "trust_remote_code": PassConfigParam(
                type_=bool, default_value=True, description="Trust remote code from HuggingFace."
            ),
            # ONNX-specific parameters (for ONNX models)
            "data_config": PassConfigParam(
                type_=Optional[Union[DataConfig, dict]],
                default_value=None,
                description="Data config for ONNX calibration.",
            ),
            "quant_mode": PassConfigParam(
                type_=str,
                default_value="static",
                search_defaults=Categorical(["dynamic", "static"]),
                description="ONNX quantization mode.",
            ),
            "quant_format": PassConfigParam(
                type_=str,
                default_value="QDQ",
                search_defaults=Categorical(["QOperator", "QDQ"]),
                description="ONNX quantization format.",
            ),
            **get_external_data_config(),
        }

    def _run_for_config(
        self, model: Union[HfModelHandler, ONNXModelHandler], config: BasePassConfig, output_model_path: str
    ) -> Union[HfModelHandler, ONNXModelHandler]:
        if isinstance(model, ONNXModelHandler):
            logger.info("[INFO] Running QuarkQuantization using Quark-ONNX API (0.10)")
            return self._run_quark_onnx(model, config, output_model_path)
        else:
            logger.info("[INFO] Running QuarkQuantization using Quark-Torch API (0.10)")
            return self._run_quark_torch(model, config, output_model_path)

    def _run_quark_onnx(
        self, model: ONNXModelHandler, config: BasePassConfig, output_model_path: str
    ) -> ONNXModelHandler:
        from quark import __version__ as QuarkVersion

        if version.parse(QuarkVersion) < version.parse("0.10.0"):
            raise ValueError("Quark ONNX Quantization requires amd-quark>=0.10.0")

        from olive.passes.quark_quantizer.onnx.quantize_quark import run_quark_quantization

        output_model_path = resolve_onnx_path(output_model_path, Path(model.model_path).name)

        new_tmp_dir = tempfile.TemporaryDirectory(prefix="olive_tmp")  # pylint: disable=R1732
        tmp_model_path = str(Path(new_tmp_dir.name) / Path(output_model_path).name)

        data_reader = None
        if config.data_config:
            data_config = validate_config(config.data_config, DataConfig)
            data_reader = data_config.to_data_container().create_calibration_dataloader()

        run_config = config.dict()
        to_delete = ["data_config", "quant_preprocess"] + list(get_external_data_config().keys())
        run_config = exclude_keys(run_config, to_delete)

        args = Namespace(
            model_input=model.model_path,
            model_output=tmp_model_path,
            calibration_data_reader=data_reader,
            **run_config,
        )

        run_quark_quantization(args)
        logger.info("[INFO] Quark quantized model saved to: %s", tmp_model_path)

        onnx_model = onnx.load(tmp_model_path)
        new_tmp_dir.cleanup()

        return model_proto_to_olive_model(onnx_model, output_model_path, config)

    def _run_quark_torch(self, model: HfModelHandler, config: BasePassConfig, output_model_path: str) -> HfModelHandler:
        """Run Quark 0.10 torch quantization using direct API calls."""
        from quark.torch import LLMTemplate, ModelQuantizer, export_gguf, export_onnx, export_safetensors

        from olive.passes.quark_quantizer.torch.language_modeling.llm_utils.data_preparation import (
            get_calib_dataloader,
        )
        from olive.passes.quark_quantizer.torch.language_modeling.llm_utils.model_preparation import (
            get_model,
            get_model_type,
            get_tokenizer,
            prepare_for_moe_quant,
        )

        output_dir = Path(output_model_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        # 1. Load model
        logger.info("[INFO] Loading model from: %s", model.model_path)
        torch_model, model_dtype = get_model(
            str(model.model_path),
            config.data_type,
            device,
            multi_gpu=False,
            multi_device=False,
            attn_implementation="eager",
            trust_remote_code=config.trust_remote_code,
        )

        # Handle quant_algo as string or list
        quant_algo_list = None
        if config.quant_algo:
            if isinstance(config.quant_algo, str):
                quant_algo_list = config.quant_algo.split(",") if "," in config.quant_algo else [config.quant_algo]
            elif isinstance(config.quant_algo, list):
                quant_algo_list = config.quant_algo

        prepare_for_moe_quant(torch_model, quant_algo_list)

        model_type = get_model_type(torch_model)
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

        # 3. Setup quantization config using LLMTemplate (Quark 0.10 API)
        logger.info("[INFO] Setting up quantization configuration using LLMTemplate")

        template_scheme = self._map_quant_scheme_to_template(config.quant_scheme, config.group_size)

        model_config_type = (
            torch_model.config.model_type
            if hasattr(torch_model.config, "model_type")
            else torch_model.config.architectures[0]
        )

        template = LLMTemplate.get(model_config_type)

        # Setup layer-specific config if provided
        layer_config = {}
        if config.group_size_per_layer:
            for layer_pattern, layer_group_size in config.group_size_per_layer:
                layer_group_size = int(layer_group_size)
                layer_quant_scheme = self._map_quant_scheme_to_template(config.quant_scheme, layer_group_size)
                layer_config[layer_pattern] = layer_quant_scheme

        # Get quantization config from template
        quant_config = template.get_config(
            scheme=template_scheme,
            algorithm=quant_algo_list,
            kv_cache_scheme=config.kv_cache_dtype,
            min_kv_scale=config.min_kv_scale,
            layer_config=layer_config if layer_config else None,
            attention_scheme=config.attention_dtype,
            exclude_layers=config.exclude_layers,
        )

        # 4. Quantize model using ModelQuantizer (Quark 0.10 API)
        logger.info("[INFO] Starting model quantization")
        quantizer = ModelQuantizer(quant_config, multi_device=False)
        torch_model = quantizer.quantize_model(torch_model, calib_dataloader)

        # 5. Freeze model (CRITICAL: required before export in Quark 0.10)
        logger.info("[INFO] Freezing quantized model for export")
        torch_model = quantizer.freeze(torch_model)

        # 6. Validate export configuration
        if config.custom_mode != "quark" and config.export_weight_format == "fake_quantized":
            raise ValueError("Exporting with 'fake_quantized' only supports custom_mode='quark'")

        # 7. Export model
        logger.info("[INFO] Exporting quantized model to: %s", output_model_path)

        # Handle model_export as string or list (matches standalone CLI)
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
                # For ONNX export, need input_args from calibration data
                with torch.inference_mode():
                    batch_iter = iter(calib_dataloader)
                    input_args = next(batch_iter)

                    # Check if uint4/int4 scheme
                    uint4_int4_flag = config.quant_scheme in [
                        "w_int4_per_channel_sym",
                        "w_uint4_per_group_asym",
                        "w_int4_per_group_sym",
                        "w_uint4_a_bfloat16_per_group_asym",
                    ]

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

    def _map_quant_scheme_to_template(self, quant_scheme: str, group_size: int) -> str:
        """Map Olive quant_scheme to Quark 0.10 LLMTemplate scheme."""
        SUPPORTED_QUANT_SCHEMES = {
            "w_int4_per_group_sym": "int4_wo",
            "w_uint4_per_group_asym": "uint4_wo",
            "w_int8_a_int8_per_tensor_sym": "int8",
            "w_fp8_a_fp8": "fp8",
            "w_mxfp4_a_mxfp4": "mxfp4",
            "w_mxfp6_e3m2_a_mxfp6_e3m2": "mxfp6_e3m2",
            "w_mxfp6_e2m3_a_mxfp6_e2m3": "mxfp6_e2m3",
            "w_bfp16_a_bfp16": "bfp16",
            "w_mx6_a_mx6": "mx6",
        }

        SUPPORT_GROUP_SIZE = [32, 64, 128]

        if quant_scheme not in SUPPORTED_QUANT_SCHEMES:
            raise ValueError(f"Unsupported quant_scheme: {quant_scheme}")

        if quant_scheme in ["w_int4_per_group_sym", "w_uint4_per_group_asym"]:
            if group_size in SUPPORT_GROUP_SIZE:
                return f"{SUPPORTED_QUANT_SCHEMES[quant_scheme]}_{group_size}"
            else:
                raise ValueError(f"Unsupported group_size: {group_size} for quant_scheme: {quant_scheme}")

        return SUPPORTED_QUANT_SCHEMES[quant_scheme]