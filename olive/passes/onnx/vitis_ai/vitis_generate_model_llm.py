#
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import logging
from pathlib import Path

from olive.model import HfModelHandler, ONNXModelHandler
from olive.passes import Pass
from olive.passes.pass_config import BasePassConfig, PassConfigParam

logger = logging.getLogger(__name__)


class VitisGenerateModelLLM(Pass):
    @classmethod
    def _default_config(cls, accelerator_spec):
        return {
            "packed_const": PassConfigParam(
                type_=bool,
                default_value=False,
                description="Enable packed constants in NPU export.",
            ),
            "cpu_only": PassConfigParam(
                type_=bool,
                default_value=False,
                description="Run only model builder OGA CPU only model, skip NPU-related steps.",
            ),
            "optimize": PassConfigParam(
                type_=str,
                default_value="decode",
                description="Optimization mode for decode fusion",
            ),
            "script_option": PassConfigParam(
                type_=str,
                default_value="jit_npu",
                description="Script variant: 'jit_npu' or 'non_jit'.",
            ),
            "max_seq_len": PassConfigParam(
                type_=int,
                default_value=4096,
                description="Maximum sequence length for optimization.",
            ),
            "use_ep": PassConfigParam(
                type_=bool,
                default_value=True,
                description="Enable EP",
            ),
            "basic": PassConfigParam(
                type_=bool,
                default_value=False,
                description="Use basic NPU flow.",
            ),
            "npu_op_version": PassConfigParam(
                type_=str,
                default_value="v2",
                description="NPU LLM op version: 'v2'",
            ),
        }

    def _run_for_config(
        self, model: HfModelHandler, config: BasePassConfig, output_model_path: str
    ) -> ONNXModelHandler:
        logger.info("[DEBUG] Running VitisGenerateModelLLM with config: %s", config)
        from model_generate import generate_npu_model

        input_model_path = model.model_path
        output_dir = Path(output_model_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("[VitisGenerateModelLLM] Generating Vitis NPU model from: %s", input_model_path)
        logger.info("[VitisGenerateModelLLM] Output directory: %s", output_dir)

        # Generate the NPU model with decode optimization and EP flow
        generate_npu_model(
            input_model=str(input_model_path),
            output_dir=str(output_dir),
            packed_const=config.packed_const,
            script_option=config.script_option,
            cpu_only=config.cpu_only,
            optimize=config.optimize,
            max_seq_len=config.max_seq_len,
            npu_op_version=config.npu_op_version,
            basic=config.basic,
            use_ep=config.use_ep,
        )
        
        # decode optimization produces fusion.onnx
        onnx_file_name = "fusion.onnx"
        
        logger.info("[VitisGenerateModelLLM] Using output model file: %s", onnx_file_name)
        
        return ONNXModelHandler(
            model_path=output_dir,
            onnx_file_name=onnx_file_name,
        )


