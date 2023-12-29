from llmfoundry.models.hf.hf_as_mpt import LlamaAsMPTForCausalLM
from transformers.models.auto.modeling_auto import \
        MODEL_FOR_CAUSAL_LM_MAPPING

from transformers.models.llama.configuration_llama import LlamaConfig

HF_AS_MPT_PATCH_MAPPING = {
    'llama': (LlamaConfig, LlamaAsMPTForCausalLM),
    'codellama': (LlamaConfig, LlamaAsMPTForCausalLM),
}

def patch_hf_with_mpt(model_type: str):   
    if model_type not in HF_AS_MPT_PATCH_MAPPING:
        raise ValueError(f'{model_type=} is not supported for MPT patching. ' +
                         f'Valid model types are {HF_AS_MPT_PATCH_MAPPING.keys()}')

    config_class_to_patch, model_class_to_patch_with = HF_AS_MPT_PATCH_MAPPING[model_type]

    MODEL_FOR_CAUSAL_LM_MAPPING._extra_content[config_class_to_patch] = model_class_to_patch_with

def undo_hf_with_mpt_patch(model_type: str):
    if model_type not in HF_AS_MPT_PATCH_MAPPING:
        raise ValueError(f'{model_type=} is not supported for MPT patching. ' +
                         f'Valid model types are {HF_AS_MPT_PATCH_MAPPING.keys()}')

    config_class_to_patch, _ = HF_AS_MPT_PATCH_MAPPING[model_type]

    MODEL_FOR_CAUSAL_LM_MAPPING._extra_content.pop(config_class_to_patch)