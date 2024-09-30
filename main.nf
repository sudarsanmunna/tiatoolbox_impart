#!/usr/bin/env nextflow

// Set up input and output directories as parameters
params.input_dir = "./input_data"
params.output_dir = "./output_data"
params.scripts_dir = "./scripts"
params.conda_env = "./conda.yml"

// Workflow definition
workflow {
    process_wsi_reading()
    process_stain_normalization()
    process_tissue_masking()
    process_patch_extraction()
    process_patch_prediction()
    process_semantic_segmentation()
    process_nucleus_instance_segmentation()
    process_advanced_modeling()
    process_wsi_registration()
    process_multi_task_segmentation()
}

// Process: WSI Reading
process process_wsi_reading {
    conda params.conda_env  // Use Conda environment

    input:
    path input_dir, mode: 'copy'

    output:
    path "${params.output_dir}/wsi_reading_output"

    script:
    """
    python ${params.scripts_dir}/01-wsi-reading.py --input $input_dir --output ${params.output_dir}/wsi_reading_output
    """
}

// Process: Stain Normalization
process process_stain_normalization {
    conda params.conda_env

    input:
    path "${params.output_dir}/wsi_reading_output"

    output:
    path "${params.output_dir}/stain_normalization_output"

    script:
    """
    python ${params.scripts_dir}/02-stain-normalization.py --input ${params.output_dir}/wsi_reading_output --output ${params.output_dir}/stain_normalization_output
    """
}

// Process: Tissue Masking
process process_tissue_masking {
    conda params.conda_env

    input:
    path "${params.output_dir}/stain_normalization_output"

    output:
    path "${params.output_dir}/tissue_masking_output"

    script:
    """
    python ${params.scripts_dir}/03-tissue-masking.py --input ${params.output_dir}/stain_normalization_output --output ${params.output_dir}/tissue_masking_output
    """
}

// Process: Patch Extraction
process process_patch_extraction {
    conda params.conda_env

    input:
    path "${params.output_dir}/tissue_masking_output"

    output:
    path "${params.output_dir}/patch_extraction_output"

    script:
    """
    python ${params.scripts_dir}/04-patch-extraction.py --input ${params.output_dir}/tissue_masking_output --output ${params.output_dir}/patch_extraction_output
    """
}

// Process: Patch Prediction
process process_patch_prediction {
    conda params.conda_env

    input:
    path "${params.output_dir}/patch_extraction_output"

    output:
    path "${params.output_dir}/patch_prediction_output"

    script:
    """
    python ${params.scripts_dir}/05-patch-prediction.py --input ${params.output_dir}/patch_extraction_output --output ${params.output_dir}/patch_prediction_output
    """
}

// Process: Semantic Segmentation
process process_semantic_segmentation {
    conda params.conda_env

    input:
    path "${params.output_dir}/patch_prediction_output"

    output:
    path "${params.output_dir}/semantic_segmentation_output"

    script:
    """
    python ${params.scripts_dir}/06-semantic-segmentation.py --input ${params.output_dir}/patch_prediction_output --output ${params.output_dir}/semantic_segmentation_output
    """
}

// Process: Nucleus Instance Segmentation
process process_nucleus_instance_segmentation {
    conda params.conda_env

    input:
    path "${params.output_dir}/semantic_segmentation_output"

    output:
    path "${params.output_dir}/nucleus_instance_segmentation_output"

    script:
    """
    python ${params.scripts_dir}/07-nucleus-instance-segmentation.py --input ${params.output_dir}/semantic_segmentation_output --output ${params.output_dir}/nucleus_instance_segmentation_output
    """
}

// Process: Advanced Modeling
process process_advanced_modeling {
    conda params.conda_env

    input:
    path "${params.output_dir}/nucleus_instance_segmentation_output"

    output:
    path "${params.output_dir}/advanced_modeling_output"

    script:
    """
    python ${params.scripts_dir}/08-advanced-modeling.py --input ${params.output_dir}/nucleus_instance_segmentation_output --output ${params.output_dir}/advanced_modeling_output
    """
}

// Process: WSI Registration
process process_wsi_registration {
    conda params.conda_env

    input:
    path "${params.output_dir}/advanced_modeling_output"

    output:
    path "${params.output_dir}/wsi_registration_output"

    script:
    """
    python ${params.scripts_dir}/09-wsi-registration.py --input ${params.output_dir}/advanced_modeling_output --output ${params.output_dir}/wsi_registration_output
    """
}

// Process: Multi-Task Segmentation
process process_multi_task_segmentation {
    conda params.conda_env

    input:
    path "${params.output_dir}/wsi_registration_output"

    output:
    path "${params.output_dir}/multi_task_segmentation_output"

    script:
    """
    python ${params.scripts_dir}/10-multi-task-segmentation.py --input ${params.output_dir}/wsi_registration_output --output ${params.output_dir}/multi_task_segmentation_output
    """
}
