#!/usr/bin/env nextflow

// Set up input and output directories as parameters
params.input_dir = "./input_data"
params.output_dir = "./output_data"
params.scripts_dir = "./scripts"
params.conda_env = "./conda.yml"

// Workflow definition
workflow {
    process_wsi_reading()
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