#!/usr/bin/env nextflow

nextflow.enable.dsl=2

// Define input parameters
params.wsi_file = "/home/ubuntu/sudarsan/tiatoolbox/tiatoolbox/examples/examples_py/tiatoolbox_impart/63.svs"
params.output_dir = "/home/ubuntu/sudarsan/tiatoolbox/tiatoolbox/examples/examples_py/tiatoolbox_impart"

// Create a channel from the wsi_file path
Channel
    .fromPath(params.wsi_file)
    .set { wsi_channel }

// Workflow definition
workflow {
    // Call the process and pass the channel
    process_wsi(wsi_channel)
}

// Process to handle WSI using tiatoolbox Python script
process process_wsi {

    input:
    path wsi_file

    output:
    // Output file will be created in the work directory first
    path 'wsi_thumbnail_output.png'

    script:
    """
    python /home/ubuntu/sudarsan/tiatoolbox/tiatoolbox/examples/examples_py/tiatoolbox_impart/wsi.py \
    --input ${wsi_file} --output wsi_thumbnail_output.png
    """
}

workflow.onComplete {
    // Move the result to the final output directory
    exec:
    """
    mv work/*/wsi_thumbnail_output.png ${params.output_dir}/wsi_thumbnail_output.png
    """
}
