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
    // Declare input path, directly use the path
    path wsi_file

    output:
    // Declare output path
    path "${params.output_dir}/wsi_thumbnail_output.png"

    script:
    """
    python /home/ubuntu/sudarsan/tiatoolbox/tiatoolbox/examples/examples_py/tiatoolbox_impart/wsi.py \
    --input ${wsi_file} --output ${params.output_dir}/wsi_thumbnail_output.png
    """
}
