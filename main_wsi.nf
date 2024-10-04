#!/usr/bin/env nextflow

nextflow.enable.dsl=2

// Define input parameters
params.wsi_file = "/home/ubuntu/sudarsan/tiatoolbox/tiatoolbox/examples/examples_py/tiatoolbox_impart/63.svs"
params.output_dir = "/home/ubuntu/sudarsan/tiatoolbox/tiatoolbox/examples/examples_py/tiatoolbox_impart"

// Workflow definition
workflow {
    // Create a channel from the wsi_file
    Channel
        .fromPath(params.wsi_file)
        .set { wsi_channel }

    // Pass the file to the process
    process_wsi(wsi_channel)
}

// Process to handle WSI using tiatoolbox Python script
process process_wsi {

    input:
    // Correct way to declare file input using 'from' a channel
    path wsi_file from wsi_channel

    output:
    // Declare output path (it will create the file inside the output directory)
    path "${params.output_dir}/wsi_thumbnail_output.png"

    script:
    """
    python /home/ubuntu/sudarsan/tiatoolbox/tiatoolbox/examples/examples_py/tiatoolbox_impart/wsi.py \
    --input ${wsi_file} --output ${params.output_dir}/wsi_thumbnail_output.png
    """
}
