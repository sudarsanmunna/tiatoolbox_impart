#!/usr/bin/env nextflow

//nextflow.enable.dsl=2

// Define the input file (WSI file) and output folder
params.wsi_file = "/home/ubuntu/sudarsan/tiatoolbox/tiatoolbox/examples/examples_py/tiatoolbox_impart/63.svs"
params.output_dir = "/home/ubuntu/sudarsan/tiatoolbox/tiatoolbox/examples/examples_py/tiatoolbox_impart"


// Workflow definition
workflow {
    process_wsi()
}


// Process to handle WSI using tiatoolbox Python script
process process_wsi {

    input:
    path wsi_file from file(params.wsi_file)  // Corrected input directive

    output:
    path "${params.output_dir}/wsi_thumbnail_output.png" into output_thumbnail

    script:
    """
    python /home/ubuntu/sudarsan/tiatoolbox/tiatoolbox/examples/examples_py/tiatoolbox_impart/wsi.py \
    --input ${wsi_file} --output ${params.output_dir}/wsi_thumbnail_output.png
    """
}
