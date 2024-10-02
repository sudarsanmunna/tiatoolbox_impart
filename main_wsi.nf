#!/usr/bin/env nextflow

nextflow.enable.dsl=2

params.wsi = '/home/ubuntu/bala/bala/ImpartLabs/tmp/DI_dombox2_0006.svs'  // Define the path to a single WSI file
params.outdir = '/home/ubuntu/sudarsan/tiatoolbox/tiatoolbox/examples/examples_py/tiatoolbox_impart/results'  // Define the output folder

process RunWSIPythonScript {

    input:
    path wsi_file  // WSI file to be processed
    
    output:
    path "${params.outdir}/output_${wsi_file.baseName}.png"

    script:
    """
    mkdir -p ${params.outdir}
    python 01-wsi-reading.py --input ${wsi_file} --output ${params.outdir}/output_${wsi_file.baseName}.png
    """
}

workflow {

    // Since you are providing a single file, we can directly pass the file to the process.
    Channel
        .fromPath(params.wsi)  // Using single file instead of a folder pattern
        .set { wsi_files }

    // Run the WSI Python analysis script for each WSI file
    wsi_files
        | RunWSIPythonScript
}
