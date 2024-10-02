#!/usr/bin/env nextflow

nextflow.enable.dsl=2

params.wsi = '/home/ubuntu/bala/bala/ImpartLabs/tmp/DI_dombox2_0006.svs'  // Path to a single WSI file
params.outdir = '/home/ubuntu/sudarsan/tiatoolbox/tiatoolbox/examples/examples_py/tiatoolbox_impart/results'  // Final output directory

process RunWSIPythonScript {

    input:
    path wsi_file  // WSI file to be processed
    
    output:
    path "output_${wsi_file.baseName}.png" into result_files  // Output file saved in work directory first

    script:
    """
    python 01-wsi-reading.py --input ${wsi_file} --output output_${wsi_file.baseName}.png
    """
}

process MoveResults {

    input:
    path result_files

    script:
    """
    mkdir -p ${params.outdir}
    mv ${result_files} ${params.outdir}/
    """
}

workflow {

    // Use a single file as input
    Channel
        .fromPath(params.wsi)  // Load the single WSI file
        .set { wsi_files }

    // Run the WSI Python analysis script for each WSI file
    wsi_files
        | RunWSIPythonScript

    // Move results to the final output directory
    result_files
        | MoveResults
}
