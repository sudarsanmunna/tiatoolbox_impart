#!/usr/bin/env nextflow

nextflow.enable.dsl=2

params.wsi = 'input_wsi_folder'  // Define the folder with input WSIs
params.outdir = 'results'  // Define the output folder

//process InstallPythonPackages {
//    """
 //   # Install necessary packages for the Python environment
//    sudo apt-get update
//    sudo apt-get -y install libopenjp2-7-dev libopenjp2-tools openslide-tools libpixman-1-dev
 //   pip install git+https://github.com/TissueImageAnalytics/tiatoolbox.git@develop
 //   """
//}

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

    // Define the input channel with all WSI files in the input directory
    Channel
        .fromPath("${params.wsi}/*.svs")  // Adjust the extension to match your WSI files
        .set { wsi_files }

    // Install dependencies
    //InstallPythonPackages()

    // Run the WSI Python analysis script for each WSI file
    wsi_files
        | RunWSIPythonScript
}
