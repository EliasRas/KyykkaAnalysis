# KyykkaAnalysis
Analysis of kyykka play times

## CLI
The analysis can be run by calling `python -m kyykkaanalysis`. The package uses the following arguments
- input_file
    - Path to the CSV file which contains the input data.
    - The format of the file is explained in [Data file formats](#data-file-formats) section.
- team_file
    - Path to the CSV file which contains the teams for the player.
    - The format of the file is explained in [Data file formats](#data-file-formats) section.
- cache_directory
    - Path to the directory to which the cached data is saved.
    - The samples generated in pipelines `prior`, `sbc` and `posterior` are saved into this directory as NetCDF files
- figure_directory
    - Path to the directory to which the visualizations are saved.
    - The plotly figures created in pipelines `data`, `prior`, `sbc` and `posterior` are saved here inside respective subdirectories as either HTML or PDF files.
- pipelines
    - List of pipelines to run.
    - Allowed values for pipelines includes:
        - `data` for printing information and creating visualizations of the data.
        - `prior` for creating visualizations of the prior distributions defined by the models.
        - `sbc` for performing and visualizing simulation based calibration.
        - `posterior` for performing posterior inference, cross-validation and model comparisons.

## Data file formats
TODO