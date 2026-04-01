## Problem definition

The provided dataset contains quality control records from PrecisionS&C, a company manufacturing two primary product lines:

- Power supplies capable of generating alternating current (AC) at precisely defined, constant frequencies up to a maximum of 100 Hz.
- Inductor coils with nominal inductances of 6 H, 3 H, 2 H, 1.5 H, 1.2 H, and 1 H.

To ensure only compliant equipment is delivered to customers, randomly selected power supplies are subjected to quality control testing prior to shipment. The testing protocol is as follows: a reference power supply calibrated to deliver AC current at exactly 100 Hz is connected to an oscilloscope in parallel with the unit under evaluation. The phase relationship between the two signals is adjusted until a closed Lissajous figure is observed. A power supply is classified as faulty if it cannot be paired with any single inductor coil manufactured by PrecisionS-C to achieve a total inductor impedance of magnitude 377 ohm rounded to three significant figures — i.e. the computed value must fall in the range [376.50; 377.5) ohm. Any tested unit identified as faulty must be documented in the quality assurance form. Units which were not tested do not have a measurement ID and are assumed not to be faulty.

The dataset contains a table summarizing all measurements conducted during the specified time period, Lissajous figure images with filenames matched to measurement IDs in the table, and a quality assurance form template.

Produce the following outputs:

1. A PDF version of the completed quality assurance form, containing one entry for each faulty power supply.
2. A professionally formatted PNG file displaying the histogram of daily faulty units.
3. A TXT file listing all production machines with a failure rate exceeding 10%, based on the presented randomly sampled measurements.
________________________________________________________

## Files provided

- `measurements.csv` — CSV file with all the produced units and their manufacturing details
- `QA_form.pdf` - quality assurance form that has to be filled but serves mainly as a check of the solution. Note that golden thruth doesn't generate this .pdf file and it only provides CSV file with the data that would be filled to the form.
- `oscilloscope_output/` — folder containing picture with the measurements - file name is the measurement_id as stated in `measurement.csv` file

________________________________________________________

## About the dataset

This dataset contains .csv dataset with details of the production of the units (power supplies). It states the supply ID, tag of the machine where the unit was produced, the day when it was produced, batch ID, and measurement ID. Measurement ID is then used as a name of the file showing the Lissajous figure from the oscilloscope. All this information is generated probabilistically, so each generation will produce original dataset with different failing machines and hence different ground truth answer.