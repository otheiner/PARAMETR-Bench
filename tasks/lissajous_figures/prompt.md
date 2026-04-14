## Problem definition

The provided dataset contains quality control records from PrecisionS&C, a company manufacturing two primary product lines:

- Power supplies capable of generating alternating current (AC) at precisely defined, constant frequencies up to a maximum of 100 Hz.
- Inductor coils with nominal inductances of 6 H, 3 H, 2 H, 1.5 H, 1.2 H, and 1 H.

To ensure only compliant equipment is delivered to customers, randomly selected power supplies are subjected to quality control testing prior to shipment. The testing protocol is as follows: a reference power supply calibrated to deliver AC current at exactly 100 Hz is connected to an oscilloscope in parallel with the unit under evaluation. The phase relationship between the two signals is adjusted until a closed Lissajous figure is observed. A power supply is classified as faulty if it cannot be paired with any single inductor coil manufactured by PrecisionS&C to achieve a total inductor impedance of magnitude 377 ohm rounded to three significant figures — i.e. the computed value must fall in the range [376.50; 377.5) ohm. Any tested unit identified as faulty must be documented for quality assurance. 

Quality assurance requires listing for each supply which was identified faulty: 
    - supply ID
    - manufacturing machine
    - batch
    - measured frequency of faulty supply in Hz, rounded to the closest integer

Units which were not tested do not have a measurement ID and are assumed not to be faulty.

Management of the company also wants to list all machines with a failure rate strictly greater than 10%, based on the presented randomly sampled measurements.

________________________________________
## Output format

Structure your response exactly as follows:

<final_result> 
-------DATA FOR QUALITY ASSURANCE-------
SUPPLY_ID      MACHINE      BATCH      FREQUENCY
<supply_ID1>  <machine>    <batch>     <frequency>
<supply_ID2>  <machine>    <batch>     <frequency>
...

------DATA FOR MANAGEMENT------
MACHINES(failure > 10%)
<machine1>
<machine2>
...
</final_result> 

________________________________________
## Files provided

The dataset containing: 
    - Table summarizing all measurements conducted during the specified time period
    - Lissajous figure images with filenames matched to measurement IDs in the table

- `measurements.csv` — Table summarizing all measurements conducted during the specified time period.
- `oscilloscope_output/` — Folder containing .png files from the oscilloscope with measured Lissajous figure - name of the files corresponds to the measurement IDs.