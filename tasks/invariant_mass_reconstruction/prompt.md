## Problem definition

Provided input contains data from the particle detector similar to the detectors ATLAS or CMS  at the LHC at CERN. The detector surrounds the beam pipe, and proton-proton collisions occur at the origin of the coordinate system (0, 0, 0).

The detector has cylindrical shape and it has multiple tracking layers and one electromagnetic calorimeter layer. We define coordinates $x$, $y$ in the transverse plane (plane perpendicular to the beam pipe) with $x=0$, $y=0$ at the beam pipe and $z$ pointing along the beam pipe, so that $x-y-z$ form right-handed coordinate system. $z=0$ is at the centre of the detector. Pseudo-rapidity $\eta$ is defined in analogy with big the LHC experiemnts - line perpendicular to the beam pipe going through $z=0$ has $\eta = 0$, and directions along the beam pipe as seen from the origin has $\eta = \pm \infty$. 

The exact detector geometry with visualisation of the coordinate system is dispalyed in input image `detector_longitudinal.png` showing the cross section of the half of the detector in the $r-z$ plane, and in `detector_transverse.png` showing the cross section through the detector in transverse $x-y$ plane.

Provided data contain information from the tracking detector and calorimeter from events with $e^+e^-$ pair coming from the common vertex located exactly at the origin (0, 0, 0). These events were selected such that both electron and positron left hit (signal deposit) in every tracking layer and energy deposit in the calorimeter layer. These event's were cleaned from hits comming from other particles, so it is possible to assume that each of the hits in these events can be attributed to the electron or positron decying from the parent particle. It can be assumed that the measurements from tracker and calorimeter have absolute precision.

Assume that you know that some electron-positron pairs come from the new unknown particle, while the rest comes from the exponentially decying background. Estimate mass $M$ and decay width $\Gamma$ of the particle in $\mathrm{GeV/c^2}$ given the provided data.

________________________________________
## Output format

The final output is used to grade your answer. 

Structure your response according to the following template:

SUMMARY:
[high level summary of your approach, including key formulas used in the task and used constants (such as particle masses or detector parameters)]

INTERMEDIATE RESULTS PER EVENT:
event ID , reconstructed invariant mass 
[value] , [value]
[value] , [value]
[value] , [value]
...

FINAL RESULTS:
unknown particle mass: [value] $\mathrm{GeV/c^2}$
unknown particle decay width: [value] $\mathrm{GeV/c^2}$

________________________________________
## Files provided

<Plese, describe structure of the input data>

- `detector_longitudinal.png` — 
- `detector_transverse.png` — 
- `ecal_hits.csv` — 
- `tracker_hits.csv` -