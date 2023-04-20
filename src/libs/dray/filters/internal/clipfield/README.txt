This directory contains the raw hex clipping tables created for dray. These tables
are for use by VisIt's clipeditor tool, which can display and edit the table.
However, the unique cases for the table were entered by hand using clipeditor
as a guide.

Each clipping case contains a mixture of tets and pyramids. Pyramids are used
for some of the 4 sided exterior shapes because the clipeditor can reorient
them into other symmetric clipping cases. The pyramids are then post-processed
using the process_pyr.py script to make a clipping table that consists only
of tetrahedral shapes suitable for dray.

VisIt's clipping tables could not be used because they contain mixtures of
cell types and also cells such as pyramids and wedges that dray does not
support. So, a tet clipping table was developed as it was compatible with dray
and also has the flexibility needed to represent the output shapes needed.

The clipping table cases preserve the entire shape but mark the output shapes
using COLOR0 or COLOR1. This means that the code can select the clipped shape
or its inverse easily (though computing the clip case differently would work too).
This mainly to preserve compatibility with VisIt's clipeditor in case the
tables need to be changed in the future.

To regenerate the hex clipping tables:

python3 process_pyr.py > ../clip_cases_hex.cpp
