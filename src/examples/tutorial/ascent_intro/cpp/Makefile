###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.
###############################################################################

###############################################################################
#
# Makefile for CPP Ascent examples.
#
# To build all examples:
#  env ASCENT_DIR={ascent install path} make
#
# From within an ascent install:
#  make
#
# Which corresponds to:
#
#  make ASCENT_DIR=../../../../..
#
###############################################################################

ASCENT_DIR ?= ../../../../../

# See $(ASCENT_DIR)/share/ascent/ascent_config.mk for detailed linking info
include $(ASCENT_DIR)/share/ascent/ascent_config.mk

# make 'all' the default target
.PHONY: default
default: all

CXX_FLAGS = -std=c++11
INC_FLAGS = $(ASCENT_INCLUDE_FLAGS)
LNK_FLAGS = $(ASCENT_LINK_RPATH) $(ASCENT_LIB_FLAGS)

# ascent first light (1)
ascent_first_light_example:
	$(CXX) $(CXX_FLAGS) $(INC_FLAGS) ascent_first_light_example.cpp $(LNK_FLAGS) -o ascent_first_light_example

# conduit examples (1-4)
conduit_examples: conduit_example1 conduit_example2 conduit_example3 conduit_example4
	echo "[done building conduit examples]"

conduit_example1:
	$(CXX) $(CXX_FLAGS) $(INC_FLAGS) conduit_example1.cpp $(LNK_FLAGS) -o conduit_example1

conduit_example2:
	$(CXX) $(CXX_FLAGS) $(INC_FLAGS) conduit_example2.cpp $(LNK_FLAGS) -o conduit_example2

conduit_example3:
	$(CXX) $(CXX_FLAGS) $(INC_FLAGS) conduit_example3.cpp $(LNK_FLAGS) -o conduit_example3

conduit_example4:
	$(CXX) $(CXX_FLAGS) $(INC_FLAGS) conduit_example4.cpp $(LNK_FLAGS) -o conduit_example4

# blueprint examples (1-3)
blueprint_examples: blueprint_example1 blueprint_example2 blueprint_example3
	echo "[done building blueprint examples]"

blueprint_example1:
	$(CXX) $(CXX_FLAGS) $(INC_FLAGS) blueprint_example1.cpp $(LNK_FLAGS) -o blueprint_example1

blueprint_example2:
	$(CXX) $(CXX_FLAGS) $(INC_FLAGS) blueprint_example2.cpp $(LNK_FLAGS) -o blueprint_example2

blueprint_example3:
	$(CXX) $(CXX_FLAGS) $(INC_FLAGS) blueprint_example3.cpp $(LNK_FLAGS) -o blueprint_example3

#ascent scene examples (1-4)
ascent_scene_examples: ascent_scene_example1 ascent_scene_example2 ascent_scene_example3 ascent_scene_example4
	echo "[done building scene examples]"

ascent_scene_example1:
	$(CXX) $(CXX_FLAGS) $(INC_FLAGS) ascent_scene_example1.cpp $(LNK_FLAGS) -o ascent_scene_example1

ascent_scene_example2:
	$(CXX) $(CXX_FLAGS) $(INC_FLAGS) ascent_scene_example2.cpp $(LNK_FLAGS) -o ascent_scene_example2

ascent_scene_example3:
	$(CXX) $(CXX_FLAGS) $(INC_FLAGS) ascent_scene_example3.cpp $(LNK_FLAGS) -o ascent_scene_example3

ascent_scene_example4:
	$(CXX) $(CXX_FLAGS) $(INC_FLAGS) ascent_scene_example4.cpp $(LNK_FLAGS) -o ascent_scene_example4

#ascent pipeline examples (1-3)
ascent_pipeline_examples: ascent_pipeline_example1 ascent_pipeline_example2 ascent_pipeline_example3
	echo "[done building pipeline examples]"

ascent_pipeline_example1:
	$(CXX) $(CXX_FLAGS) $(INC_FLAGS) ascent_pipeline_example1.cpp $(LNK_FLAGS) -o ascent_pipeline_example1

ascent_pipeline_example2:
	$(CXX) $(CXX_FLAGS) $(INC_FLAGS) ascent_pipeline_example2.cpp $(LNK_FLAGS) -o ascent_pipeline_example2

ascent_pipeline_example3:
	$(CXX) $(CXX_FLAGS) $(INC_FLAGS) ascent_pipeline_example3.cpp $(LNK_FLAGS) -o ascent_pipeline_example3


#ascent extract examples (1-5)
ascent_extract_examples: ascent_extract_example1 ascent_extract_example2 ascent_extract_example3 ascent_extract_example4 ascent_extract_example5
	echo "[done building extract examples]"

ascent_extract_example1:
	$(CXX) $(CXX_FLAGS) $(INC_FLAGS) ascent_extract_example1.cpp $(LNK_FLAGS) -o ascent_extract_example1

ascent_extract_example2:
	$(CXX) $(CXX_FLAGS) $(INC_FLAGS) ascent_extract_example2.cpp $(LNK_FLAGS) -o ascent_extract_example2

ascent_extract_example3:
	$(CXX) $(CXX_FLAGS) $(INC_FLAGS) ascent_extract_example3.cpp $(LNK_FLAGS) -o ascent_extract_example3

ascent_extract_example4:
	$(CXX) $(CXX_FLAGS) $(INC_FLAGS) ascent_extract_example4.cpp $(LNK_FLAGS) -o ascent_extract_example4

ascent_extract_example5:
	$(CXX) $(CXX_FLAGS) $(INC_FLAGS) ascent_extract_example5.cpp $(LNK_FLAGS) -o ascent_extract_example5

#ascent query examples (1)
ascent_query_examples: ascent_query_example1
	echo "[done building query examples]"

ascent_query_example1:
	$(CXX) $(CXX_FLAGS) $(INC_FLAGS) ascent_query_example1.cpp $(LNK_FLAGS) -o ascent_query_example1

ascent_binning_examples: ascent_binning_example1
	echo "[done building binning examples]"

ascent_binning_example1:
	$(CXX) $(CXX_FLAGS) $(INC_FLAGS) ascent_binning_example1.cpp $(LNK_FLAGS) -o ascent_binning_example1

#ascent trigger examples (1)
ascent_trigger_examples: ascent_trigger_example1
	echo "[done building trigger examples]"

ascent_trigger_example1:
	$(CXX) $(CXX_FLAGS) $(INC_FLAGS) ascent_trigger_example1.cpp $(LNK_FLAGS) -o ascent_trigger_example1


all:  ascent_first_light_example conduit_examples blueprint_examples ascent_scene_examples ascent_pipeline_examples ascent_extract_examples ascent_query_examples ascent_binning_examples ascent_trigger_examples
	echo "[done building all examples]"

clean:
	rm -f ascent_first_light_example
	rm -f blueprint_example1
	rm -f blueprint_example2
	rm -f blueprint_example3
	rm -f conduit_example1
	rm -f conduit_example2
	rm -f conduit_example3
	rm -f conduit_example4
	rm -f ascent_scene_example1
	rm -f ascent_scene_example2
	rm -f ascent_scene_example3
	rm -f ascent_scene_example4
	rm -f ascent_pipeline_example1
	rm -f ascent_pipeline_example2
	rm -f ascent_pipeline_example3
	rm -f ascent_extract_example1
	rm -f ascent_extract_example2
	rm -f ascent_extract_example3
	rm -f ascent_extract_example4
	rm -f ascent_extract_example5
	rm -f ascent_query_example1
	rm -f ascent_binning_example1
	rm -f ascent_trigger_example1
