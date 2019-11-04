###############################################################################
# Copyright (c) 2015-2019, Lawrence Livermore National Security, LLC.
#
# Produced at the Lawrence Livermore National Laboratory
#
# LLNL-CODE-716457
#
# All rights reserved.
#
# This file is part of Ascent.
#
# For details, see: http://ascent.readthedocs.io/.
#
# Please also read ascent/LICENSE
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the disclaimer below.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the disclaimer (as noted below) in the
#   documentation and/or other materials provided with the distribution.
#
# * Neither the name of the LLNS/LLNL nor the names of its contributors may
#   be used to endorse or promote products derived from this software without
#   specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
# LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
# OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
# IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
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

INC_FLAGS = $(ASCENT_INCLUDE_FLAGS)
LNK_FLAGS = $(ASCENT_LINK_RPATH) $(ASCENT_LIB_FLAGS)

# ascent first light (1)
ascent_first_light_example:
	$(CXX) $(INC_FLAGS) ascent_first_light_example.cpp $(LNK_FLAGS) -o ascent_first_light_example

# conduit examples (1-4)
conduit_examples: conduit_example1 conduit_example2 conduit_example3 conduit_example4 
	echo "[done building conduit examples]"

conduit_example1:
	$(CXX) $(INC_FLAGS) conduit_example1.cpp $(LNK_FLAGS) -o conduit_example1

conduit_example2:
	$(CXX) $(INC_FLAGS) conduit_example2.cpp $(LNK_FLAGS) -o conduit_example2

conduit_example3:
	$(CXX) $(INC_FLAGS) conduit_example3.cpp $(LNK_FLAGS) -o conduit_example3

conduit_example4:
	$(CXX) $(INC_FLAGS) conduit_example4.cpp $(LNK_FLAGS) -o conduit_example4

# blueprint examples (1-3)
blueprint_examples: blueprint_example1 blueprint_example2 blueprint_example3
	echo "[done building blueprint examples]"

blueprint_example1:
	$(CXX) $(INC_FLAGS) blueprint_example1.cpp $(LNK_FLAGS) -o blueprint_example1

blueprint_example2:
	$(CXX) $(INC_FLAGS) blueprint_example2.cpp $(LNK_FLAGS) -o blueprint_example2

blueprint_example3:
	$(CXX) $(INC_FLAGS) blueprint_example3.cpp $(LNK_FLAGS) -o blueprint_example3

#ascent scene examples (1-4)
ascent_scene_examples: ascent_scene_example1 ascent_scene_example2 ascent_scene_example3 ascent_scene_example4
	echo "[done building scene examples]"

ascent_scene_example1:
	$(CXX) $(INC_FLAGS) ascent_scene_example1.cpp $(LNK_FLAGS) -o ascent_scene_example1

ascent_scene_example2:
	$(CXX) $(INC_FLAGS) ascent_scene_example2.cpp $(LNK_FLAGS) -o ascent_scene_example2

ascent_scene_example3:
	$(CXX) $(INC_FLAGS) ascent_scene_example3.cpp $(LNK_FLAGS) -o ascent_scene_example3

ascent_scene_example4:
	$(CXX) $(INC_FLAGS) ascent_scene_example4.cpp $(LNK_FLAGS) -o ascent_scene_example4

#ascent pipeline examples (1-3)
ascent_pipeline_examples: ascent_pipeline_example1 ascent_pipeline_example2 ascent_pipeline_example3
	echo "[done building pipeline examples]"

ascent_pipeline_example1:
	$(CXX) $(INC_FLAGS) ascent_pipeline_example1.cpp $(LNK_FLAGS) -o ascent_pipeline_example1

ascent_pipeline_example2:
	$(CXX) $(INC_FLAGS) ascent_pipeline_example2.cpp $(LNK_FLAGS) -o ascent_pipeline_example2

ascent_pipeline_example3:
	$(CXX) $(INC_FLAGS) ascent_pipeline_example3.cpp $(LNK_FLAGS) -o ascent_pipeline_example3


#ascent extract examples (1-5)
ascent_extract_examples: ascent_extract_example1 ascent_extract_example2 ascent_extract_example3 ascent_extract_example4 ascent_extract_example5
	echo "[done building extract examples]"

ascent_extract_example1:
	$(CXX) $(INC_FLAGS) ascent_extract_example1.cpp $(LNK_FLAGS) -o ascent_extract_example1

ascent_extract_example2:
	$(CXX) $(INC_FLAGS) ascent_extract_example2.cpp $(LNK_FLAGS) -o ascent_extract_example2

ascent_extract_example3:
	$(CXX) $(INC_FLAGS) ascent_extract_example3.cpp $(LNK_FLAGS) -o ascent_extract_example3

ascent_extract_example4:
	$(CXX) $(INC_FLAGS) ascent_extract_example4.cpp $(LNK_FLAGS) -o ascent_extract_example4

ascent_extract_example5:
	$(CXX) $(INC_FLAGS) ascent_extract_example5.cpp $(LNK_FLAGS) -o ascent_extract_example5

#ascent query examples (1)
ascent_query_examples: ascent_query_example1
	echo "[done building query examples]"

ascent_query_example1:
	$(CXX) $(INC_FLAGS) ascent_query_example1.cpp $(LNK_FLAGS) -o ascent_query_example1

#ascent trigger examples (1)
ascent_trigger_examples: ascent_trigger_example1
	echo "[done building trigger examples]"

ascent_trigger_example1:
	$(CXX) $(INC_FLAGS) ascent_trigger_example1.cpp $(LNK_FLAGS) -o ascent_trigger_example1


all:  ascent_first_light_example conduit_examples blueprint_examples ascent_scene_examples ascent_pipeline_examples ascent_extract_examples ascent_query_examples ascent_trigger_examples
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
	rm -f ascent_trigger_example1
