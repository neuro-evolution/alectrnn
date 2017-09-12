# probably want to import something like cmaes
# probably want to import some other stuff too like numpy
# not sure if I want main or interaface for main

# possibility
# alectrnn could just be a wrapper for everything else
	# it just is a class with a big list of cmaes settings
	# agent settings
	# objective settings
	# ALE settings
	# graph options

# Actual main would just import alectrnn setup dictionary or something
# and run alectrnn.




# Import CMAES
# Import Agent
# Import ALE
# Import Objective

# CMAES Param dict -> pass unpacked keys
# Make CMAES Object
# ALE Param dict -> pass upacked keys
# Make ALE Object
# Agent Param dict -> pass ALE-obj -> pass unpacked keys
# Make Agent Object
# Pass Objective to CMAES-object
# Pass Objective arguments (ALE-object, Agent-object) to CMAES-object
# Engage CMAES-object

## NOTE! If CMAES is set to partial centroid mutation, and centroid is 
## initialized to zero, then we effectively start the network off sparse
