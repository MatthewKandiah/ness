To do list:
	change convergence condition to look for convergence in entanglement of formation

	refactor code: create classes for the mapping and Liouvillian to make code more readable and reusable

	make separate scripts for running locally and running on CSF:
		1. Local can use for loops, incrementing RC dimension until convergence is found. CSF wants a single variable array which can then be submitted in parallel to the supercomputer. This means that the CSF version will not be using any sort of convergence criteria, it will just calculate the values we tell it to. The local version can be used to test cases for convergence.
		2. Both versions probably need to use iterative methods due to memory bottleneck in building the Liouvillian. Local version might need to use a prefactor. This shortens computing time at the expense of memory overhead. CSF version should not use a prefactor, since computations can be carried out in parallel, the trade off is probably not worth it.  