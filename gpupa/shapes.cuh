#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "kernel.cuh"

struct parallelepiped
{
	vect3 p1, p2, p3;


	parallelepiped(vect3 p, vect3 q, vect3 r);
};