#include "kernel.cuh"

struct parallelepiped
{
	static const size_t edges = 12;
	vect3 color;
	vect3 p, q, r, s;
	polygon pols[edges];
	vect3 center;

	//__host__ __device__ parallelepiped(vect3 p, vect3 q, vect3 r, vect3 s);
	__host__ __device__ parallelepiped(vect3 center_, float x = 1, float y = 1, float z = 1, vect3 color_ = {255, 255, 255});

	__host__ __device__ void update_pols();

	//__host__ __device__ vect3 center();
};

struct icosahedron
{
	static const size_t edges = 20;
	vect3 color;
	vect3 n, s;
	float a;
	polygon pols[edges];
	vect3 center;

	__host__ __device__ icosahedron(vect3 center_, float a, vect3 color_ = { 255, 255, 255 });
	__host__ __device__ void update_pols();
};

//__host__ __device__ parallelepiped rotate(parallelepiped* par, float aplha, float beta, float gamma);