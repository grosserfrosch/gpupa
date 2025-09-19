#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>


__host__ __device__ bool equ(float a, float b);

__host__ __device__ bool les_equ(float a, float b);

__host__ __device__ bool gre_equ(float a, float b);

struct vect3
{
	float x, y, z;
	float len;

	__host__ __device__ vect3(float x_, float y_, float z_);
	__host__ __device__ vect3(float val);
	__host__ __device__ vect3();

	__host__ __device__ vect3 operator + (vect3 v);
	__host__ __device__ vect3 operator - (vect3 v);
	__host__ __device__ vect3 operator * (float val);
	__host__ __device__ vect3 operator / (float val);

	__host__ __device__ float operator * (vect3 v);
	__host__ __device__ vect3 operator ^ (vect3 v);

	__host__ __device__ bool operator == (vect3 v);
};

__host__ __device__ vect3 norm(vect3 vec);

struct line3
{
	vect3 p1, p2;

	//line defines by two points
	__host__ __device__ line3(vect3 p1_, vect3 p2_);
	__host__ __device__ line3(vect3 *p1_, vect3 *p2_);
	__host__ __device__ line3();

	__host__ __device__ bool contains(vect3 point);

	__host__ __device__ bool is_correct();
};

struct plane
{
	float A, B, C, D;
	vect3 normale;

	__host__ __device__ plane(vect3 p1, vect3 p2, vect3 p3);
	__host__ __device__ plane(float a, float b, float c, float d);
	__host__ __device__ plane();

	__host__ __device__ bool is_correct();

	__host__ __device__ float value(vect3 p);
	__host__ __device__ float value(vect3* p);
	__host__ __device__ bool contains(vect3 p);
	__host__ __device__ bool contains(vect3* p);
	__host__ __device__ bool intersects(line3 l, vect3 *p);
	__host__ __device__ bool posit_intersects(line3 l, vect3* p);
	__host__ __device__ bool intersects(line3 *l, vect3* p);
	__host__ __device__ bool posit_intersects(line3 *l, vect3* p);
	__host__ __device__ vect3 get_normale();
};

struct sphere
{
	vect3 color;
	vect3 p;
	float r;

	__host__ __device__ sphere(vect3 p_, float r_, vect3 col = {255, 255, 255});
	__host__ __device__ sphere();

	__host__ __device__ bool contains(vect3 q);
	__host__ __device__ bool intersects(line3 l, vect3 *q);
};

struct polygon
{
	plane pl;
	vect3 p1, p2, p3;
	vect3 color;

	__host__ __device__ polygon(vect3 p1_, vect3 p2_, vect3 p3_, vect3 col = {255, 255, 255});
	__host__ __device__ polygon();

	__host__ __device__ bool is_correct();

	__host__ __device__ bool contains(vect3 p);
	__host__ __device__ bool contains(vect3* p);
	__host__ __device__ bool intersects(line3 l, vect3* p);
	__host__ __device__ bool posit_intersects(line3 l, vect3* p);
	__host__ __device__ bool intersects(line3 *l, vect3* p);
	__host__ __device__ bool posit_intersects(line3 *l, vect3* p);

	__host__ __device__ vect3 normale();
};

__host__ __device__ vect3 rotation(vect3* p, float OX, float OY, float OZ, vect3 center = {0, 0, 0});

__host__ __device__ polygon rotation(polygon* pol, float OX, float OY, float OZ, vect3 center = {0, 0, 0});

//__host__ __device__ polygon rotation(polygon* pol, vect3 ort);

void p_ray_tracing(polygon* sph, unsigned int pol_num, vect3* cam, vect3* O, vect3* x, vect3* y, vect3* lights, unsigned int lights_num, std::uint8_t *disp, const unsigned int width, const unsigned int height);