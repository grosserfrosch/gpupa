#include "shapes.cuh"
#include <cmath>

//do not use
//__host__ __device__ parallelepiped::parallelepiped(vect3 p, vect3 q, vect3 r, vect3 s)
//{
//	p1 = p;
//	p2 = q;
//	p3 = r;
//	p4 = s;
//
//	if (!plane{p, q, r}.contains(s))
//		update_pols();
//}

__host__ __device__ parallelepiped::parallelepiped(vect3 center_, float x, float y, float z, vect3 color_)
{
	center = center_;
	p = center + vect3{ -x / 2, -y / 2, -z / 2 };
	q = center + vect3{ -x / 2, -y / 2, z / 2 };
	r = center + vect3{ x / 2, -y / 2, -z / 2 };
	s = center + vect3{ -x / 2, +y / 2, -z / 2 };
	color = color_;

	if (!plane{ p, q, r }.contains(s))
		update_pols();
}

__host__ __device__ void parallelepiped::update_pols()
{
	vect3 u = q - p, v = r - p, w = s - p;

	pols[0] = polygon{ p, q, r, color };
	pols[1] = polygon{ r + u, q, r, color };

	pols[2] = polygon{ p, q, s, color };
	pols[3] = polygon{ q + w, q, s, color };

	pols[4] = polygon{ s + v, q + v + w, r, color };
	pols[5] = polygon{ s + u + v, q + v, r, color };

	pols[6] = polygon{ s, q + w, r + w, color };
	pols[7] = polygon{ s + v, q + w, r + w + u, color };

	pols[8] = polygon{ s + u, q, r + u, color };
	pols[9] = polygon{ s + u, q + v + w, r + u, color };

	pols[10] = polygon{ p, s, r, color };
	pols[11] = polygon{ p + v + w, s, r, color };

}

//__host__ __device__ vect3 parallelepiped::center()
//{
//	//return (vect3{ q + r + s - p } * 0.5);
//}


//__host__ __device__ parallelepiped rotate(parallelepiped* par, float aplha, float beta, float gamma)
//{
//	return *par;
//}

__host__ __device__ icosahedron::icosahedron(vect3 center_, float a_, vect3 color_)
{
	center = center_;
	color = color_;
	a = a_;
	n = center + vect3{0, 0, 0.951f * a};
	s = center - vect3{ 0, 0, 0.951f * a};
	update_pols();
}

__host__ __device__ void icosahedron::update_pols()
{
	float h = a*sinf(31.715 * (3.1416 / 180));
	float l = a*cosf(31.715 * (3.1416 / 180));
	vect3 p = n + (vect3{ 0, l, -h });
	vect3 q = s + (vect3{ 0, l, h});
	q = rotation(&q, 0, 0, 36 * (3.1416 / 180), center);
	float phi = 1.618f;

	pols[0] = polygon{ n, p, rotation(&p, 0, 0, 72 * (3.1416 / 180), center), {255, 0, 0} };
	pols[1] = polygon{ n, p, rotation(&p, 0, 0, -72 * (3.1416 / 180), center), {255, 255, 128} };
	pols[2] = polygon{ n, rotation(&p, 0, 0, 72 * (3.1416 / 180), center), rotation(&p, 0, 0, 144 * (3.1416 / 180), center), {255, 255, 0} };
	pols[3] = polygon{ n, rotation(&p, 0, 0, -72 * (3.1416 / 180), center), rotation(&p, 0, 0, -144 * (3.1416 / 180), center), {0, 255, 0} };
	pols[4] = polygon{ n, rotation(&p, 0, 0, -144 * (3.1416 / 180), center), rotation(&p, 0, 0, 144 * (3.1416 / 180), center), {0, 0, 255} };

	pols[5] = polygon{ s, q, rotation(&q, 0, 0, 72 * (3.1416 / 180), center), {255, 0, 255} };
	pols[6] = polygon{ s, q, rotation(&q, 0, 0, -72 * (3.1416 / 180), center), {0, 255, 255} };
	pols[7] = polygon{ s, rotation(&q, 0, 0, 72 * (3.1416 / 180), center), rotation(&q, 0, 0, 144 * (3.1416 / 180), center), {255, 128, 0} };
	pols[8] = polygon{ s, rotation(&q, 0, 0, -72 * (3.1416 / 180), center), rotation(&q, 0, 0, -144 * (3.1416 / 180), center), {255, 255, 255} };
	pols[9] = polygon{ s, rotation(&q, 0, 0, -144 * (3.1416 / 180), center), rotation(&q, 0, 0, 144 * (3.1416 / 180), center), {128, 0, 128} };

	pols[10] = polygon{ p, q, rotation(&p, 0, 0, 72 * (3.1416 / 180), center), {0, 128, 128} };
	pols[11] = polygon{ q, rotation(&p, 0, 0, 72 * (3.1416 / 180), center), rotation(&q, 0, 0, 72 * (3.1416 / 180), center), {64, 64, 255} };
	pols[12] = polygon{ rotation(&p, 0, 0, 72 * (3.1416 / 180), center), rotation(&q, 0, 0, 72 * (3.1416 / 180), center), rotation(&p, 0, 0, 144 * (3.1416 / 180), center), {255, 0, 128} };
	pols[13] = polygon{ rotation(&q, 0, 0, 72 * (3.1416 / 180), center), rotation(&p, 0, 0, 144 * (3.1416 / 180), center), rotation(&q, 0, 0, 144 * (3.1416 / 180), center), {0, 0, 128} };

	pols[14] = polygon{ rotation(&p, 0, 0, 144 * (3.1416 / 180), center), rotation(&q, 0, 0, 144 * (3.1416 / 180), center), rotation(&p, 0, 0, 216 * (3.1416 / 180), center), {128, 128, 128} };
	pols[15] = polygon{ rotation(&q, 0, 0, 144 * (3.1416 / 180), center), rotation(&p, 0, 0, 216 * (3.1416 / 180), center), rotation(&q, 0, 0, 216 * (3.1416 / 180), center), {192, 192, 0} };
	pols[16] = polygon{ rotation(&p, 0, 0, 216 * (3.1416 / 180), center), rotation(&q, 0, 0, 216 * (3.1416 / 180), center), rotation(&p, 0, 0, 288 * (3.1416 / 180), center), {192, 64, 64} };
	pols[17] = polygon{ rotation(&q, 0, 0, 216 * (3.1416 / 180), center), rotation(&p, 0, 0, 288 * (3.1416 / 180), center), rotation(&q, 0, 0, 288 * (3.1416 / 180), center), {255, 192, 64} };
	pols[18] = polygon{ rotation(&p, 0, 0, 288 * (3.1416 / 180), center), rotation(&q, 0, 0, 288 * (3.1416 / 180), center), p, {64, 192, 192} };
	pols[19] = polygon{ rotation(&q, 0, 0, 288 * (3.1416 / 180), center), p, q, {255, 64, 192} };

	/*vect3 p0 = center + vect3{ -1, phi, 0 };
	vect3 p1 = center + vect3{ 0, 1, phi };
	vect3 p2 = center + vect3{ 0, 1, phi };
	vect3 p3 = center + vect3{ 0, 1, phi };
	vect3 p4 = center + vect3{ 0, 1, phi };
	vect3 p5 = center + vect3{ 0, 1, phi };
	vect3 p6 = center + vect3{ 0, 1, phi };
	vect3 p7 = center + vect3{ 0, 1, phi };
	vect3 p8 = center + vect3{ 0, 1, phi };
	vect3 p9 = center + vect3{ 0, 1, phi };
	vect3 p10 = center + vect3{ 0, 1, phi };
	vect3 p11 = center + vect3{ 0, 1, phi };*/

	// (0, +-1, +-phi), (+-phi, 0, +-1) ? (+-1, +-phi, 0)
}
