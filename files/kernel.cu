#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "kernel.cuh"
#include <cmath>
#include <vector>


const float __constant__ epsilon = 0.00001;

bool __host__ __device__ equ(float a, float b)
{
    return (a <= b + epsilon) && (a >= b - epsilon);
}
bool __host__ __device__ les_equ(float a, float b)
{
    return (a <= b + epsilon);
}
bool __host__ __device__ gre_equ(float a, float b)
{
    return (a >= b - epsilon);
}

int __host__ __device__ sign(float a)
{
    return (gre_equ(a, 0) ? 1 : (les_equ(a, 0) ? -1 : 0));
}

__host__ __device__ vect3::vect3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) 
{
    len = sqrt(x * x + y * y + z * z);
}
__host__ __device__ vect3::vect3(float val) : vect3(val, val, val) {}
__host__ __device__ vect3::vect3() : vect3(0, 0, 0) {}

__host__ __device__ vect3 vect3::operator + (vect3 v)
{
    vect3 vec{ x + v.x, y + v.y, z + v.z };
    return vec;
}
__host__ __device__ vect3 vect3::operator - (vect3 v)
{
    vect3 vec{ x - v.x, y - v.y, z - v.z };
    return vec;
}
__host__ __device__ vect3 vect3::operator * (float val)
{
    vect3 vec{ x * val, y * val, z * val };
    return vec;
}
__host__ __device__ vect3 vect3::operator / (float val)
{
    vect3 vec{ x / val, y / val, z / val };
    return vec;
}

__host__ __device__ float vect3::operator * (vect3 v)
{
    return x*v.x + y*v.y + z*v.z;
}
__host__ __device__ vect3 vect3::operator ^ (vect3 v)
{
    vect3 vec{ y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x };
    return vec;
}

__host__ __device__ bool vect3::operator == (vect3 v)
{
    return equ(x, v.x) && equ(y, v.y) && equ(z, v.z);
}

__host__ __device__ vect3 norm(vect3 vec) 
{
    if (equ(vec.len, 0))
        return vect3{0, 0, 0};

    return vec / vec.len;
}

__host__ __device__ line3::line3(vect3 p1_, vect3 p2_) : p1(p1_), p2(p2_) {}
__host__ __device__ line3::line3(vect3* p1_, vect3* p2_) : p1(*p1_), p2(*p2_) {}
__host__ __device__ line3::line3() : line3{ {0, 0, 0}, {1, 1, 1} } {}

__host__ __device__ bool line3::contains(vect3 point)
{
    auto dx = point.x - p1.x,
        dy = point.y - p1.y,
        dz = point.z - p1.z;

    vect3 v = p1 - p2;

    float ch_x = 0, ch_y = 0, ch_z = 0;
    bool a_x_zero = false, a_y_zero = false, a_z_zero = false;

    if (equ(v.x, 0))
        if (!equ(dx, 0))
            return false;
        else
            a_x_zero = true;
    else
        ch_x = dx / v.x;

    if (equ(v.y, 0))
        if (!equ(dy, 0))
            return false;
        else
            a_y_zero = true;
    else
        ch_y = dy / v.y;

    if (equ(v.z, 0))
        if (!equ(dz, 0))
            return false;
        else
            a_z_zero = true;
    else
        ch_z = dz / v.z;

    if (equ(ch_x, ch_y) && equ(ch_y, ch_z))
        return true;

    if (a_x_zero)
    {
        if (equ(ch_y, ch_x))
            return true;
        else if (equ(ch_y, ch_z))
            return true;
        else
            return false;
    }
    else if (a_y_zero)
    {
        if (equ(ch_y, ch_z))
            return true;
        else if (equ(ch_x, ch_z))
            return true;
        else
            return false;
    }
    else if (a_z_zero)
    {
        if (equ(ch_x, ch_y))
            return true;
        else
            return false;
    }
    return false;

}

__host__ __device__ bool line3::is_correct()
{
    return !(p1 == p2);
}


__host__ __device__ plane::plane(vect3 p1, vect3 p2, vect3 p3) 
{
    vect3 v1 = p2 - p1, v2 = p3 - p1;

    if ((line3{ p1, p2 }).contains(p3))
    {
        A = 0; 
        B = 0;
        C = 0;
        D = 0;
    }

    vect3 e1 = norm(v1);
    vect3 e2 = norm(v2);

    vect3 q1 = p1;
    vect3 q2 = p1 + e1;
    vect3 q3 = p1 + e2;

    A = (q2.y - q1.y) * (q3.z - q1.z) - (q3.y - q1.y) * (q2.z - q1.z);
    B = -(q2.x - q1.x) * (q3.z - q1.z) + (q3.x - q1.x) * (q2.z - q1.z);
    C = (q2.x - q1.x) * (q3.y - q1.y) - (q3.x - q1.x) * (q2.y - q1.y);
    D = -A * q1.x - B * q1.y - C * q1.z;
    normale = norm(vect3{ A, B, C });
}
__host__ __device__ plane::plane(float a, float b, float c, float d) : A(a), B(b), C(c), D(d), normale{ norm(vect3{A, B, C}) } {}
__host__ __device__ plane::plane() : plane{ 0, 0, 0, 0 } {}

__host__ __device__ bool plane::is_correct()
{
    return !(equ(A, 0) && equ(B, 0) && equ(C, 0) && equ(D, 0));
}

__host__ __device__ float plane::value(vect3 p)
{
    return A * p.x + B * p.y + C * p.z + D;
}
__host__ __device__ float plane::value(vect3 *p)
{
    return A * p->x + B * p->y + C * p->z + D;
}
__host__ __device__ bool plane::contains(vect3 p)
{
    auto ans = A * p.x + B * p.y + C * p.z + D;
    return equ(ans, 0);
}
__host__ __device__ bool plane::contains(vect3* p)
{
    auto ans = A * p->x + B * p->y + C * p->z + D;
    return equ(ans, 0);
}
__host__ __device__ bool plane::intersects(line3 l, vect3* p)
{
    if (equ(vect3{ A, B, C } * (l.p2 - l.p1), 0))
        return false;

    vect3 v = l.p2 - l.p1;
    float px = (l.p1).x, py = l.p1.y, pz = l.p1.z;
    float vx = v.x, vy = v.y, vz = v.z;
    auto t0 = -(A * px + B * py + C * pz + D);

    t0 /= A * vx + B * vy + C * vz;

    *p = vect3{ px + t0 * vx, py + t0 * vy, pz + t0 * vz };
    return true;

}
__host__ __device__ bool plane::posit_intersects(line3 l, vect3* p)
{
    if (equ(this->get_normale() * (l.p2 - l.p1), 0))
        return false;

    vect3 v = l.p2 - l.p1;
    float px = (l.p1).x, py = l.p1.y, pz = l.p1.z;
    float vx = v.x, vy = v.y, vz = v.z;
    float t0 = -(A * px + B * py + C * pz + D);

    t0 /= A * vx + B * vy + C * vz;
    if (t0 <= 0)
        return false;
    *p = vect3{ px + t0 * vx, py + t0 * vy, pz + t0 * vz };
    return true;

}
__host__ __device__ bool plane::intersects(line3* l, vect3* p)
{
    if (equ(vect3{ A, B, C } *(l->p2 - l->p1), 0))
        return false;

    vect3 v = l->p2 - l->p1;
    float px = (l->p1).x, py = l->p1.y, pz = l->p1.z;
    float vx = v.x, vy = v.y, vz = v.z;
    auto t0 = -(A * px + B * py + C * pz + D);

    t0 /= A * vx + B * vy + C * vz;

    *p = vect3{ px + t0 * vx, py + t0 * vy, pz + t0 * vz };
    return true;
}
__host__ __device__ bool plane::posit_intersects(line3* l, vect3* p)
{
    vect3 v = l->p2 - l->p1;
    if (equ(vect3{ A, B, C } * v, 0))
        return false;

    
    auto t0 = -(A * (l->p1).x + B * l->p1.y + C * l->p1.z + D);

    t0 /= A * v.x + B * v.y + C * v.z;
    if (t0 <= 0)
        return false;
    *p = vect3{ (l->p1).x + t0 * v.x, l->p1.y + t0 * v.y, l->p1.z + t0 * v.z };
    return true;
}
__host__ __device__ vect3 plane::get_normale()
{
    if (normale == vect3{ 0, 0, 0 })
        normale = norm(vect3{ A, B, C });
    return normale;
}

__host__ __device__ sphere::sphere(vect3 p_, float r_, vect3 col) : p(p_), r(r_), color(col) {}
__host__ __device__ sphere::sphere() : sphere{ {0, 0, 0}, 1, {255, 255, 255} } {}

__host__ __device__ bool sphere::contains(vect3 q)
{
    return equ((q - p) * (q - p) - r * r, 0);
}
__host__ __device__ bool sphere::intersects(line3 l, vect3* q)
{
    vect3 vec = l.p2 - l.p1;

    float a = vec*vec;
    float b = 2 * (l.p1 * vec - p * vec);
    float c = (l.p1 * l.p1) - r * r - 2 * (p * l.p1) + (p * p);

    if (b * b < 4 * a * c)
    {
        *q = vect3{ 0, 0, 0 };
        return false;
    }

    float d_sqrt = sqrt(b * b - 4 * a * c);
    float t1 = (-b + d_sqrt) / (2 * a), t2 = (-b - d_sqrt) / (2 * a);

    //vect3 a1 = { l.p1.x + t1 * vec.x, l.p1.y + t1 * vec[1], point[2] + t1 * vec[2] };
    //vect3 a2 = { point[0] + t2 * vec[0], point[1] + t2 * vec[1], point[2] + t2 * vec[2] };

    vect3 a1 = l.p1 + (vec * t1);
    vect3 a2 = l.p1 + (vec * t2);

    if ((a1 - l.p1).len < (a2 - l.p1).len)
    {
        *q = a1;
    }
    else
    {
        *q = a2;
    }
    return true;
}

__host__ __device__ polygon::polygon(vect3 p1_, vect3 p2_, vect3 p3_, vect3 col) : pl(plane{ p1_, p2_, p3_ }), p1(p1_), p2(p2_), p3(p3_), color(col) {}
__host__ __device__ polygon::polygon() : polygon{ {0, 0, 1}, {1, 0, 0}, {0, 1, 0} } {}

__host__ __device__ bool polygon::is_correct()
{
    return pl.is_correct();
}

__host__ __device__ bool polygon::contains(vect3 p)
{
    vect3 l1 = (p - p1) ^ (p1 - p2);
    vect3 l2 = (p - p2) ^ (p2 - p3);
    vect3 l3 = (p - p3) ^ (p3 - p1);

    float q1 = pl.value(p + l1);
    float q2 = pl.value(p + l2);
    float q3 = pl.value(p + l3);

    return (les_equ(q1, 0) && les_equ(q2, 0) && les_equ(q3, 0)) ||
           (gre_equ(q1, 0) && gre_equ(q2, 0) && gre_equ(q3, 0));
}
__host__ __device__ bool polygon::contains(vect3 *p)
{
    vect3 l1 = (*p - p1) ^ (p1 - p2);
    vect3 l2 = (*p - p2) ^ (p2 - p3);
    vect3 l3 = (*p - p3) ^ (p3 - p1);

    float q1 = pl.value(*p + l1);
    float q2 = pl.value(*p + l2);
    float q3 = pl.value(*p + l3);

    return (les_equ(q1, 0) && les_equ(q2, 0) && les_equ(q3, 0)) ||
           (gre_equ(q1, 0) && gre_equ(q2, 0) && gre_equ(q3, 0));
}
__host__ __device__ bool polygon::intersects(line3 l, vect3* p)
{
    if (!pl.intersects(l, p))
        return false;
    return contains(*p);
}
__host__ __device__ bool polygon::posit_intersects(line3 l, vect3* p)
{
    if (!pl.posit_intersects(l, p))
        return false;
    return contains(*p);
}
__host__ __device__ bool polygon::intersects(line3* l, vect3* p)
{
    if (!pl.intersects(l, p))
        return false;
    return contains(*p);
}
__host__ __device__ bool parallelipiped_check(polygon* pol, vect3* p)
{
    return (gre_equ(p->x, fmin(pol->p1.x, fmin(pol->p2.x, pol->p3.x))) && les_equ(p->x, fmax(pol->p1.x, fmax(pol->p2.x, pol->p3.x)))) &&
        (gre_equ(p->y, fmin(pol->p1.y, fmin(pol->p2.y, pol->p3.y))) && les_equ(p->y, fmax(pol->p1.y, fmax(pol->p2.y, pol->p3.y)))) &&
        (gre_equ(p->z, fmin(pol->p1.z, fmin(pol->p2.z, pol->p3.z))) && les_equ(p->z, fmax(pol->p1.z, fmax(pol->p2.z, pol->p3.z))));
}
__host__ __device__ bool polygon::posit_intersects(line3* l, vect3* p)
{
    if (!pl.posit_intersects(l, p))
        return false;

    if (!parallelipiped_check(this, p))
        return false;

    return contains(p);

}

__host__ __device__ inline float polygon::MT_inters(line3* l, vect3* p)
{
    vect3 dir = l->p2 - l->p1;
    vect3 e1 = p2 - p1;
    vect3 e2 = p3 - p1;
    vect3 ray_cross_e2 = dir ^ e2;

    float det = e1 * ray_cross_e2;
    if (equ(det, 0.0))
        return -1.f; 

    float inv_det = 1.0 / det;
    vect3 s = l->p1 - p1;
    float u = inv_det * (s * ray_cross_e2);

    if ((u < 0 && abs(u) > epsilon) || (u > 1 && abs(u - 1) > epsilon))
        return -1.f;

    vect3 s_cross_e1 = s ^ e1;
    float v = inv_det * (dir * s_cross_e1);

    if ((v < 0 && abs(v) > epsilon) || (u + v > 1 && abs(u + v - 1) > epsilon))
        return -1.f;

    float t = inv_det * (e2 * s_cross_e1);

    if (t > epsilon) 
    {
        *p = vect3{ l->p1 + dir * t };
        return t;
    }
    else 
        return -1.f;

}

__host__ __device__ vect3 polygon::normale()
{
    return pl.get_normale();
}

__host__ __device__ vect3 rotation(vect3* p, float OX, float OY, float OZ, vect3 center)
{
    float cosa = cos(OX), cosb = cos(OY), cosg = cos(OZ),
          sina = sin(OX), sinb = sin(OY), sing = sin(OZ);

    float a = p->x - center.x;
    float b = p->y - center.y;
    float c = p->z - center.z;

    float x = a * (cosb * cosg) - b * (sing * cosb) + c * (sinb);
    float y = a * (sina * sinb * cosg + sing * cosa) + b * (-sina * sinb * sing + cosa * cosg) - c * (sina * cosb);
    float z = a * (sina * sing - sinb * cosa * cosg) + b * (sina * cosg + sinb * sing * cosa) + c * (cosa * cosb);

    return vect3{ x + center.x, y + center.y, z + center.z };
}

__host__ __device__ polygon rotation(polygon* pol, float OX, float OY, float OZ, vect3 center)
{
    vect3 p = rotation(&pol->p1, OX, OY, OZ, center);
    vect3 q = rotation(&pol->p2, OX, OY, OZ, center);
    vect3 r = rotation(&pol->p3, OX, OY, OZ, center);
    return polygon{p, q, r, pol->color};

}

//__host__ __device__ vect3 rotation(vect3 p, vect3 ort, float a)
//{
//    float cosa = cos(a), sina = sin(a);
//
//    float x = p->x * (cosa + (1 - cosa) * (ort.x*ort.x)) - p->y * ((1 - cosa)*ort.x*ort.y - sina*ort.z) + p->z * ((1-cosa)*ort.x*ort.z + sina*ort.y);
//    float y = p->x * ((1 - cosa)*ort.y*ort.x + sina*ort.z) + p->y * (cosa + (1 - cosa)*ort.y*ort.y) - p->z * ((1 - cosa)*ort.y*ort.z-sina*ort.x);
//    float z = p->x * ((1 - cosa)*ort.z*ort.x - sina*ort.y) + p->y * ((1 - cosa)*ort.z*ort.y+sina*ort.x) + p->z * (cosa + (1 - cosa)*ort.z*ort.z);
//
//    return vect3{ x, y, z };
//}
//
//__host__ __device__ polygon rotation(polygon* pol, vect3 ort, float a)
//{
//    vect3 p = rotation(pol->p1, ort, a);
//    vect3 q = rotation(pol->p2, ort, a);
//    vect3 r = rotation(pol->p3, ort, a);
//    return polygon{ p, q, r, pol->color };
//}


__global__ void trace(polygon* pols, unsigned int pol_num, vect3* cam, vect3* O, vect3* x, vect3* y, vect3* lights,
                      unsigned int lights_num, std::uint8_t* disp, const unsigned int width, const unsigned int height)
{
    vect3 inters;
    float coef = 0;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    //unsigned int k = blockIdx.z * blockDim.z + threadIdx.z;
    float dist = 100000, new_dist = 100000;
    float h_sc = 1.f / height, w_sc = 1.f / width;
    vect3 candidate = 0;
    vect3 light_inters_candidate;
    line3 lighttocand;
    vect3 col;
    int pol_cand_num = -1;
    bool int_check = false, ch = false;

    if ((i >= height) || (j >= width))
        return;
    
    line3 ray{ *cam, *O + (*x * (float(j) * h_sc)) + (*y * (float(i) * w_sc)) };
    for (int k = 0; k < pol_num; k++)
    {
        new_dist = pols[k].MT_inters(&ray, &inters);

        //new_dist = 10;
        //int_check = true;
        if (les_equ(new_dist, dist) && new_dist > epsilon)
        {
            //new_dist = int_check * (inters - *cam).len + (!int_check) * dist;
            //new_dist = (inters - *cam).len;
            //if (les_equ(new_dist, dist))
            //{
                //dist = new_dist * (ch)+dist * (!ch);
                dist = new_dist;
                //candidate = inters * (ch)+candidate * (!ch);
                candidate = inters;
                pol_cand_num = k;
            //}
        }

    }
    float coef_temp = 0;
    unsigned int coord = (i * width + j) * 4;

    if (pol_cand_num != -1)
    {
        for (int l = 0; l < lights_num; l++)
        {
            lighttocand = { &lights[l], &candidate};
            pols[pol_cand_num].intersects(&lighttocand, &light_inters_candidate);
            if (light_inters_candidate == candidate)
            {
                auto t1 = ((lights[l] - candidate) * pols[pol_cand_num].normale());
                auto t2 = (lights[l] - candidate).len;
                coef_temp = abs(t1 / t2) * (sign(pols[pol_cand_num].pl.value(cam)) == sign(pols[pol_cand_num].pl.value(lights[l])));
                if (coef_temp > coef)
                    coef = coef_temp;
            }
        }
        
        //coef /= lights_num;
        col = pols[pol_cand_num].color;
        disp[coord] = int(coef * col.x);
        disp[coord + 1] = int(coef * col.y);
        disp[coord + 2] = int(coef * col.z);
    }
    else
    {
        disp[coord] = 0;
        disp[coord + 1] = 0;
        disp[coord + 2] = 0;
    }
}

Info gpu_init(Info inf)
{
    //std::uint8_t* disp_dev = 0;
    //polygon* pols_dev = 0;
    //vect3* cam_dev = 0;
    //vect3* O_dev = 0;
    //vect3* x_dev = 0;
    //vect3* y_dev = 0;
    //vect3* light_dev = 0;
    Info dev = {};
    //dev.disp = 0;
    //dev.pols = 0;
    //dev.cam = 0;
    //dev.O = 0;
    //dev.x = 0;
    //dev.y = 0;
    //dev.light = 0;
    dev.lights_num = inf.lights_num;
    dev.pol_num = inf.pol_num;
    dev.width = inf.width;
    dev.height = inf.height;
    cudaError_t err;
    unsigned int N = inf.width * inf.height * 4;

    err = cudaSetDevice(0);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
    }
  
    err = cudaMalloc((void**)&dev.disp, N * sizeof(std::uint8_t));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc display failed!");
    }

    err = cudaMalloc((void**)&dev.pols, dev.pol_num * sizeof(polygon));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc sphere failed!");
    }
    err = cudaMalloc((void**)&dev.cam, sizeof(vect3));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc cam failed!");
    }
    err = cudaMalloc((void**)&dev.O, sizeof(vect3));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc O failed!");
    }
    err = cudaMalloc((void**)&dev.x, sizeof(vect3));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc x failed!");
    }
    err = cudaMalloc((void**)&dev.y, sizeof(vect3));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc y failed!");
    }
    err = cudaMalloc((void**)&dev.light, dev.lights_num * sizeof(vect3));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc light failed!");
    }
    return dev;
}

void gpu_free(Info dev)
{
    cudaFree(dev.disp);
    cudaFree(dev.pols);
    cudaFree(dev.cam);
    cudaFree(dev.O);
    cudaFree(dev.x);
    cudaFree(dev.y);
    cudaFree(dev.light);
}

void p_ray_tracing(Info inf, Info dev)
{
    cudaError_t err;
    unsigned int N = inf.width * inf.height * 4;

    err = cudaSetDevice(0);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
    }
    err = cudaMemcpy(dev.disp, inf.disp, N * sizeof(std::uint8_t), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy disp failed!");
    }
    err = cudaMemcpy(dev.pols, inf.pols, inf.pol_num * sizeof(polygon), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy sphere failed!");
    }
    err = cudaMemcpy(dev.cam, inf.cam, sizeof(vect3), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) 
    {
        fprintf(stderr, "cudaMemcpy cam failed!");
    }
    err = cudaMemcpy(dev.O, inf.O, sizeof(vect3), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy O failed!");
    }
    err = cudaMemcpy(dev.x, inf.x, sizeof(vect3), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy x failed!");
    }
    err = cudaMemcpy(dev.y, inf.y, sizeof(vect3), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy y failed!");
    }
    err = cudaMemcpy(dev.light, inf.light, inf.lights_num * sizeof(vect3), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy light failed!");
    }
    //printf("-----\n%f %f %f\n%f %f %f\n%f %f %f-----\n", pols_dev[0].p1.x, pols_dev[0].p1.y, pols_dev[0].p1.z, pols_dev[0].p2.x, pols_dev[0].p2.y, pols_dev[0].p3.z, pols_dev[0].p3.x, pols_dev[0].p3.y, pols_dev[0].p3.z);

    const dim3 threadsPerBlock(16, 16);
    const dim3 numBlocks(ceil(inf.width / threadsPerBlock.x) + 1, ceil(inf.height / threadsPerBlock.y) + 1);

    // Prefetch the x and y arrays to the GPU
    //cudaMemPrefetchAsync(dev_a, size * size * sizeof(float), 0, 0);
    //cudaMemPrefetchAsync(dev_b, size * size * sizeof(float), 0, 0);

    trace <<<numBlocks, threadsPerBlock >>> (dev.pols, dev.pol_num, dev.cam, dev.O, dev.x, dev.y, dev.light, dev.lights_num, dev.disp, dev.width, dev.height);

    // Check for any errors launching the kernel
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "tracing launch failed: %s\n", cudaGetErrorString(err));
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", err);
    }

    // Copy output vector from GPU buffer to host memory.
    err = cudaMemcpy(inf.disp, dev.disp, N * sizeof(std::uint8_t), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy final failed!");
    }
}
