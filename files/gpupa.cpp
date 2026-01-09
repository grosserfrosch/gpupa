#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>
#include <iostream>
#include <cmath>
#include "shapes.cuh"
#include <chrono>

unsigned int width = 900, height = 900;
float disp_w = 1.f, disp_h = 1.f;
const float scale_w = width / disp_w, scale_h = height / disp_h;

void lin_trace(polygon* pols, unsigned int pol_num, vect3* cam, vect3* O, vect3* x, vect3* y, vect3* light, std::uint8_t* disp, unsigned int width, unsigned int height)
{
    vect3 inters;
    float coef = 0;
    //unsigned int k = blockIdx.z * blockDim.z + threadIdx.z;
    float dist = 100000, new_dist = 100000;
    float h_sc = 1.f / height, w_sc = 1.f / width;
    vect3 candidate = 0;
    vect3 light_inters_candidate;
    line3 lighttocand;
    vect3 col;
    polygon* pol_cand = 0;
    bool int_check = false, ch = false;
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            line3 ray{ *cam, *O + (*x * (float(j) * h_sc)) + (*y * (float(i) * w_sc)) };
            for (int k = 0; k < pol_num; k++)
            {
                int_check = pols[k].posit_intersects(&ray, &inters);
                if (int_check)
                {
                    //new_dist = int_check * (inters - *cam).len + (!int_check) * dist;
                    new_dist = (inters - *cam).len;
                    if (les_equ(new_dist, dist))
                    {
                        //dist = new_dist * (ch)+dist * (!ch);
                        dist = new_dist;
                        //candidate = inters * (ch)+candidate * (!ch);
                        candidate = inters;
                        /*pol_cand.p1 = pols[k].p1 * (ch)+pol_cand.p1 * (!ch);
                        pol_cand.p2 = pols[k].p2 * (ch)+pol_cand.p2 * (!ch);
                        pol_cand.p3 = pols[k].p3 * (ch)+pol_cand.p3 * (!ch);
                        pol_cand.color = pols[k].color * (ch)+vect3{255, 255, 255}*(!ch);*/
                        pol_cand = &(pols[k]);
                    }
                }

            }

            unsigned int coord = (i * width + j) * 4;
            if (pol_cand != 0)
            {
                lighttocand = { light, &candidate };
                pol_cand->intersects(&lighttocand, &light_inters_candidate);
                if (light_inters_candidate == candidate)
                {
                    auto t1 = ((*light - candidate) * pol_cand->normale());
                    auto t2 = (*light - candidate).len;
                    coef = abs(t1 / t2);
                }
                col = pol_cand->color;
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
    }
}


int main()
{
    sf::RenderWindow window(sf::VideoMode({ width, height }), "Boris");
    window.setFramerateLimit(120);
    float pi = 3.1416;

    vect3 cam{ 0, 0, 0 };
    float dist = 0.5;
    vect3 lov{ vect3{ 0, dist, 0 } };
    vect3 disp_center{ cam + lov };
    plane disp_plane{ lov.x, lov.y, lov.z, -(lov.x * disp_center.x + lov.y * disp_center.y + lov.z * disp_center.z) };
    float dir{ -disp_plane.value(cam) };
    vect3 O{ disp_center + vect3{ -disp_w / 2, 0, disp_h / 2}};
    vect3 W{ disp_center + vect3{ disp_w / 2, 0, disp_h / 2 } };
    vect3 H{ disp_center + vect3{ -disp_w / 2, 0, -disp_h / 2}};
    vect3 x{ W - O }, y{ H - O };
    //now OH and OW -- basis vectors

    unsigned int n = 4;
    
    vect3 p;
    parallelepiped par{ {0, 5, 0}, 2, 2, 2, {255, 0, 0} };
    icosahedron ico{ {0, 5, 0}, 2, {255, 0, 0} };
    polygon* pols = ico.pols;
    unsigned int k = ico.edges;
    std::cout << p.x << ' ' << p.y << ' ' << p.z << '\n';

    std::vector<std::uint8_t> pixelBuffer(width * height * 4);
    std::uint8_t *buffer = new std::uint8_t[width * height * 4];
    
    //sf::Texture texture{ sf::Vector2u{width, height} };
    sf::Texture texture;
    texture.create(width, height);

    sf::Sprite sprite{texture};
    //vect3 ort = norm(vect3{-2.5, });
    
    sf::ContextSettings settings = window.getSettings();
    unsigned int lights_num = 3;
    vect3* light_spots = new vect3[lights_num]{ vect3{0, 4, 5}, vect3{5, 0, 0}, vect3{-5, 5, 0} };
    float frame_time = 0;
    clock_t start_frame = 0, end_frame = 0;
    Info inf = {};
    inf.lights_num = lights_num;
    inf.cam = &cam;
    inf.disp = buffer;
    inf.height = height;
    inf.width = width;
    inf.light = light_spots;
    inf.O = &O;
    inf.x = &x;
    inf.y = &y;
    inf.pols = pols;
    inf.pol_num = k;
    Info dev = gpu_init(inf);
    for (int i = 3; i < width * height * 4; i += 4)
        buffer[i] = 255;
    while (window.isOpen())
    {
        start_frame = clock();
        sf::Event event;
        while (window.pollEvent(event))
        {
            if (event.type == sf::Event::Closed)
                window.close();
        }
        clock_t begin = clock();
        
        
        //light_spot = norm(light_spot);
        //ray_tracing(sph, cam, O, x, y, pixelBuffer, width, height);
        inf.pols = pols;
        p_ray_tracing(inf, dev);
        
        //p_ray_tracing(pols, k, &cam, &O, &x, &y, light_spots, lights_num, buffer, width, height);
        //lin_trace(pols, k, &cam, &O, &x, &y, &light_spot, buffer, width, height);
        clock_t end = clock();
        //std::cout << float(end - begin) / CLOCKS_PER_SEC << std::endl;
        //std::cout << light_spot.x << ' ' << light_spot.y << ' ' << light_spot.z << '\n';
        //window.clear();
        texture.update(buffer);
        sprite.setTexture(texture);
        window.draw(sprite);
        window.display();
        end_frame = clock();
        frame_time = float(end_frame - start_frame) / CLOCKS_PER_SEC;
        std::cout << "fps: " << (1 / frame_time) << '\n';
        for (int i = 0; i < k; i++)
            pols[i] = rotation(&pols[i], frame_time * pi / 5, frame_time * pi / 7, frame_time * pi / 25, ico.center);
        
    }
    gpu_free(dev);
    //delete[] sphs;
}
