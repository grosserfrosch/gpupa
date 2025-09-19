#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>
#include <iostream>
#include <cmath>
#include "shapes.cuh"
#include <chrono>

unsigned int width = 500, height = 500;
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
    sf::RenderWindow window(sf::VideoMode({ width, height }), "Boris", sf::Style::Titlebar);
    window.setFramerateLimit(30);
    float pi = 3.1416;
    vect3 q{ 1, 1, 0 };
    std::cout << q.x << ' ' << q.y << ' ' << q.z << '\n';
    q = rotation(&q, pi, 0, 0);
    std::cout << q.x << ' ' << q.y << ' ' << q.z << '\n';

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
    std::cout << x.x << ' ' << x.y << ' ' << x.z << '\n';
    std::cout << y.x << ' ' << y.y << ' ' << y.z << '\n';
    std::cout << O.x << ' ' << O.y << ' ' << O.z << '\n';
    std::cout << disp_plane.A << ' ' << disp_plane.B << ' ' << disp_plane.C << ' ' << disp_plane.D << '\n';
    std::cout << x.len << ' ' << y.len << '\n';
    sphere tit1{ {-2, 5, 0}, 2, {252, 180, 121} };
    sphere tit2{ {2, 5, 0}, 2, {252, 180, 121} };
    sphere nip1{ {2, 3.3, 0}, 0.5, {245, 71, 100} };
    sphere nip2{ {-2, 3.3, 0}, 0.5, {245, 71, 100} };

    //polygon pol1{ {-5, 5, 0}, {-2.5, 5, 5}, {0, 5 ,0}, {255, 0, 0} };
    /*polygon pol2{ {-2.5, 5, 5}, {0, 5, 0}, {2.5, 5 ,5}, {0, 0, 255} };
    polygon pol3{ {0, 5, 0}, {2.5, 5, 5}, {5, 5 ,0}, {0, 255, 0} };
    polygon pol4{ {-5, 5, 0}, {0, 5, -5}, {5, 5 ,0}, {255, 255, 255} };
    polygon pol5{ {0, 5, 0}, {2.5, 5, 5}, {5, 5 ,0}, {0, 255, 0} };
    polygon pol6{ {-5, 5, 0}, {0, 5, -5}, {5, 5 ,0}, {255, 255, 255} };
    polygon pol7{ {-5, 5, 0}, {-2.5, 5, 5}, {0, 5 ,0}, {255, 0, 0} };
    polygon pol8{ {-2.5, 5, 5}, {0, 5, 0}, {2.5, 5 ,5}, {0, 0, 255} };
    polygon pol9{ {0, 5, 0}, {2.5, 5, 5}, {5, 5 ,0}, {0, 255, 0} };
    polygon pol10{ {-5, 5, 0}, {0, 5, -5}, {5, 5 ,0}, {255, 255, 255} };
    polygon pol11{ {0, 5, 0}, {2.5, 5, 5}, {5, 5 ,0}, {0, 255, 0} };
    polygon pol12{ {-5, 5, 0}, {0, 5, -5}, {5, 5 ,0}, {255, 255, 255} };*/
    unsigned int n = 4;
    sphere* sphs = new sphere[n]{ tit1, tit2, nip1, nip2 };
    
    //polygon* pols = new polygon[k]{ pol2, pol1, pol3, pol4, pol5, pol6, pol7, pol8, pol9, pol10, pol11, pol12 };
    vect3 p;
    parallelepiped par{ {0, 5, 0}, 2, 2, 2, {255, 0, 0} };
    icosahedron ico{ {0, 5, 0}, 2, {255, 255, 0} };
    polygon* pols = ico.pols;
    unsigned int k = ico.edges;
    //std::cout << pol1.intersects(line3{ {0, 0, 0}, O + y*(1/2)}, &p) << std::endl;
    std::cout << p.x << ' ' << p.y << ' ' << p.z << '\n';
    //test(pols);
    //std::cout << pols[0].color.x << std::endl;
    std::vector<std::uint8_t> pixelBuffer(width * height * 4);
    std::uint8_t *buffer = new std::uint8_t[width * height * 4];
    sf::Texture texture{ sf::Vector2u{width, height} };

    sf::Sprite sprite{texture};
    //vect3 ort = norm(vect3{-2.5, });
    for (int i = 0; i < width * height * 4; i+= 4)
    {
        int j = i / 4;
        int y = j % width;
        int x = j / width;
        pixelBuffer[i + 3] = 255;

        pixelBuffer[i] = int(sqrt(x*x + y*y)) % 256;
        pixelBuffer[i + 1] = 0;
        pixelBuffer[i + 2] = 0;

        
        //std::cout << i << std::endl;
    }
    sf::ContextSettings settings = window.getSettings();
    unsigned int lights_num = 3;
    vect3* light_spots = new vect3[lights_num]{ vect3{0, 4, 5}, vect3{5, 0, 0}, vect3{-5, 5, 0} };
    float t = 0;
    while (window.isOpen())
    {
        while (const std::optional event = window.pollEvent())
        {
            if (event->is<sf::Event::Closed>())
                window.close();
        }
        clock_t begin = clock();

        //light_spot = { 0, 5 * sin(t)*sin(t), 5 * sin(t) * sin(t)};
        //light_spot = norm(light_spot);
        //ray_tracing(sph, cam, O, x, y, pixelBuffer, width, height);
        for (int i = 0; i < k; i++)
            pols[i] = rotation(&pols[i], pi / 300, pi / 400, pi / 100, ico.center);
        p_ray_tracing(pols, k, &cam, &O, &x, &y, light_spots, lights_num, buffer, width, height);
        //lin_trace(pols, k, &cam, &O, &x, &y, &light_spot, buffer, width, height);
        clock_t end = clock();
        //std::cout << float(end - begin) / CLOCKS_PER_SEC << std::endl;
        //std::cout << light_spot.x << ' ' << light_spot.y << ' ' << light_spot.z << '\n';
        window.clear();
        texture.update(buffer);
        sprite.setTexture(texture);
        window.draw(sprite);
        window.display();
    }

    delete[] sphs;
}