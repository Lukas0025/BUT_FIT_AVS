/**
 * @file    tree_mesh_builder.cpp
 *
 * @author  Lukáš Plevač <xpleva07@stud.fit.vutbr.cz>
 *
 * @brief   Parallel Marching Cubes implementation using OpenMP tasks + octree early elimination
 *
 * @date    16.11.2022
 **/

#include <iostream>
#include <math.h>
#include <limits>

#include "tree_mesh_builder.h"

#define sqrt3 1.73205080757
#define sqrt3Half 0.86602540378

TreeMeshBuilder::TreeMeshBuilder(unsigned gridEdgeSize)
    : BaseMeshBuilder(gridEdgeSize, "Octree")
{
    mTriangles.resize(omp_get_max_threads());
}

unsigned TreeMeshBuilder::marchCubes(const ParametricScalarField &field)
{
    // Suggested approach to tackle this problem is to add new method to
    // this class. This method will call itself to process the children.
    // It is also strongly suggested to first implement Octree as sequential
    // code and only when that works add OpenMP tasks to achieve parallelism

    unsigned n = 0;

    #pragma omp parallel shared(n)
    {
        #pragma omp master
        {
            #pragma omp task shared(n)
            n = proccessNode(0, 0, 0, mGridSize, mGridSize, mGridSize, field);
        }

        #pragma omp barrier
        #pragma omp critical
        triangles.insert(triangles.end(), mTriangles[omp_get_thread_num()].begin(), mTriangles[omp_get_thread_num()].end());
    }
    
    return n;
}

unsigned TreeMeshBuilder::proccessNode(float from_x, float from_y, float from_z, float to_x, float to_y, float to_z, const ParametricScalarField field) {

    //printf("Called (%f, %f, %f) (%f, %f, %f)\n", from_x, from_y, from_z, to_x, to_y, to_z);

    unsigned size = std::abs(to_x - from_x);
    unsigned shift = size / 2;


    if (size <= 5) { //process node

        unsigned totalTriangles = 0;
        unsigned cubesCount = size*size*size;

        for(size_t i = 0; i < cubesCount; ++i) {
            Vec3_t<float> cubeOffset = {
                from_x + i % size,
                from_y + (i / size) % size,
                from_z + i / (size * size)
            };

            totalTriangles += buildCube(cubeOffset, field);
        }

        return totalTriangles;
    }

    float halfSize = size / 2.0f;

    Vec3_t<float> cubeCenter3d = {
        (from_x + halfSize) * mGridResolution,
        (from_y + halfSize) * mGridResolution,
        (from_z + halfSize) * mGridResolution
    };

    if (evaluateFieldAt(cubeCenter3d, field) > (mIsoLevel + sqrt3Half * (size * mGridResolution))) { //not in object
        //printf("Droping (%f, %f, %f)(%f, %f, %f)\n", cubeCenter3d.x, cubeCenter3d.y, cubeCenter3d.z, (to_x + from_x) / 2.0, (to_y + from_y) / 2.0, (to_z + from_z) / 2.0);
        return 0;
    }

    unsigned res = 0;
    for (unsigned i = 0; i < 8; i++) {
        #pragma omp task shared(res)
        {
            #pragma omp atomic
            res += proccessNode(
                from_x + shift * (i % 2),
                from_y + shift * (i % 4 > 1),
                from_z + shift * (i > 3),

                to_x   - shift * !(i % 2),
                to_y   - shift * (i % 4 < 2),
                to_z   - shift * (i < 4),

                field
            );
        }
    }

    #pragma omp taskwait
    return res;
}

float TreeMeshBuilder::evaluateFieldAt(const Vec3_t<float> &pos, const ParametricScalarField &field)
{
// NOTE: This method is called from "buildCube(...)"!

    // 1. Store pointer to and number of 3D points in the field
    //    (to avoid "data()" and "size()" call in the loop).
    const Vec3_t<float> *pPoints = field.getPoints().data();
    const unsigned count = unsigned(field.getPoints().size());

    float value = std::numeric_limits<float>::max();

    // 2. Find minimum square distance from points "pos" to any point in the
    //    field.
    for(unsigned i = 0; i < count; ++i)
    {
        float distanceSquared  = (pos.x - pPoints[i].x) * (pos.x - pPoints[i].x);
        distanceSquared       += (pos.y - pPoints[i].y) * (pos.y - pPoints[i].y);
        distanceSquared       += (pos.z - pPoints[i].z) * (pos.z - pPoints[i].z);

        // Comparing squares instead of real distance to avoid unnecessary
        // "sqrt"s in the loop.
        value = std::min(value, distanceSquared);
    }

    // 3. Finally take square root of the minimal square distance to get the real distance
    return sqrt(value);
}

void TreeMeshBuilder::emitTriangle(const BaseMeshBuilder::Triangle_t &triangle)
{
    // NOTE: This method is called from "buildCube(...)"!

    // Store generated triangle into vector (array) of generated triangles.
    // The pointer to data in this array is return by "getTrianglesArray(...)" call
    // after "marchCubes(...)" call ends.
    mTriangles[omp_get_thread_num()].push_back(triangle);
}
