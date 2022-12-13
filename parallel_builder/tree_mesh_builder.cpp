/**
 * @file    tree_mesh_builder.cpp
 *
 * @author  FULL NAME <xlogin00@stud.fit.vutbr.cz>
 *
 * @brief   Parallel Marching Cubes implementation using OpenMP tasks + octree early elimination
 *
 * @date    DATE
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
    omp_set_num_threads(4);
    mTriangles.resize(4);
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
            #pragma omp task firstprivate(mGridSize, field) shared(n)
            n = proccessNode((0, 0, 0), (mGridSize, mGridSize, mGridSize), field);
        }
    }

    triangles.insert(triangles.end(), mTriangles[omp_get_thread_num()].begin(), mTriangles[omp_get_thread_num()].end());
    
    return n;
}

unsigned TreeMeshBuilder::proccessNode(Vec3_t<float> from, Vec3_t<float> to, const ParametricScalarField &field) {

    Vec3_t<float> cubeCenter3d = (
        (from.x + to.x) / 2,
        (from.y + to.y) / 2,
        (from.z + to.z) / 2
    );

    float size = to.x - from.x;

    if (size < 1) { //process node
        return buildCube(cubeCenter3d, field);
    }

    /*if (evaluateFieldAt(cubeCenter3d, field) <= (field.getIsoLevel() + sqrt3Half * size)) { //not in object
        return 0;
    }*/

    unsigned shift = size / 2;
    unsigned res[8];

    for (unsigned i = 0; i < 8; i++) {
        #pragma omp task firstprivate(from, to, i, shift, field) shared(res)
        res[i] = proccessNode(
            (from.x + shift * (i % 2 == 0), from.y + shift * (i % 4 == 0), from.z + shift * (i % 8 == 0)),
            (to.x   - shift * (i % 2 != 0), to.y   - shift * (i % 4 != 0), to.z   - shift * (i % 8 != 0)),
            field
        );
    }

    #pragma omp taskwait
    return res[0] + res[1] + res[2] + res[3] + res[4] + res[5] + res[6] + res[7];
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
