/**
 * @file    loop_mesh_builder.h
 *
 * @author  Lukáš Plevač <xpleva07@stud.fit.vutbr.cz>
 *
 * @brief   Parallel Marching Cubes implementation using OpenMP loops
 *
 * @date    16.11.2022
 **/

#ifndef LOOP_MESH_BUILDER_H
#define LOOP_MESH_BUILDER_H

#include <vector>
#include "base_mesh_builder.h"
#include <omp.h>

class LoopMeshBuilder : public BaseMeshBuilder
{
public:
    LoopMeshBuilder(unsigned gridEdgeSize);

protected:
    unsigned marchCubes(const ParametricScalarField &field);
    float evaluateFieldAt(const Vec3_t<float> &pos, const ParametricScalarField &field);
    void emitTriangle(const Triangle_t &triangle);
    const Triangle_t *getTrianglesArray() const { return triangles.data(); }
    
    std::vector<std::vector<Triangle_t>> mTriangles; ///< Temporary array of triangles
    std::vector<Triangle_t> triangles;
};

#endif // LOOP_MESH_BUILDER_H
