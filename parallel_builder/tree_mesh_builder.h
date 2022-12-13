/**
 * @file    tree_mesh_builder.h
 *
 * @author  FULL NAME <xlogin00@stud.fit.vutbr.cz>
 *
 * @brief   Parallel Marching Cubes implementation using OpenMP tasks + octree early elimination
 *
 * @date    DATE
 **/

#ifndef TREE_MESH_BUILDER_H
#define TREE_MESH_BUILDER_H

#include "base_mesh_builder.h"
#include <omp.h>

class TreeMeshBuilder : public BaseMeshBuilder
{
public:
    TreeMeshBuilder(unsigned gridEdgeSize);

protected:
    unsigned marchCubes(const ParametricScalarField &field);
    float evaluateFieldAt(const Vec3_t<float> &pos, const ParametricScalarField &field);
    void emitTriangle(const Triangle_t &triangle);
    unsigned proccessNode(Vec3_t<float> from, Vec3_t<float> to, const ParametricScalarField &field);
    const Triangle_t *getTrianglesArray() const { return triangles.data(); }
    
    std::vector<std::vector<Triangle_t>> mTriangles; ///< Temporary array of triangles
    std::vector<Triangle_t> triangles;
};

#endif // TREE_MESH_BUILDER_H
