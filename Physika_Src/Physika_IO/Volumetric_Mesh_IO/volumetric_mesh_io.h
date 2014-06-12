/*
 * @file volumetric_mesh_io.h 
 * @brief volumetric mesh loader/saver, load/save volumetric mesh from/to file.
 * @author Fei Zhu
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013 Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#ifndef PHYSIKA_IO_VOLUMETRIC_MESH_IO_VOLUMETRIC_MESH_IO_H_
#define PHYSIKA_IO_VOLUMETRIC_MESH_IO_VOLUMETRIC_MESH_IO_H_

#include <string>

namespace Physika{

template <typename Scalar,int Dim> class VolumetricMesh;

template <typename Scalar, int Dim>
class VolumetricMeshIO
{
public:
    enum SaveOption{
        SINGLE_FILE,  //save volumetric mesh to a single .smesh file
        NODE_AND_ELE_FILES  //save nodes and elements to separate files, included by .smesh file
    };
    VolumetricMeshIO(){}
    ~VolumetricMeshIO(){}
    //load volumetric mesh from given file, return NULL if fails
    //the memory of volumetric mesh needs to be released by the caller
    static VolumetricMesh<Scalar,Dim>* load(const std::string &filename);
    //save volumetric mesh to file, return true if succeed, otherwise return false
    static bool save(const std::string &filename, const VolumetricMesh<Scalar,Dim> *volumetric_mesh, SaveOption option = SINGLE_FILE);
protected:
    static bool saveToSingleFile(const std::string &filename, const VolumetricMesh<Scalar,Dim> *volumetric_mesh);
    static bool saveToSeparateFiles(const std::string &filename, const VolumetricMesh<Scalar,Dim> *volumetric_mesh);
};

}  //end of namespace Physika

#endif //PHYSIKA_IO_VOLUMETRIC_MESH_IO_VOLUMETRIC_MESH_IO_H_
