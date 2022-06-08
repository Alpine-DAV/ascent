// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "dray/data_model/iso_ops.hpp"

#include <iomanip>

namespace dray
{
namespace eops
{



  std::ostream & operator<<(std::ostream &out, const IsocutInfo &ici)
  {
    if (!ici.m_cut_type_flag)
      out << "NoCut";
    else if (ici.m_cut_type_flag & IsocutInfo::CutSimpleTri)
      out << "SimpleTri";
    else if (ici.m_cut_type_flag & IsocutInfo::CutSimpleQuad)
      out << "SimpleQuad";
    else
    {
      bool something = false;
      if (ici.m_cut_type_flag & IsocutInfo::IntNoFace)
      {
        out << (something ? "|" : "") << "IntNoFace";
        something = true;
      }
      if (ici.m_cut_type_flag & IsocutInfo::IntManyFace)
      {
        out << (something ? "|" : "") << "IntManyFace";
        something = true;
      }
      if (ici.m_cut_type_flag & IsocutInfo::FaceNoEdge)
      {
        out << (something ? "|" : "") << "FaceNoEdge";
        something = true;
      }
      if (ici.m_cut_type_flag & IsocutInfo::FaceManyEdge)
      {
        out << (something ? "|" : "") << "FaceManyEdge";
        something = true;
      }
      if (ici.m_cut_type_flag & IsocutInfo::EdgeManyPoint)
      {
        out << (something ? "|" : "") << "EdgeManyPoint";
        something = true;
      }
    }
    out << "\tF";
    for (int i = 0; i < 8; i++)
      if (ici.m_bad_faces_flag & (1u << i))
        out << " " << i;
    out << "\tE";
    for (int i = 0; i < 32; i++)
      if (ici.m_bad_edges_flag & (1u << i))
        out << " " << std::setw(2) << i;

    return out;
  }

}// eops
}//dray
