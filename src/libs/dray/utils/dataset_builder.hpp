// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_DATSET_BUILDER_HPP
#define DRAY_DATSET_BUILDER_HPP

#include <dray/types.hpp>
#include <dray/vec.hpp>
#include <dray/data_model/elem_attr.hpp>

#include <conduit.hpp>

#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <numeric>



// Sequential builder to emit cells (disconnected from all other cells),
// with associated vertex coordinates, and vertex- or cell-centered field data.
// Then after finished building, can be converted to a blueprint dataset.

// Supports low-order.

namespace dray
{

  template <typename T, int32 ncomp>
  struct HexVData
  {
    Vec<T, ncomp> m_data[8];
  };

  template <typename T, int32 ncomp>
  struct HexEData
  {
    Vec<T, ncomp> m_data[1];
  };

  /*
  template <typename T, int32 ncomp>
  struct TetVData
  {
    Vec<T, ncomp> m_data[4];
  };

  template <typename T, int32 ncomp>
  struct TetEData
  {
    Vec<T, ncomp> m_data[1];
  };
  */


  /** HexRecord */
  class HexRecord
  {
    public:
      using VScalarT = HexVData<Float, 1>;
      using EScalarT = HexEData<Float, 1>;
      using VVectorT = HexVData<Float, 3>;
      using EVectorT = HexEData<Float, 3>;
      using CoordT = HexVData<Float, 3>;

    private:
      int32 m_birthtime;
      bool m_is_immortal;

      std::vector<CoordT> m_coord_data;
      std::vector<bool> m_coord_data_initd;
      std::map<std::string, int32> m_coord_idx;
      std::vector<std::string> m_coord_name;

      std::vector<VScalarT> m_scalar_vert_data;
      std::vector<EScalarT> m_scalar_elem_data;
      std::vector<VVectorT> m_vector_vert_data;
      std::vector<EVectorT> m_vector_elem_data;
      std::vector<bool> m_scalar_vert_data_initd;
      std::vector<bool> m_scalar_elem_data_initd;
      std::vector<bool> m_vector_vert_data_initd;
      std::vector<bool> m_vector_elem_data_initd;

      std::vector<std::string> m_scalar_vert_name;
      std::vector<std::string> m_scalar_elem_name;
      std::vector<std::string> m_vector_vert_name;
      std::vector<std::string> m_vector_elem_name;

      std::map<std::string, int32> m_scalar_vert_name2id;
      std::map<std::string, int32> m_scalar_elem_name2id;
      std::map<std::string, int32> m_vector_vert_name2id;
      std::map<std::string, int32> m_vector_elem_name2id;

    public:
      /** HexRecord() : These are maps from names to ids, e.g. as in DataSetBuilder.
       *                Keeps consistent ordering from input.
       *                Normally you shouldn't create your own HexRecord,
       *                but rather you should use DataSetBuilder::new_empty_hex_record().
       */
      HexRecord(const std::map<std::string, int32> &coord_name2id,
                const std::map<std::string, int32> &scalar_vert_name2id,
                const std::map<std::string, int32> &scalar_elem_name2id);

      HexRecord(const std::map<std::string, int32> &coord_name2id,
                const std::map<std::string, int32> &scalar_vert_name2id,
                const std::map<std::string, int32> &scalar_elem_name2id,
                const std::map<std::string, int32> &vector_vert_name2id,
                const std::map<std::string, int32> &vector_elem_name2id);

      /** birthtime() */
      int32 birthtime() const { return m_birthtime; }

      /** birthtime() */
      void birthtime(int32 birthtime) { m_birthtime = birthtime; }

      /** immortal() */
      bool immortal() const { return m_is_immortal; }

      /** immortal() */
      void immortal(bool immortal) { m_is_immortal = immortal; }

      /** is_initd_self() */
      bool is_initd_self() const;

      /** is_initd_extern() */
      bool is_initd_extern(const std::map<std::string, int32> &coord_name2id,
                           const std::map<std::string, int32> &scalar_vert_name2id,
                           const std::map<std::string, int32> &scalar_elem_name2id,
                           const std::map<std::string, int32> &vector_vert_name2id,
                           const std::map<std::string, int32> &vector_elem_name2id ) const;

      /** print_uninitd_coords */
      void print_uninitd_coords(bool println = true) const;

      /** print_uninitd_fields */
      void print_uninitd_fields(bool println = true) const;

      /** reset_all() */
      void reset_all();

      /** reuse_all() */
      void reuse_all();

      /** coord_data() */
      const CoordT & coord_data(const std::string &cname) const;

      /** coord_data() */
      void coord_data(const std::string &cname, const CoordT &coord_data);

      /** scalar_vert_data() */
      const VScalarT & scalar_vert_data(const std::string &fname) const;

      /** scalar_vert_data() */
      void scalar_vert_data(const std::string &fname, const VScalarT &vert_data);

      /** scalar_elem_data() */
      const EScalarT & scalar_elem_data(const std::string &fname) const;

      /** scalar_elem_data() */
      void scalar_elem_data(const std::string &fname, const EScalarT &elem_data);

      /** vector_vert_data() */
      const VVectorT & vector_vert_data(const std::string &fname) const;

      /** vector_vert_data() */
      void vector_vert_data(const std::string &fname, const VVectorT &vert_data);

      /** vector_elem_data() */
      const EVectorT & vector_elem_data(const std::string &fname) const;

      /** vector_elem_data() */
      void vector_elem_data(const std::string &fname, const EVectorT &elem_data);

  };


  /** DSBBuffer */
  struct DSBBuffer
  {
    DSBBuffer(int32 n_coordsets,
              int32 n_vert_scalar,
              int32 n_elem_scalar,
              int32 n_vert_vector,
              int32 n_elem_vector);

    void clear_records();

    int32 m_num_timesteps;
    int32 m_num_elems;
    int32 m_num_verts;

    std::vector<int32> m_timesteps;
    std::vector<bool> m_is_immortal;

    std::vector<std::vector<Vec<Float, 3>>> m_coord_data;

    std::vector<std::vector<Vec<Float, 1>>> m_scalar_vert_data;
    std::vector<std::vector<Vec<Float, 1>>> m_scalar_elem_data;
    std::vector<std::vector<Vec<Float, 3>>> m_vector_vert_data;
    std::vector<std::vector<Vec<Float, 3>>> m_vector_elem_data;
  };


  /** DataSetBuilder */
  class DataSetBuilder
  {
    public:
      enum ShapeMode { Hex, Tet, NUM_SHAPES };

      DataSetBuilder(ShapeMode shape_mode,
                     const std::vector<std::string> &coord_names,
                     const std::vector<std::string> &scalar_vert_names,
                     const std::vector<std::string> &scalar_elem_names,
                     const std::vector<std::string> &vector_vert_names,
                     const std::vector<std::string> &vector_elem_names );

      /** to_blueprint() : Copies cells tagged for cycle into conduit node, returns number of cells. */
      int32 to_blueprint(conduit::Node &bp_dataset, int32 cycle = 0) const;

      int32 num_timesteps() const { return m_central_buffer.m_num_timesteps; }

      ShapeMode shape_mode() const { return m_shape_mode; }
      void shape_mode_hex() { m_shape_mode = Hex; }
      void shape_mode_tet() { m_shape_mode = Tet; }

      int32 num_buffers() const { return m_inflow_buffers.size(); }
      void resize_num_buffers(int32 num_buffers);
      void flush_and_close_all_buffers();
      void clear_buffer(int32 buffer_id) { m_inflow_buffers[buffer_id].clear_records(); }

      HexRecord new_empty_hex_record() const;

      void add_hex_record(int32 buffer_id, HexRecord &record);
      void add_hex_record_direct(HexRecord &record);

      // Maps from coordset name to coordset index.
      const std::map<std::string, int32> &coord_name2id() const { return m_coord_idx; }

      // Maps from field name to field index in corresponding field category.
      const std::map<std::string, int32> &scalar_vert_name2id() const { return m_scalar_vert_name2id; }
      const std::map<std::string, int32> &scalar_elem_name2id() const { return m_scalar_elem_name2id; }
      const std::map<std::string, int32> &vector_vert_name2id() const { return m_vector_vert_name2id; }
      const std::map<std::string, int32> &vector_elem_name2id() const { return m_vector_elem_name2id; }

      // Coordset vector.
      const std::vector<Vec<Float, 3>> &coord_data(int32 idx) const;

      // Vectors of field data, by category.
      const std::vector<Vec<Float, 1>> &scalar_vert_data(int32 idx) const;
      const std::vector<Vec<Float, 1>> &scalar_elem_data(int32 idx) const;
      const std::vector<Vec<Float, 3>> &vector_vert_data(int32 idx) const;
      const std::vector<Vec<Float, 3>> &vector_elem_data(int32 idx) const;

    private:
      static int32 shape_npe[NUM_SHAPES];

      void add_hex_record(DSBBuffer &buffer, HexRecord &record);

      ShapeMode m_shape_mode;

      std::map<std::string, int32> m_coord_idx;
      std::map<std::string, int32> m_scalar_vert_name2id;
      std::map<std::string, int32> m_scalar_elem_name2id;
      std::map<std::string, int32> m_vector_vert_name2id;
      std::map<std::string, int32> m_vector_elem_name2id;

      DSBBuffer m_central_buffer;
      std::vector<DSBBuffer> m_inflow_buffers;
  };

}//namespace dray


#endif//DRAY_DATSET_BUILDER_HPP
