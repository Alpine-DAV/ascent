// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/utils/dataset_builder.hpp>


namespace dray
{
  namespace optimized
  {
    template <typename T>
    inline T min(T a, T b)
    {
      return (a <= b ? a : b);
    }

    template <typename T>
    inline T max(T a, T b)
    {
      return (a >= b ? a : b);
    }
  }


  //
  // HexRecord definitions
  //

  /** HexRecord() : Keeps consistent ordering from input. */
  HexRecord::HexRecord(const std::map<std::string, int32> &coord_name2id,
                       const std::map<std::string, int32> &scalar_vert_name2id,
                       const std::map<std::string, int32> &scalar_elem_name2id)
    : HexRecord(coord_name2id, scalar_vert_name2id, scalar_elem_name2id, {}, {})
  {
  }

  HexRecord::HexRecord(const std::map<std::string, int32> &coord_name2id,
                       const std::map<std::string, int32> &scalar_vert_name2id,
                       const std::map<std::string, int32> &scalar_elem_name2id,
                       const std::map<std::string, int32> &vector_vert_name2id,
                       const std::map<std::string, int32> &vector_elem_name2id )
    : m_birthtime(0),
      m_is_immortal(false),

      m_coord_idx(coord_name2id),
      m_coord_data_initd(coord_name2id.size(), false),
      m_coord_data(coord_name2id.size()),
      m_coord_name(coord_name2id.size()),

      m_scalar_vert_name2id(scalar_vert_name2id),
      m_scalar_vert_data_initd(scalar_vert_name2id.size(), false),
      m_scalar_vert_data(scalar_vert_name2id.size()),
      m_scalar_vert_name(scalar_vert_name2id.size()),
      m_scalar_elem_name2id(scalar_elem_name2id),
      m_scalar_elem_data_initd(scalar_elem_name2id.size(), false),
      m_scalar_elem_data(scalar_elem_name2id.size()),
      m_scalar_elem_name(scalar_elem_name2id.size()),

      m_vector_vert_name2id(vector_vert_name2id),
      m_vector_vert_data_initd(vector_vert_name2id.size(), false),
      m_vector_vert_data(vector_vert_name2id.size()),
      m_vector_vert_name(vector_vert_name2id.size()),
      m_vector_elem_name2id(vector_elem_name2id),
      m_vector_elem_data_initd(vector_elem_name2id.size(), false),
      m_vector_elem_data(vector_elem_name2id.size()),
      m_vector_elem_name(vector_elem_name2id.size())

  {
    for (const auto &name_idx : coord_name2id)
      m_coord_name.at(name_idx.second) = name_idx.first;
    for (const auto &name_idx : scalar_vert_name2id)
      m_scalar_vert_name.at(name_idx.second) = name_idx.first;
    for (const auto &name_idx : scalar_elem_name2id)
      m_scalar_elem_name.at(name_idx.second) = name_idx.first;
    for (const auto &name_idx : vector_vert_name2id)
      m_vector_vert_name.at(name_idx.second) = name_idx.first;
    for (const auto &name_idx : vector_elem_name2id)
      m_vector_elem_name.at(name_idx.second) = name_idx.first;
  }

  /** is_initd_self() */
  bool HexRecord::is_initd_self() const
  {
    return (   *std::min_element(m_coord_data_initd.begin(),   m_coord_data_initd.end())
            && *std::min_element(m_scalar_vert_data_initd.begin(), m_scalar_vert_data_initd.end())
            && *std::min_element(m_scalar_elem_data_initd.begin(), m_scalar_elem_data_initd.end())
            && *std::min_element(m_vector_vert_data_initd.begin(), m_vector_vert_data_initd.end())
            && *std::min_element(m_vector_elem_data_initd.begin(), m_vector_elem_data_initd.end())
            );
  }

  /** is_initd_extern() */
  bool HexRecord::is_initd_extern(const std::map<std::string, int32> &coord_name2id,
                                  const std::map<std::string, int32> &scalar_vert_name2id,
                                  const std::map<std::string, int32> &scalar_elem_name2id,
                                  const std::map<std::string, int32> &vector_vert_name2id,
                                  const std::map<std::string, int32> &vector_elem_name2id ) const
  {
    bool initd = true;
    for (const auto & name_idx : coord_name2id)
      initd &= m_coord_data_initd[m_coord_idx.at(name_idx.first)];
    for (const auto & name_idx : scalar_vert_name2id)
      initd &= m_scalar_vert_data_initd[m_scalar_vert_name2id.at(name_idx.first)];
    for (const auto & name_idx : scalar_elem_name2id)
      initd &= m_scalar_elem_data_initd[m_scalar_elem_name2id.at(name_idx.first)];
    for (const auto & name_idx : vector_vert_name2id)
      initd &= m_vector_vert_data_initd[m_vector_vert_name2id.at(name_idx.first)];
    for (const auto & name_idx : vector_elem_name2id)
      initd &= m_vector_elem_data_initd[m_vector_elem_name2id.at(name_idx.first)];
    return initd;
  }

  /** print_uninitd_coords */
  void HexRecord::print_uninitd_coords(bool println) const
  {
    const char *RED = "\u001b[31m";
    const char *NRM = "\u001b[0m";

    const char end = (println ? '\n' : ' ');
    for (int32 idx = 0; idx < m_coord_name.size(); ++idx)
      if (!m_coord_data_initd[idx])
        printf("%sCoordinate data '%s' is uninitialized.%c%s",
            RED, m_coord_name[idx].c_str(), end, NRM);
  }

  /** print_uninitd_fields */
  void HexRecord::print_uninitd_fields(bool println) const
  {
    const char *RED = "\u001b[31m";
    const char *NRM = "\u001b[0m";

    const char end = (println ? '\n' : ' ');
    for (int32 idx = 0; idx < m_scalar_vert_name.size(); ++idx)
      if (!m_scalar_vert_data_initd[idx])
        printf("%sField data (vert) '%s' is uninitialized.%c%s",
            RED, m_scalar_vert_name[idx].c_str(), end, NRM);
    for (int32 idx = 0; idx < m_scalar_elem_name.size(); ++idx)
      if (!m_scalar_elem_data_initd[idx])
        printf("%sField data (elem) '%s' is uninitialized.%c%s",
            RED, m_scalar_elem_name[idx].c_str(), end, NRM);
    for (int32 idx = 0; idx < m_vector_vert_name.size(); ++idx)
      if (!m_vector_vert_data_initd[idx])
        printf("%sField data (vert) '%s' is uninitialized.%c%s",
            RED, m_vector_vert_name[idx].c_str(), end, NRM);
    for (int32 idx = 0; idx < m_vector_elem_name.size(); ++idx)
      if (!m_vector_elem_data_initd[idx])
        printf("%sField data (elem) '%s' is uninitialized.%c%s",
            RED, m_vector_elem_name[idx].c_str(), end, NRM);
  }

  /** reset_all() */
  void HexRecord::reset_all()
  {
    m_coord_data_initd.clear();
    m_coord_data_initd.resize(m_coord_idx.size(), false);
    m_scalar_vert_data_initd.clear();
    m_scalar_vert_data_initd.resize(m_scalar_vert_name2id.size(), false);
    m_scalar_elem_data_initd.clear();
    m_scalar_elem_data_initd.resize(m_scalar_elem_name2id.size(), false);
    m_vector_vert_data_initd.clear();
    m_vector_vert_data_initd.resize(m_vector_vert_name2id.size(), false);
    m_vector_elem_data_initd.clear();
    m_vector_elem_data_initd.resize(m_vector_elem_name2id.size(), false);
  }

  /** reuse_all() */
  void HexRecord::reuse_all()
  {
    m_coord_data_initd.clear();
    m_coord_data_initd.resize(m_coord_idx.size(), true);
    m_scalar_vert_data_initd.clear();
    m_scalar_vert_data_initd.resize(m_scalar_vert_name2id.size(), true);
    m_scalar_elem_data_initd.clear();
    m_scalar_elem_data_initd.resize(m_scalar_elem_name2id.size(), true);
    m_vector_vert_data_initd.clear();
    m_vector_vert_data_initd.resize(m_vector_vert_name2id.size(), true);
    m_vector_elem_data_initd.clear();
    m_vector_elem_data_initd.resize(m_vector_elem_name2id.size(), true);
  }

  /** coord_data() */
  const HexRecord::CoordT & HexRecord::coord_data(const std::string &cname) const
  {
    return m_coord_data[m_coord_idx.at(cname)];
  }

  /** coord_data() */
  void HexRecord::coord_data(const std::string &cname, const CoordT &coord_data)
  {
    const int32 idx = m_coord_idx.at(cname);
    m_coord_data[idx] = coord_data;
    m_coord_data_initd[idx] = true;
  }

  /** scalar_vert_data() */
  const HexRecord::VScalarT & HexRecord::scalar_vert_data(const std::string &fname) const
  {
    return m_scalar_vert_data[m_scalar_vert_name2id.at(fname)];
  }

  /** scalar_vert_data() */
  void HexRecord::scalar_vert_data(const std::string &fname, const VScalarT &vert_data)
  {
    const int32 idx = m_scalar_vert_name2id.at(fname);
    m_scalar_vert_data[idx] = vert_data;
    m_scalar_vert_data_initd[idx] = true;
  }

  /** scalar_elem_data() */
  const HexRecord::EScalarT & HexRecord::scalar_elem_data(const std::string &fname) const
  {
    return m_scalar_elem_data[m_scalar_elem_name2id.at(fname)];
  }

  /** scalar_elem_data() */
  void HexRecord::scalar_elem_data(const std::string &fname, const EScalarT &elem_data)
  {
    const int32 idx = m_scalar_elem_name2id.at(fname);
    m_scalar_elem_data[idx] = elem_data;
    m_scalar_elem_data_initd[idx] = true;
  }

  /** vector_vert_data() */
  const HexRecord::VVectorT & HexRecord::vector_vert_data(const std::string &fname) const
  {
    return m_vector_vert_data[m_vector_vert_name2id.at(fname)];
  }

  /** vector_vert_data() */
  void HexRecord::vector_vert_data(const std::string &fname, const VVectorT &vert_data)
  {
    const int32 idx = m_vector_vert_name2id.at(fname);
    m_vector_vert_data[idx] = vert_data;
    m_vector_vert_data_initd[idx] = true;
  }

  /** vector_elem_data() */
  const HexRecord::EVectorT & HexRecord::vector_elem_data(const std::string &fname) const
  {
    return m_vector_elem_data[m_vector_elem_name2id.at(fname)];
  }

  /** vector_elem_data() */
  void HexRecord::vector_elem_data(const std::string &fname, const EVectorT &elem_data)
  {
    const int32 idx = m_vector_elem_name2id.at(fname);
    m_vector_elem_data[idx] = elem_data;
    m_vector_elem_data_initd[idx] = true;
  }




  //
  // DSBBuffer definitions.
  //

  /** DSBBuffer() */
  DSBBuffer::DSBBuffer(int32 n_coordsets,
                       int32 n_vert_scalar,
                       int32 n_elem_scalar,
                       int32 n_vert_vector,
                       int32 n_elem_vector)
    : m_num_timesteps(1),
      m_num_elems(0),
      m_num_verts(0),
      m_coord_data(n_coordsets),
      m_scalar_vert_data(n_vert_scalar),
      m_scalar_elem_data(n_elem_scalar),
      m_vector_vert_data(n_vert_vector),
      m_vector_elem_data(n_elem_vector)
  { }

  /** clear_records() */
  void DSBBuffer::clear_records()
  {
    m_num_timesteps = 1;
    m_num_elems = 0;
    m_num_verts = 0;

    m_timesteps.clear();
    m_is_immortal.clear();

    for (auto &field : m_coord_data)
      field.clear();
    for (auto &field : m_scalar_vert_data)
      field.clear();
    for (auto &field : m_scalar_elem_data)
      field.clear();
    for (auto &field : m_vector_vert_data)
      field.clear();
    for (auto &field : m_vector_elem_data)
      field.clear();
  }



  //
  // DataSetBuilder definitions
  //

  /** shape_npe[] */
  int32 DataSetBuilder::shape_npe[DataSetBuilder::NUM_SHAPES] = {8, 4};

  /** DataSetBuilder() */
  DataSetBuilder::DataSetBuilder(ShapeMode shape_mode,
                                 const std::vector<std::string> &coord_names,
                                 const std::vector<std::string> &scalar_vert_names,
                                 const std::vector<std::string> &scalar_elem_names,
                                 const std::vector<std::string> &vector_vert_names,
                                 const std::vector<std::string> &vector_elem_names )
    : m_shape_mode(shape_mode),
      m_central_buffer(coord_names.size(),
                       scalar_vert_names.size(),
                       scalar_elem_names.size(),
                       vector_vert_names.size(),
                       vector_elem_names.size())
  {
    int32 idx;

    idx = 0;
    for (const std::string &cname : coord_names)
      m_coord_idx[cname] = idx++;

    idx = 0;
    for (const std::string &fname : scalar_vert_names)
      m_scalar_vert_name2id[fname] = idx++;

    idx = 0;
    for (const std::string &fname : scalar_elem_names)
      m_scalar_elem_name2id[fname] = idx++;

    idx = 0;
    for (const std::string &fname : vector_vert_names)
      m_vector_vert_name2id[fname] = idx++;

    idx = 0;
    for (const std::string &fname : vector_elem_names)
      m_vector_elem_name2id[fname] = idx++;
  }

  /** resize_num_buffers() */
  void DataSetBuilder::resize_num_buffers(int32 num_buffers)
  {
    m_inflow_buffers.clear();
    if (num_buffers > 0)
    {
      const int32 n_coordsets = m_coord_idx.size();
      const int32 n_vert_scalar = m_scalar_vert_name2id.size();
      const int32 n_elem_scalar = m_scalar_elem_name2id.size();
      const int32 n_vert_vector = m_vector_vert_name2id.size();
      const int32 n_elem_vector = m_vector_elem_name2id.size();

      m_inflow_buffers.emplace_back(n_coordsets,
                                    n_vert_scalar,
                                    n_elem_scalar,
                                    n_vert_vector,
                                    n_elem_vector);
      m_inflow_buffers.resize(num_buffers, m_inflow_buffers[0]);
    }
  }

  /** new_empty_hex_record() */
  HexRecord DataSetBuilder::new_empty_hex_record() const
  {
    if (m_shape_mode != Hex)
      throw std::logic_error("Cannot call new_empty_hex_record() on a non-Hex DataSetBuilder.");
    return HexRecord(m_coord_idx,
                     m_scalar_vert_name2id,
                     m_scalar_elem_name2id,
                     m_vector_vert_name2id,
                     m_vector_elem_name2id);
  }

  /** add_hex_record() */
  void DataSetBuilder::add_hex_record(int32 buffer_id, HexRecord &record)
  {
    add_hex_record(m_inflow_buffers[buffer_id], record);
  }

  /** add_hex_record() */
  void DataSetBuilder::add_hex_record_direct(HexRecord &record)
  {
    add_hex_record(m_central_buffer, record);
  }

  /** add_hex_record() : Copies all registered data fields, then flags them as uninitialized. */
  void DataSetBuilder::add_hex_record(DSBBuffer &buffer, HexRecord &record)
  {
    using VScalarT = HexRecord::VScalarT;
    using EScalarT = HexRecord::EScalarT;
    using VVectorT = HexRecord::VVectorT;
    using EVectorT = HexRecord::EVectorT;
    using CoordT   = HexRecord::CoordT;

    if (m_shape_mode != Hex)
      throw std::logic_error("Cannot call add_hex_record() on a non-Hex DataSetBuilder.");

    if (!record.is_initd_extern(m_coord_idx,
                                m_scalar_vert_name2id,
                                m_scalar_elem_name2id,
                                m_vector_vert_name2id,
                                m_vector_elem_name2id))
    {
      record.print_uninitd_coords();
      record.print_uninitd_fields();
      throw std::logic_error("Attempt to add to DataSetBuilder, but record is missing coords/fields.");
    }

    constexpr int32 verts_per_elem = 8;
    const int32 vtk_2_lex[8] = {0, 1, 3, 2,  4, 5, 7, 6};

    buffer.m_timesteps.push_back(record.birthtime());
    buffer.m_is_immortal.push_back(record.immortal());

    buffer.m_num_timesteps = optimized::max(buffer.m_num_timesteps, record.birthtime() + 1);

    buffer.m_num_elems++;
    buffer.m_num_verts += verts_per_elem;

    for (const auto &name_idx : m_coord_idx)
    {
      const std::string &cname = name_idx.first;
      const int32 cidx = name_idx.second;
      const CoordT &fdata = record.coord_data(cname);
      for (int32 j = 0; j < verts_per_elem; ++j)
        buffer.m_coord_data[cidx].push_back(fdata.m_data[vtk_2_lex[j]]);
    }

    for (const auto &name_idx : m_scalar_vert_name2id)
    {
      const std::string &fname = name_idx.first;
      const int32 fidx = name_idx.second;
      const VScalarT &fdata = record.scalar_vert_data(fname);
      for (int32 j = 0; j < verts_per_elem; ++j)
        buffer.m_scalar_vert_data[fidx].push_back(fdata.m_data[vtk_2_lex[j]]);
    }

    for (const auto &name_idx : m_scalar_elem_name2id)
    {
      const std::string &fname = name_idx.first;
      const int32 fidx = name_idx.second;
      const EScalarT &fdata = record.scalar_elem_data(fname);
      buffer.m_scalar_elem_data[fidx].push_back(fdata.m_data[0]);
    }

    for (const auto &name_idx : m_vector_vert_name2id)
    {
      const std::string &fname = name_idx.first;
      const int32 fidx = name_idx.second;
      const VVectorT &fdata = record.vector_vert_data(fname);
      for (int32 j = 0; j < verts_per_elem; ++j)
        buffer.m_vector_vert_data[fidx].push_back(fdata.m_data[vtk_2_lex[j]]);
    }

    for (const auto &name_idx : m_vector_elem_name2id)
    {
      const std::string &fname = name_idx.first;
      const int32 fidx = name_idx.second;
      const EVectorT &fdata = record.vector_elem_data(fname);
      buffer.m_vector_elem_data[fidx].push_back(fdata.m_data[0]);
    }

    record.reset_all();
  }


  /** transfer_vec() : Empties a src vector into a dst vector. */
  template <typename T>
  void transfer_vec(std::vector<T> &dst_vec, std::vector<T> &src_vec)
  {
    dst_vec.insert(dst_vec.end(), src_vec.begin(), src_vec.end());
    src_vec.clear();
  }

  /** transfer_each() : Empties each src vector into corresponding dst vector. */
  template <typename T>
  void transfer_each_vec(std::vector<std::vector<T>> &dst, std::vector<std::vector<T>> &src)
  {
    for (int32 vec_idx = 0; vec_idx < dst.size(); ++vec_idx)
    {
      std::vector<T> &dst_vec = dst[vec_idx];
      std::vector<T> &src_vec = src[vec_idx];
      dst_vec.insert(dst_vec.end(), src_vec.begin(), src_vec.end());
      src_vec.clear();
    }
  }


  /** flush_and_close_all_buffers() */
  void DataSetBuilder::flush_and_close_all_buffers()
  {
    // Initialize sizes from m_central_buffer.
    int32 num_timesteps = m_central_buffer.m_num_timesteps;
    int32 total_elems = m_central_buffer.m_num_elems;
    int32 total_verts = m_central_buffer.m_num_verts;

    // Accumulate sizes of inflows.
    for (const DSBBuffer &inbuf : m_inflow_buffers)
    {
      num_timesteps = optimized::max(num_timesteps, inbuf.m_num_timesteps);
      total_elems += inbuf.m_num_elems;
      total_verts += inbuf.m_num_verts;
    }

    // Reserve space in m_central_buffer.
    m_central_buffer.m_timesteps.reserve(total_elems);
    m_central_buffer.m_is_immortal.reserve(total_elems);
    for (std::vector<Vec<Float, 3>> &field : m_central_buffer.m_coord_data)
      field.reserve(total_verts);
    for (std::vector<Vec<Float, 1>> &field : m_central_buffer.m_scalar_vert_data)
      field.reserve(total_verts);
    for (std::vector<Vec<Float, 1>> &field : m_central_buffer.m_scalar_elem_data)
      field.reserve(total_elems);
    for (std::vector<Vec<Float, 3>> &field : m_central_buffer.m_vector_vert_data)
      field.reserve(total_verts);
    for (std::vector<Vec<Float, 3>> &field : m_central_buffer.m_vector_elem_data)
      field.reserve(total_elems);

    // Flush from inflows to m_central_buffer.
    for (DSBBuffer &inbuf : m_inflow_buffers)
    {
      transfer_vec(m_central_buffer.m_timesteps, inbuf.m_timesteps);
      transfer_vec(m_central_buffer.m_is_immortal, inbuf.m_is_immortal);
      transfer_each_vec(m_central_buffer.m_coord_data, inbuf.m_coord_data);
      transfer_each_vec(m_central_buffer.m_scalar_vert_data, inbuf.m_scalar_vert_data);
      transfer_each_vec(m_central_buffer.m_scalar_elem_data, inbuf.m_scalar_elem_data);
      transfer_each_vec(m_central_buffer.m_vector_vert_data, inbuf.m_vector_vert_data);
      transfer_each_vec(m_central_buffer.m_vector_elem_data, inbuf.m_vector_elem_data);

      inbuf.clear_records();
    }

    // Close inflows.
    m_inflow_buffers.clear();

    // Update sizes in metadata.
    m_central_buffer.m_num_timesteps = num_timesteps;
    m_central_buffer.m_num_elems = total_elems;
    m_central_buffer.m_num_verts = total_verts;
  }


  /** coord_data() */
  const std::vector<Vec<Float, 3>> &
  DataSetBuilder::coord_data(int32 idx) const
  {
    return m_central_buffer.m_coord_data.at(idx);
  }

  /** scalar_vert_data() */
  const std::vector<Vec<Float, 1>> &
  DataSetBuilder::scalar_vert_data(int32 idx) const
  {
    return m_central_buffer.m_scalar_vert_data.at(idx);
  }

  /** scalar_elem_data() */
  const std::vector<Vec<Float, 1>> &
  DataSetBuilder::scalar_elem_data(int32 idx) const
  {
    return m_central_buffer.m_scalar_elem_data.at(idx);
  }

  /** vector_vert_data() */
  const std::vector<Vec<Float, 3>> &
  DataSetBuilder::vector_vert_data(int32 idx) const
  {
    return m_central_buffer.m_vector_vert_data.at(idx);
  }

  /** vector_elem_data() */
  const std::vector<Vec<Float, 3>> &
  DataSetBuilder::vector_elem_data(int32 idx) const
  {
    return m_central_buffer.m_vector_elem_data.at(idx);
  }


  /** to_blueprint() */
  int32 DataSetBuilder::to_blueprint(conduit::Node &bp_dataset, int32 cycle) const
  {
    /*
     * https://llnl-conduit.readthedocs.io/en/latest/blueprint_mesh.html#outputting-meshes-for-visualization
     */

    if (num_buffers() > 0)
      std::cout << "Warning: calling to_blueprint() with buffers unflushed!\n";

    const DSBBuffer &buf = this->m_central_buffer;

    const int32 n_elems = buf.m_num_elems;
    const int32 npe = shape_npe[m_shape_mode];

    // Index all element records selected by cycle.
    // TODO if this step gets prohibitively expensive,
    //  sort by cycle on-line in add_XXX_record().
    std::vector<int32> sel;
    for (int32 eid = 0; eid < n_elems; ++eid)
      if (buf.m_timesteps[eid] == cycle || (buf.m_is_immortal[eid] && cycle >= buf.m_timesteps[eid]))
        sel.push_back(eid);

    const int32 n_sel_elems = sel.size();
    const int32 n_sel_verts = n_sel_elems * npe;


    //
    // Init node.
    //
    bp_dataset.reset();
    bp_dataset["state/time"] = (float64) cycle;
    bp_dataset["state/cycle"] = (uint64) cycle;

    conduit::Node &coordsets = bp_dataset["coordsets"];
    conduit::Node &topologies = bp_dataset["topologies"];
    conduit::Node &fields = bp_dataset["fields"];

    //
    // Duplicate fields for each coordset.
    //
    for (const auto &name_idx : m_coord_idx)
    {
      const std::string &cname = name_idx.first;
      const int32 cidx = name_idx.second;

      const std::string topo_name = cname;
      const std::string coordset_name = cname + "_coords";

      //
      // Coordset.
      //
      conduit::Node &coordset = coordsets[coordset_name];
      coordset["type"] = "explicit";
      conduit::Node &coord_vals = coordset["values"];
      coordset["values/x"].set(conduit::DataType::float64(n_sel_verts));
      coordset["values/y"].set(conduit::DataType::float64(n_sel_verts));
      coordset["values/z"].set(conduit::DataType::float64(n_sel_verts));
      float64 *x_vals = coordset["values/x"].value();
      float64 *y_vals = coordset["values/y"].value();
      float64 *z_vals = coordset["values/z"].value();
      const std::vector<Vec<Float, 3>> & in_coord_data = buf.m_coord_data[cidx];
      for (int32 eidx = 0; eidx < n_sel_elems; ++eidx)
        for (int32 nidx = 0; nidx < npe; ++nidx)
        {
          x_vals[eidx * npe + nidx] = (float64) in_coord_data[sel[eidx] * npe + nidx][0];
          y_vals[eidx * npe + nidx] = (float64) in_coord_data[sel[eidx] * npe + nidx][1];
          z_vals[eidx * npe + nidx] = (float64) in_coord_data[sel[eidx] * npe + nidx][2];
        }


      //
      // Topology.
      //
      conduit::Node &topo = topologies[topo_name];
      topo["type"] = "unstructured";
      topo["coordset"] = coordset_name;
      topo["elements/shape"] = "hex";
      topo["elements/connectivity"].set(conduit::DataType::int32(n_sel_verts));
      int32 * conn = topo["elements/connectivity"].value();
      std::iota(conn, conn + n_sel_verts, 0);


      const std::string jstr = "_";

      //
      // Fields.
      //
      for (const auto &name_idx : m_scalar_vert_name2id)  // Scalar vertex fields.
      {
        const std::string &fname = name_idx.first;
        const int32 fidx = name_idx.second;
        const std::string field_name = cname + jstr + fname;

        conduit::Node &field = fields[field_name];
        field["association"] = "vertex";
        field["type"] = "scalar";
        field["topology"] = topo_name;
        field["values"].set(conduit::DataType::float64(n_sel_verts));

        float64 *out_vals = field["values"].value();
        const std::vector<Vec<Float, 1>> &in_field_data = buf.m_scalar_vert_data[fidx];
        for (int32 eidx = 0; eidx < n_sel_elems; ++eidx)
          for (int32 nidx = 0; nidx < npe; ++nidx)
            out_vals[eidx * npe + nidx] = in_field_data[sel[eidx] * npe + nidx][0];
      }

      for (const auto &name_idx : m_scalar_elem_name2id)  // Scalar element fields.
      {
        const std::string &fname = name_idx.first;
        const int32 fidx = name_idx.second;
        const std::string field_name = cname + jstr + fname;

        conduit::Node &field = fields[field_name];
        field["association"] = "element";
        field["type"] = "scalar";
        field["topology"] = topo_name;
        field["values"].set(conduit::DataType::float64(n_sel_elems));

        float64 *out_vals = field["values"].value();
        const std::vector<Vec<Float, 1>> &in_field_data = buf.m_scalar_elem_data[fidx];
        for (int32 eidx = 0; eidx < n_sel_elems; ++eidx)
          out_vals[eidx] = in_field_data[sel[eidx]][0];
      }


      const std::string tangent_names[3] = {"u", "v", "w"};

      for (const auto &name_idx : m_vector_vert_name2id)  // Vector vertex fields.
      {
        const std::string &fname = name_idx.first;
        const int32 fidx = name_idx.second;
        const std::string field_name = cname + jstr + fname;

        constexpr int32 ncomp = 3;
        conduit::Node &field = fields[field_name];
        field["association"] = "vertex";
        field["type"] = "vector";
        field["topology"] = topo_name;
        field["values"];

        float64 * out_vals[ncomp];
        for (int32 d = 0; d < ncomp; ++d)
        {
          field["values"][tangent_names[d]].set(conduit::DataType::float64(n_sel_verts));
          out_vals[d] = field["values"][tangent_names[d]].value();
        }

        const std::vector<Vec<Float, 3>> &in_field_data = buf.m_vector_vert_data[fidx];
        for (int32 eidx = 0; eidx < n_sel_elems; ++eidx)
          for (int32 nidx = 0; nidx < npe; ++nidx)
            for (int32 d = 0; d < ncomp; ++d)
              out_vals[d][eidx * npe + nidx] = in_field_data[sel[eidx] * npe + nidx][d];
      }

      for (const auto &name_idx : m_vector_elem_name2id)  // Vector element fields.
      {
        const std::string &fname = name_idx.first;
        const int32 fidx = name_idx.second;
        const std::string field_name = cname + jstr + fname;

        constexpr int32 ncomp = 3;
        conduit::Node &field = fields[field_name];
        field["association"] = "element";
        field["type"] = "vector";
        field["topology"] = topo_name;
        field["values"];

        float64 * out_vals[ncomp];
        for (int32 d = 0; d < ncomp; ++d)
        {
          field["values"][tangent_names[d]].set(conduit::DataType::float64(n_sel_elems));
          out_vals[d] = field["values"][tangent_names[d]].value();
        }

        const std::vector<Vec<Float, 3>> &in_field_data = buf.m_vector_elem_data[fidx];
        for (int32 eidx = 0; eidx < n_sel_elems; ++eidx)
          for (int32 d = 0; d < ncomp; ++d)
            out_vals[d][eidx] = in_field_data[sel[eidx]][d];
      }
      // End all fields.

    }//for coordset

    return n_sel_elems;
  }

}//namespace dray
