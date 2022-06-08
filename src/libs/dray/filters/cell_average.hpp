#ifndef DRAY_CELL_AVERAGE_HPP
#define DRAY_CELL_AVERAGE_HPP

#include <dray/data_model/collection.hpp>

#include <string>

namespace dray
{

class CellAverage
{
public:
    Collection execute(Collection &input, const std::string &in_field, std::string out_field = "");    
};

}

#endif
