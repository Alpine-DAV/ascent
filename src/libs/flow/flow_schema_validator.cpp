//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


//-----------------------------------------------------------------------------
///
/// file: flow_schema_validator.cpp
///
//-----------------------------------------------------------------------------

#include "flow_schema_validator.hpp"

// standard lib includes
#include <vector>
#include <sstream>
#include <regex>

//-----------------------------------------------------------------------------
// -- begin flow --
//-----------------------------------------------------------------------------
namespace flow
{

//-----------------------------------------------------------------------------
// -- begin flow::schema --
//-----------------------------------------------------------------------------
namespace schema
{

//-----------------------------------------------------------------------------
// -- begin flow::schema::detail --
//-----------------------------------------------------------------------------
namespace detail
{

// ---------- Error Message Helpers ----------
size_t MAX_OPTIONS = 3;

std::string node_path(const std::string &path)
{
    return path.empty() ? "<root>" : path;
}

std::string first_error_string(const conduit::Node &info,
                               const std::string &fallback = "validation failed")
{
    if(info.has_child("errors") && info["errors"].number_of_children() > 0)
    {
        return info["errors"].child(0).as_string();
    }
    return fallback;
}

void add_input_error(conduit::Node &info,
                     const std::string &path,
                     const std::string &rule,
                     const std::string &message,
                     const std::string &expected = "")
{
    std::ostringstream oss;
    oss << "Validation failed at '" << node_path(path) << "'";
    if(!rule.empty()) oss << " (" << rule << ")";
    oss << ": " << message << ".";
    if(!expected.empty()) oss << " Expected " << expected << ".";

    info["errors"].append() = oss.str();
}

void add_schema_error(conduit::Node &info,
                      const std::string &path,
                      const std::string &rule,
                      const std::string &message)
{
    std::ostringstream oss;
    oss << "Schema error near '" << node_path(path) << "'";
    if(!rule.empty()) oss << " (" << rule << ")";
    oss << ": " << message  << ".";

    info["errors"].append() = oss.str();
}

// ---------- Type Checking ----------
std::string get_type_string(const conduit::Node &schema)
{
    if(schema.has_child("type") && schema["type"].dtype().is_string())
    {
        return schema["type"].as_string();
    }
    return "";
}

bool check_type(const conduit::Node &input,
                const conduit::Node &schema,
                conduit::Node &info,
                const std::string &path)
{
    const std::string schema_defined_type = get_type_string(schema);
    if(schema_defined_type.empty())
    {
        return true;
    }

    const auto data_type = input.dtype();
    bool ok = true;

    if(schema_defined_type == "object")
    {
        ok = data_type.is_object();
    }
    else if(schema_defined_type == "string")
    {
        ok = data_type.is_string();
    }
    else if(schema_defined_type == "number")
    {
        ok = data_type.is_number();
    }
    else if(schema_defined_type == "integer")
    {
        ok = data_type.is_integer();
    }
    else if(schema_defined_type == "array")
    {
        ok = (data_type.is_list() ||
             (data_type.is_number() && data_type.number_of_elements() >= 1) ||
             data_type.is_object());
    }
    else
    {
        add_schema_error(info, path, "type",
                         "unknown schema type '" + schema_defined_type + "'");
        return false;
    }

    if(!ok)
    {
        std::string expected = "type '" + schema_defined_type + "'";
        if(schema_defined_type == "array")
        {
            expected += " (Conduit list, object, or numeric leaf array)";
        }

        add_input_error(info, path, "type",
                        "type mismatch",
                        expected);
    }

    return ok;
}

// ---------- Format Checking ----------
std::map<const std::string, FormatCheckFunction> format_checker_functions;

bool validate_format(const conduit::Node &schema,
                     const conduit::Node &input,
                     conduit::Node &info,
                     const std::string &path)
{
    if(!schema.has_child("format") || !schema["format"].dtype().is_string())
    {
        return true;
    }

    const std::string fmt = schema["format"].as_string();
    if (format_checker_functions.find(fmt) == format_checker_functions.end())
    {
        return true;
    }

    std::string err;
    bool ok = format_checker_functions[fmt](input.as_string(), err);
    if(!ok)
    {
        add_input_error(info, path, "format - " + fmt, err);
    }
    return ok;
}

// ---------- Validate Nodes ----------
bool validate_string(const conduit::Node &schema,
                     const conduit::Node &input,
                     conduit::Node &info,
                     const std::string &path)
{
    if(!input.dtype().is_string()) return true;

    const std::string s = input.as_string();
    bool ok = true;

    if(schema.has_child("minLength"))
    {
        const int min_length = schema["minLength"].to_int();
        if((int)s.size() < min_length)
        {
            add_input_error(info, path, "minLength",
                            "string is too short. Length is " + std::to_string((int)s.size()),
                            "string length >= " + std::to_string(min_length));
            ok = false;
        }
    }

    if(schema.has_child("maxLength"))
    {
        const int max_length = schema["maxLength"].to_int();
        if((int)s.size() > max_length)
        {
            add_input_error(info, path, "maxLength",
                            "string is too long. Length is " + std::to_string((int)s.size()),
                            "string length <= " + std::to_string(max_length));
            ok = false;
        }
    }

    if(schema.has_child("pattern"))
    {
        const std::string pattern = schema["pattern"].as_string();

        try
        {
            const std::regex re(pattern);
            if(!std::regex_search(s, re))
            {
                add_input_error(info, path, "pattern",
                                "string does not match required pattern '" + pattern + "'");
                ok = false;
            }
        }
        catch(const std::regex_error &)
        {
            add_schema_error(info, path, "pattern", "invalid regex pattern '" + pattern +"'");
            ok = false;
        }
    }

    return ok;
}

bool validate_enum(const conduit::Node &schema,
                   const conduit::Node &input,
                   conduit::Node &info,
                   const std::string &path)
{
    if(!schema.has_child("enum")) return true;
    if(!input.dtype().is_string()) return true;

    const conduit::Node &e = schema["enum"];

    const std::string input_value = input.as_string();
    std::ostringstream allowed;
    allowed << "{";
    for(conduit::index_t i = 0; i < e.number_of_children(); ++i)
    {
        if(i > 0) allowed << ", ";
        const std::string v = e.child(i).as_string();
        allowed << v;
        if(input_value == v) return true;
    }
    allowed << "}";

    add_input_error(info, path, "enum",
                    "'" + input_value + "' is not one of the allowed enum entries",
                    "one of " + allowed.str());

    return false;
}

bool validate_number(const conduit::Node &schema,
                     const conduit::Node &input,
                     conduit::Node &info,
                     const std::string &path)
{
    if(get_type_string(schema) != "number" && get_type_string(schema) != "integer") return true;

    const double v = input.to_float64();
    bool ok = true;

    if(schema.has_child("minimum"))
    {
        const double min_val = schema["minimum"].to_float64();
        if(v < min_val)
        {
            std::ostringstream exp;
            exp << "number >= " << min_val;
            add_input_error(info, path, "minimum",
                            std::to_string(v) + " is below the allowed minimum",
                            exp.str());
            ok = false;
        }
    }

    if(schema.has_child("exclusiveMinimum"))
    {
        const double exclusive_min_val = schema["exclusiveMinimum"].to_float64();
        if(v <= exclusive_min_val)
        {
            std::ostringstream exp;
            exp << "number > " << exclusive_min_val;
            add_input_error(info, path, "exclusiveMinimum",
                            std::to_string(v) + " is not greater than the exclusive minimum",
                            exp.str());
            ok = false;
        }
    }

    if(schema.has_child("maximum"))
    {
        const double max_val = schema["maximum"].to_float64();
        if(v > max_val)
        {
            std::ostringstream exp;
            exp << "number <= " << max_val;
            add_input_error(info, path, "maximum",
                            std::to_string(v) + " is above the allowed maximum",
                            exp.str());
            ok = false;
        }
    }

    if(schema.has_child("exclusiveMaximum"))
    {
        const double exclusive_max_val = schema["exclusiveMaximum"].to_float64();
        if(v >= exclusive_max_val)
        {
            std::ostringstream exp;
            exp << "number < " << exclusive_max_val;
            add_input_error(info, path, "exclusiveMaximum",
                            std::to_string(v) + " is not less than the exclusive maximum",
                            exp.str());
            ok = false;
        }
    }

    return ok;
}

// ---------- Object-Specific Validation Helpers ----------

// Earlier declaration so validate node can be refrenced by helpers.
bool validate_node(const conduit::Node &schema,
                   const conduit::Node &input,
                   conduit::Node &info,
                   const std::string &path);

bool validate_required(const conduit::Node &schema,
                       const conduit::Node &input,
                       conduit::Node &info,
                       const std::string &path)
{
    if(!schema.has_child("required") || !input.dtype().is_object())
    {
        return true;
    }

    bool ok = true;
    const conduit::Node &req = schema["required"];

    for(conduit::index_t i = 0; i < req.number_of_children(); ++i)
    {
        const std::string k = req.child(i).as_string();
        if(!input.has_child(k))
        {
            const std::string missing_path = conduit::utils::join_path(path, k);
            add_input_error(info, missing_path, "required",
                            "required field is missing",
                            "field to be present");
            ok = false;
        }
    }

    return ok;
}

bool validate_forbid(const conduit::Node &schema,
                     const conduit::Node &input,
                     conduit::Node &info,
                     const std::string &path)
{
    if(!schema.has_path("constraints/forbid") || !input.dtype().is_object())
    {
        return true;
    }

    bool ok = true;
    const conduit::Node &forbid = schema["constraints/forbid"];
    for(conduit::index_t i = 0; i < forbid.number_of_children(); ++i)
    {
        const std::string k = forbid.child(i).as_string();
        if(input.has_child(k))
        {
            const std::string forbidden_path = conduit::utils::join_file_path(path, k);
            add_input_error(info, forbidden_path, "forbidden",
                            "forbidden field is present");
            ok = false;
        }
    }
    return ok;
}

bool validate_const(const conduit::Node &schema,
                    const conduit::Node &input,
                    conduit::Node &info,
                    const std::string &path)
{
    if(!schema.has_path("constraints/const"))
    {
        return true;
    }

    const conduit::Node &const_schema = schema["constraints/const"];

    // String const
    if(const_schema.dtype().is_string())
    {
        const std::string expect = const_schema.as_string();
        if(!input.dtype().is_string())
        {
            add_input_error(info, path, "const",
                "expected constant string value '" +
                expect + "', but got type '" + input.dtype().name() + "'");
            return false;
        }

        const std::string input_value = input.as_string();
        if(input_value != expect)
        {
            add_input_error(info, path, "const",
                "expected exact value '" +
                expect + "', but got '" + input_value + "'");
            return false;
        }

        return true;
    }

    // Integer const
    if(const_schema.dtype().is_integer())
    {
        const long long expect = (long long)const_schema.to_int64();
        if(!input.dtype().is_integer())
        {
            add_input_error(info, path, "const",
                "expected constant integer value " +
                std::to_string(expect) + ", but got type '" + input.dtype().name());
            return false;
        }

        const long long input_value = (long long)input.to_int64();
        if(input_value != expect)
        {
            add_input_error(info, path, "const",
                "expected exact value " +
                std::to_string(expect) + ", but got " + std::to_string(input_value));
            return false;
        }

        return true;
    }

    // General numeric const
    if(const_schema.dtype().is_number())
    {
        const double expect = const_schema.to_float64();
        if(!input.dtype().is_number())
        {
            add_input_error(info, path, "const",
                "expected constant numeric value " +
                std::to_string(expect) + ", but got type '" + input.dtype().name() + "'");
            return false;
        }

        const double got = input.to_float64();
        if(got != expect)
        {
            add_input_error(info, path, "const",
                "expected exact value " +
                std::to_string(expect) + ", got " + std::to_string(got));
            return false;
        }

        return true;
    }

    add_schema_error(info, path, "const",
        "unsupported const type '" + const_schema.dtype().name() +
        "'. Currently supported const types are string, integer, and number");

    return false;
}

bool validate_not_const_fields(const conduit::Node &schema,
                               const conduit::Node &input,
                               conduit::Node &info,
                               const std::string &path)
{
    if(!schema.has_path("constraints/not_const") || !input.dtype().is_object())
    {
        return true;
    }

    bool ok = true;
    const conduit::Node &not_const_schema = schema["constraints/not_const"];

    for(conduit::index_t i = 0; i < not_const_schema.number_of_children(); ++i)
    {
        const conduit::Node &forbidden_val = not_const_schema.child(i);
        const std::string field = forbidden_val.name();
        const std::string field_path = conduit::utils::join_file_path(path, field);

        if(!input.has_child(field)) continue;

        const conduit::Node &actual = input[field];

        if(forbidden_val.dtype().is_string())
        {
            if(actual.dtype().is_string() &&
               actual.as_string() == forbidden_val.as_string())
            {
                add_input_error(info, path, "not const",
                    "value '" + actual.as_string() + "' is forbidden here");
                ok = false;
            }
        }
        else if(forbidden_val.dtype().is_integer())
        {
            if(actual.dtype().is_integer() &&
               actual.to_int64() == forbidden_val.to_int64())
            {
                add_input_error(info, path, "not const",
                    "value " + std::to_string((long long)actual.to_int64()) +
                    " is forbidden here");
                ok = false;
            }
        }
        else if(forbidden_val.dtype().is_number())
        {
            if(actual.dtype().is_number() &&
               actual.to_float64() == forbidden_val.to_float64())
            {
                add_input_error(info, path, "not const",
                    "value " + std::to_string(actual.to_float64()) +
                    " is forbidden here");
                ok = false;
            }
        }
        else
        {
            add_schema_error(info, path, "not const",
                "unsupported forbidden value type '" + forbidden_val.dtype().name() +
                "'. Currently supported types are string, integer, and number");
            ok = false;
        }
    }

    return ok;
}

bool validate_properties(const conduit::Node &schema,
                         const conduit::Node &input,
                         conduit::Node &info,
                         const std::string &path)
{
    if(!schema.has_child("properties") || !input.dtype().is_object())
    {
        return true;
    }

    bool ok = true;
    const conduit::Node &props = schema["properties"];
    for(conduit::index_t i = 0; i < props.number_of_children(); ++i)
    {
        const std::string k = props[i].name();
        if(input.has_child(k))
        {
            ok = validate_node(props[k], input[k], info, conduit::utils::join_file_path(path, k)) && ok;
        }
    }

    return ok;
}

bool validate_additional_properties(const conduit::Node &schema,
                                    const conduit::Node &input,
                                    conduit::Node &info,
                                    const std::string &path)
{
    if(!input.dtype().is_object())
    {
        return true;
    }

    bool allow_additional = true;
    if(schema.has_child("additionalProperties"))
    {
        allow_additional = schema["additionalProperties"].to_int() != 0;
    }

    if(allow_additional)
    {
        return true;
    }

    const bool has_props = schema.has_child("properties");
    const conduit::Node props_dummy;
    const conduit::Node &props = has_props ? schema["properties"] : props_dummy;

    bool ok = true;
    for(conduit::index_t i = 0; i < input.number_of_children(); ++i)
    {
        const std::string k = input[i].name();
        if(!has_props || !props.has_child(k))
        {
            add_input_error(info,
                            conduit::utils::join_file_path(path, k),
                            "additionalProperties",
                            "unexpected additional field is not allowed here");
            ok = false;
        }
    }

    return ok;
}

// ---------- Schema Constraints ----------
bool validate_dependencies(const conduit::Node &schema,
                           const conduit::Node &input,
                           conduit::Node &info,
                           const std::string &path)
{
    if(!schema.has_path("constraints/dependencies") || !input.dtype().is_object())
    {
        return true;
    }

    bool ok = true;
    const conduit::Node &deps = schema["constraints/dependencies"];

    for(conduit::index_t i = 0; i < deps.number_of_children(); ++i)
    {
        const std::string dep_trigger = deps[i].name();
        if(!input.has_child(dep_trigger))
        {
            continue;
        }

        const conduit::Node &reqs = deps[dep_trigger];
        for(conduit::index_t j = 0; j < reqs.number_of_children(); ++j)
        {
            const std::string needed = reqs.child(j).as_string();
            if(!input.has_child(needed))
            {
                add_input_error(info, path, "dependencies",
                                "field '" + needed + "' is required because '" +
                                dep_trigger + "' is present",
                                "'" + dep_trigger + "' implies '" + needed + "'");
                ok = false;
            }
        }
    }

    return ok;
}

bool validate_exclusive_children(const conduit::Node &schema,
                                 const conduit::Node &input,
                                 conduit::Node &info,
                                 const std::string &path)
{
    if(!schema.has_path("constraints/exclusiveChildren") || !input.dtype().is_object())
    {
        return true;
    }

    const conduit::Node &keys = schema["constraints/exclusiveChildren"];
    const bool allow_none = schema.has_path("constraints/allowNoneInExclusiveGroup")
                            ? (schema["constraints/allowNoneInExclusiveGroup"].to_int() != 0)
                            : true;

    int count = 0;

    std::ostringstream allowed, found;
    allowed << "{";
    found << "{";
    for(conduit::index_t i = 0; i < keys.number_of_children(); ++i)
    {
        if(i > 0) allowed << ", ";
        const std::string k = keys.child(i).as_string();
        allowed << k;
        if(input.has_child(k))
        {
            count++;
            if(count > 0) 
            {
                found << ", ";
            }
            found << k;
        }
    }
    allowed << "}";
    found << "}";

    if(allow_none ? (count <= 1) : (count == 1))
    {
        return true;
    }


    const std::string expected =
        allow_none ? "zero or one of " + allowed.str()
                   : "exactly one of " + allowed.str();

    const std::string msg =
        (count == 0)
            ? "none of the mutually-exclusive fields are present"
            : "multiple mutually-exclusive fields are present: " + found.str();

    add_input_error(info, path, "exclusiveChildren", msg, expected);
    return false;
}

// ---------- Multi Option Schemas ----------
static bool validate_all_of(const conduit::Node &schema,
                            const conduit::Node &input,
                            conduit::Node &info,
                            const std::string &path)
{
    if(!schema.has_child("allOf")) return true;

    const conduit::Node &opts = schema["allOf"];
    int matches = 0;
    std::vector<std::string> option_msgs;

    for(conduit::index_t i = 0; i < opts.number_of_children(); ++i)
    {
        conduit::Node tmp;
        tmp.reset();

        const bool ok = validate_node(opts.child(i), input, tmp, path);
        if(ok)
        {
            matches++;
        }
        else
        {
            option_msgs.push_back(first_error_string(tmp, "failed"));
        }
    }

    if (matches == opts.number_of_children()) return true;

    std::ostringstream msg;
    msg << "expected all of the " << opts.number_of_children()
        << " schema options to match, but ";
    if(matches == 0) msg << "none matched";
    else msg << "only " << matches << " matched";

    add_input_error(info, path, "all of", msg.str());

    for(size_t i = 0; i < option_msgs.size() && i < MAX_OPTIONS; ++i)
    {
        info["errors"].append() = "    Option " + std::to_string((int)i) + " hint: " + option_msgs[i];
    }

    return false;
}

bool validate_one_of(const conduit::Node &schema,
                     const conduit::Node &input,
                     conduit::Node &info,
                     const std::string &path)
{
    if(!schema.has_child("oneOf"))
    {
        return true;
    }

    const conduit::Node &opts = schema["oneOf"];
    int matches = 0;
    std::vector<std::string> option_msgs;

    for(conduit::index_t i = 0; i < opts.number_of_children(); ++i)
    {
        conduit::Node tmp;
        tmp.reset();

        const bool ok = validate_node(opts.child(i), input, tmp, path);
        if(ok)
        {
            matches++;
        }
        else
        {
            option_msgs.push_back(first_error_string(tmp, "failed"));
        }
    }

    if(matches == 1)
    {
        return true;
    }

    std::ostringstream msg;
    msg << "expected exactly one of " << opts.number_of_children()
        << " schema options to match, but ";
    if(matches == 0)
    {
        msg << "none matched";
    }
    else
    {
        msg << matches << " matched";
    }

    add_input_error(info, path, "one of", msg.str());

    // give a couple of hints
    for(size_t i = 0; i < option_msgs.size() && i < 2; ++i)
    {
        info["errors"].append() = "    Option " + std::to_string((int)i) + " hint: " + option_msgs[i];
    }

    return false;
}

bool validate_any_of(const conduit::Node &schema,
                     const conduit::Node &input,
                     conduit::Node &info,
                     const std::string &path)
{
    if(!schema.has_child("anyOf"))
    {
        return true;
    }

    const conduit::Node &opts = schema["anyOf"];
    int matches = 0;
    std::vector<std::string> option_msgs;

    for(conduit::index_t i = 0; i < opts.number_of_children(); ++i)
    {
        conduit::Node tmp;
        tmp.reset();

        const bool ok = validate_node(opts.child(i), input, tmp, path);
        if(ok)
        {
            matches++;
        }
        else
        {
            option_msgs.push_back(first_error_string(tmp, "failed"));
        }
    }

    if(matches >= 1)
    {
        return true;
    }

    add_input_error(info, path, "any of",
                    "input did not match any allowed schema option",
                    "at least one schema option to match");

    for(size_t i = 0; i < option_msgs.size() && i < MAX_OPTIONS; ++i)
    {
        info["errors"].append() = "    Option " + std::to_string((int)i) + " hint: " + option_msgs[i];
    }

    return false;
}

bool validate_object(const conduit::Node &schema,
                     const conduit::Node &input,
                     conduit::Node &info,
                     const std::string &path)
{
    bool ok = true;

    // Base checks first
    ok = validate_required(schema, input, info, path) && ok;
    ok = validate_forbid(schema, input, info, path) && ok;
    ok = validate_not_const_fields(schema, input, info, path) && ok;
    ok = validate_dependencies(schema, input, info, path) && ok;
    ok = validate_exclusive_children(schema, input, info, path) && ok;

    // Enforce unknown fields after we know properties
    ok = validate_additional_properties(schema, input, info, path) && ok;

    // Recurse into declared properties that exist in input
    ok = validate_properties(schema, input, info, path) && ok;

    // Finally, enforce oneOf (treating options as extra constraints on this same object)
    ok = validate_all_of(schema, input, info, path) && ok;
    ok = validate_one_of(schema, input, info, path) && ok;
    ok = validate_any_of(schema, input, info, path) && ok;

    return ok;
}

bool validate_array(const conduit::Node &schema,
                    const conduit::Node &input,
                    conduit::Node &info,
                    const std::string &path)
{
    bool ok = true;

    const auto data_type = input.dtype();
    const int count = (data_type.is_list() || data_type.is_object())
                          ? (long long)input.number_of_children()
                          : (long long)data_type.number_of_elements();

    // Json Schema uses min/max bounds for array length.
    if(schema.has_child("minItems"))
    {
        const int min_items = schema["minItems"].to_int();
        if(count < min_items)
        {
            add_input_error(info, path, "minItems",
                      "array has too few items",
                      "at least " + std::to_string(min_items) + ", got " + std::to_string(count));
            ok = false;
        }
    }

    if(schema.has_child("maxItems"))
    {
        const int max_items = schema["maxItems"].to_int();
        if(count > max_items)
        {
            add_input_error(info, path, "maxItems",
                      "array has too many items",
                      "expected at most " + std::to_string(max_items) + ", got " + std::to_string(count));
            ok = false;
        }
    }

    if(!schema.has_child("items"))
    {
        return ok; // unconstrained items
    }

    const conduit::Node &item_schema = schema["items"];
    if(data_type.is_list() || data_type.is_object())
    {
        for(conduit::index_t i = 0; i < count; ++i)
        {
            ok = validate_node(item_schema, input.child(i), info,
                            path + "[" + std::to_string((int)i) + "]") && ok;
        }
    }
    else if(data_type.is_number() && data_type.number_of_elements() >= 1)
    {
        // Conduit numeric leaf arrays: validate each element against the item schema.
        // Use an external scalar node that aliases the i'th element.
        const conduit::DataType::TypeID type_id =
            static_cast<conduit::DataType::TypeID>(data_type.id());

        const conduit::index_t byte_stride = data_type.stride();
        const unsigned char *base =
            static_cast<const unsigned char*>(input.data_ptr());

        for(conduit::index_t i = 0; i < count; ++i)
        {
            conduit::Node element;
            element.set_external(conduit::DataType(type_id, 1),
                                 const_cast<unsigned char*>(base + i * byte_stride));

            ok = validate_node(item_schema, element, info,
                               path + "[" + std::to_string((int)i) + "]") && ok;
        }
    }
    return ok;
}

bool validate_node(const conduit::Node &schema,
                   const conduit::Node &input,
                   conduit::Node &info,
                   const std::string &path)
{
    if (schema.has_path("constraints/skip") && schema["constraints/skip"].to_int() != 0)
    {
        return true;
    }

    const std::string schema_defined_type = get_type_string(schema);
    if(schema_defined_type == "object" && input.dtype().is_empty())
    {
        conduit::Node empty_obj;
        empty_obj.set(conduit::DataType::object());
        return validate_object(schema, empty_obj, info, path);
    }
    if(schema_defined_type == "array" && input.dtype().is_empty())
    {
        conduit::Node empty_list;
        empty_list.set(conduit::DataType::list());
        return validate_array(schema, empty_list, info, path);
    }
    
    bool ok = true;

    ok = check_type(input, schema, info, path) && ok;
    ok = validate_const(schema, input, info, path) && ok;
    ok = validate_enum(schema, input, info, path) && ok;

    if(schema_defined_type == "string")
    {
        ok = validate_string(schema, input, info, path) && ok;
    }

    if(schema_defined_type == "number" || schema_defined_type == "integer")
    {
        ok = validate_number(schema, input, info, path) && ok;
    }

    if(!ok)
    {
        return false; // type mismatch stops recursion
    }

    if(schema_defined_type == "object")
    {
        return validate_object(schema, input, info, path);
    }
    if (schema_defined_type == "array")
    {
        return validate_array(schema, input, info, path);
    }

    ok = validate_all_of(schema, input, info, path) && ok;
    ok = validate_one_of(schema, input, info, path) && ok;
    ok = validate_any_of(schema, input, info, path) && ok;
    ok = validate_format(schema, input, info, path) && ok;

    return ok;
}

};
//-----------------------------------------------------------------------------
// -- end flow::schema::detail --
//-----------------------------------------------------------------------------

void register_format_checker(const std::string &format_name, FormatCheckFunction callback)
{
    detail::format_checker_functions[format_name] = callback;
}

// ---------- Schema Validation Entry-Point ----------
bool validate(const conduit::Node &schema,
              const conduit::Node &input,
              conduit::Node &info)
{
    info.reset();
    bool ok = detail::validate_node(schema, input, info, "");

    if(!ok && !info.has_child("errors"))
    {
        info["errors"].append() = "Validation failed (no details)";
    }

    return ok;
}


};
//-----------------------------------------------------------------------------
// -- end flow::schema --
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end flow --
//-----------------------------------------------------------------------------
