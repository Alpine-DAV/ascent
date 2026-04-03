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

// ---------- General Helpers ----------
std::map<const std::string, FormatCheckFunction> format_checker_functions;

void add_error(conduit::Node &info, const std::string &msg)
{
    if(!info.has_child("errors"))
    {
        info["errors"].reset();
    }
    info["errors"].append() = msg;
}

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
        return true; // schema didn't specify; treat as "accept anything"
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
        add_error(info, "At '" + (path.empty() ? std::string("<root>") : path) +
                        "': unknown schema type '" + schema_defined_type + "'");
        return false;
    }

    if(!ok)
    {
        add_error(info, "Type mismatch at '" + (path.empty() ? std::string("<root>") : path) +
                        "': expected " + schema_defined_type + ", got " + input.dtype().name());
    }

    return ok;
}

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
        int mn = schema["minLength"].to_int();
        if((int)s.size() < mn)
        {
            add_error(info, "String at '" + (path.empty() ? "<root>" : path) +
                            "' is shorter than minLength=" + std::to_string(mn));
            ok = false;
        }
    }

    if(schema.has_child("maxLength"))
    {
        int mx = schema["maxLength"].to_int();
        if((int)s.size() > mx)
        {
            add_error(info, "String at '" + (path.empty() ? "<root>" : path) +
                            "' is longer than maxLength=" + std::to_string(mx));
            ok = false;
        }
    }

    if(schema.has_child("pattern"))
    {
        const std::string pattern = schema["pattern"].as_string();
        const std::string loc = path.empty() ? "<root>" : path;
        
        try
        {
            const std::regex re(pattern);
            if(!std::regex_match(s, re))
            {
                add_error(info, "String at '" + loc +
                                "' does not match pattern '" + pattern + "'");
                ok = false;
            }
        }
        catch(const std::regex_error &)
        {
            add_error(info, "Schema at '" + loc +
                            "' has invalid regex pattern '" + pattern + "'");
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
    const conduit::Node *e = nullptr;
    if(schema.has_child("enum")) e = &schema["enum"];
    else return true;

    if(!input.dtype().is_string()) return true;

    const std::string got = input.as_string();
    for(conduit::index_t i = 0; i < e->number_of_children(); ++i)
    {
        if(got == e->child(i).as_string()) return true;
    }

    add_error(info, "Value not in enum at '" +
                    (path.empty() ? "<root>" : path) +
                    "': got '" + got + "'");
    return false;
}

bool validate_number(const conduit::Node &schema,
                            const conduit::Node &input,
                            conduit::Node &info,
                            const std::string &path)
{
    if(!input.dtype().is_number()) return true;

    const double v = input.to_float64();
    bool ok = true;

    if(schema.has_child("minimum"))
    {
        double mn = schema["minimum"].to_float64();
        if(v < mn)
        {
            add_error(info, "Number at '" + (path.empty() ? "<root>" : path) +
                            "' must be >= " + std::to_string(mn));
            ok = false;
        }
    }

    if(schema.has_child("exclusiveMinimum"))
    {
        double mn = schema["exclusiveMinimum"].to_float64();
        if(v <= mn)
        {
            add_error(info, "Number at '" + (path.empty() ? "<root>" : path) +
                            "' must be > " + std::to_string(mn));
            ok = false;
        }
    }

    if(schema.has_child("maximum"))
    {
        double mx = schema["maximum"].to_float64();
        if(v > mx)
        {
            add_error(info, "Number at '" + (path.empty() ? "<root>" : path) +
                            "' must be <= " + std::to_string(mx));
            ok = false;
        }
    }

    if(schema.has_child("exclusiveMaximum"))
    {
        double mx = schema["exclusiveMaximum"].to_float64();
        if(v >= mx)
        {
            add_error(info, "Number at '" + (path.empty() ? "<root>" : path) +
                            "' must be < " + std::to_string(mx));
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
        return true; // type error handled elsewhere
    }

    bool ok = true;
    const conduit::Node &req = schema["required"];
    for(conduit::index_t i = 0; i < req.number_of_children(); ++i)
    {
        const std::string k = req.child(i).as_string();
        if(!input.has_child(k))
        {
            add_error(info, "Missing required field '" + conduit::utils::join_path(path, k) + "'");
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
            add_error(info, "Field '" + conduit::utils::join_file_path(path, k) + "' is forbidden by schema");
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

    const conduit::Node &c = schema["constraints/const"];
    // Only implement string const for now (that’s all we used above)
    if(input.dtype().is_string() && c.dtype().is_string())
    {
        const std::string got = input.as_string();
        const std::string expect = c.as_string();
        if(got != expect)
        {
            add_error(info, "Value mismatch at '" + (path.empty() ? std::string("<root>") : path) +
                            "': expected '" + expect + "', got '" + got + "'");
            return false;
        }
    }
    return true;
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
    const conduit::Node &nc = schema["constraints/not_const"];
    for(conduit::index_t i = 0; i < nc.number_of_children(); ++i)
    {
        const std::string field = nc[i].name();
        const conduit::Node &forbidden_val = nc[field];

        if(input.has_child(field) && input[field].dtype().is_string() && forbidden_val.dtype().is_string())  
        {
            const std::string got = input[field].as_string();
            const std::string bad = forbidden_val.as_string();
            if(got == bad)
            {
                add_error(info, "Value forbidden at '" + conduit::utils::join_file_path(path, field) +
                                "': must not be '" + bad + "'");
                ok = false;
            }
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
            add_error(info, "Unexpected field '" + conduit::utils::join_file_path(path, k) +
                            "' (additionalProperties=false)");
            ok = false;
        }
    }
    return ok;
}

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
        const std::string trigger = deps[i].name();
        if(!input.has_child(trigger))
        {
            continue;
        }

        const conduit::Node &reqs = deps[trigger];
        for(conduit::index_t j = 0; j < reqs.number_of_children(); ++j)
        {
        const std::string needed = reqs.child(j).as_string();
        if(!input.has_child(needed))
            {
                add_error(info, "Dependency violation at '"
                                + (path.empty() ? std::string("<root>") : path)
                                + "': if '" + trigger + "' is provided, '" 
                                + needed + "' must also be provided");
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

    std::vector<std::string> present;
    present.reserve((size_t)keys.number_of_children());

    for(conduit::index_t i = 0; i < keys.number_of_children(); ++i)
    {
        const std::string k = keys.child(i).as_string();
        if(input.has_child(k))
        {
            present.push_back(k);
        }
    }

    const int count = (int)present.size();
    const bool ok = allow_none ? (count <= 1) : (count == 1);

    if(ok)
    {
        return true;
    }

    std::ostringstream oss;
    oss << "Exclusive-children violation at '"
        << (path.empty() ? "<root>" : path) << "': expected "
        << (allow_none ? "zero or one" : "exactly one")
        << " of {";

    for(conduit::index_t i = 0; i < keys.number_of_children(); ++i)
    {
        if(i != 0)
        {
            oss << ", ";
        }
        oss << keys.child(i).as_string();
    }
    oss << "}";

    if(count > 0)
    {
        oss << ", but found: {";
        for(size_t i = 0; i < present.size(); ++i)
        {
            if(i != 0)
            {
                oss << ", ";
            }
            oss << present[i];
        }
        oss << "}";
    }
    else
    {
        oss << ", but found none";
    }

    add_error(info, oss.str());
    return false;
}

static bool validate_all_of(const conduit::Node &schema,
                            const conduit::Node &input,
                            conduit::Node &info,
                            const std::string &path)
{
    if(!schema.has_child("allOf")) return true;

    bool ok = true;
    const conduit::Node &opts = schema["allOf"];
    for(conduit::index_t i = 0; i < opts.number_of_children(); ++i)
    {
        ok = validate_node(opts.child(i), input, info, path) && ok;
    }
    return ok;
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

    // keep at most one representative failure per option for clarity
    std::vector<std::string> option_msgs;

    for(conduit::index_t i = 0; i < opts.number_of_children(); ++i)
    {
        conduit::Node tmp;
        tmp.reset();

        bool ok = validate_node(opts.child(i), input, tmp, path);

        if(ok)
        {
            matches++;
        }
        else
        {
            // pick first error as representative
            if(tmp.has_child("errors") && tmp["errors"].number_of_children() > 0)
            {
                option_msgs.push_back(tmp["errors"].child(0).as_string());
            }
            else
            {
                option_msgs.push_back("Option " + std::to_string((int)i) + " failed");
            }
        }
    }

    if(matches == 1)
    {
        return true;
    }

    std::ostringstream oss;
    oss << "oneOf violation at '" << (path.empty() ? "<root>" : path) << "': ";
    if(matches == 0)
    {
        oss << "input did not match any supported schemas";
    }
    else
    {
        oss << "input matched " << matches << " options (ambiguous)";
    }
    add_error(info, oss.str());

    // give a couple of hints
    for(size_t i = 0; i < option_msgs.size() && i < 2; ++i)
    {
        add_error(info, std::string("  hint: ") + option_msgs[i]);
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

    // collect a couple representative failures for hints
    std::vector<std::string> option_msgs;

    for(conduit::index_t i = 0; i < opts.number_of_children(); ++i)
    {
        conduit::Node tmp;
        tmp.reset();

        bool ok = validate_node(opts.child(i), input, tmp, path);

        if(ok)
        {
            matches++;
        }
        else if(tmp.has_child("errors") && tmp["errors"].number_of_children() > 0)
        {
            option_msgs.push_back(tmp["errors"].child(0).as_string());
        }
    }

    if(matches >= 1)
    {
        return true;
    }

    add_error(info, "anyOf violation at '" + (path.empty() ? std::string("<root>") : path) +
                    "': input did not match any option");

    for(size_t i = 0; i < option_msgs.size() && i < 2; ++i)
    {
        add_error(info, std::string("  hint: ") + option_msgs[i]);
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
        add_error(info, "Invalid expression at '" + (path.empty() ? std::string("<root>") : path) + "': " + err);
    }
    return ok;
}

bool validate_array(const conduit::Node &schema,
                    const conduit::Node &input,
                    conduit::Node &info,
                    const std::string &path)
{
    bool ok = true;

    const auto data_type = input.dtype();
    const conduit::index_t count = (data_type.is_list() || data_type.is_object()) ? input.number_of_children() : data_type.number_of_elements();

    // Json Schema uses min/max bounds for array length.
    if(schema.has_child("minItems"))
    {
        const conduit::index_t min_items = (conduit::index_t)schema["minItems"].to_int();
        if(count < min_items)
        {
            add_error(info,
                      "Array at '" + (path.empty() ? std::string("<root>") : path) +
                      "' has too few items: expected at least " +
                      std::to_string((long long)min_items) + ", got " +
                      std::to_string((long long)count));
            ok = false;
        }
    }

    if(schema.has_child("maxItems"))
    {
        const conduit::index_t max_items = (conduit::index_t)schema["maxItems"].to_int();
        if(count > max_items)
        {
            add_error(info,
                      "Array at '" + (path.empty() ? std::string("<root>") : path) +
                      "' has too many items: expected at most " +
                      std::to_string((long long)max_items) + ", got " +
                      std::to_string((long long)count));
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
