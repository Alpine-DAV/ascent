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

// ---------- General Helpers ----------
static void add_error(conduit::Node &info, const std::string &msg)
{
    if(!info.has_child("errors"))
    {
        info["errors"].reset();
    }
    info["errors"].append() = msg;
}

static std::string get_type_string(const conduit::Node &schema)
{
    if(schema.has_child("type") && schema["type"].dtype().is_string())
    {
        return schema["type"].as_string();
    }
    return "";
}

static bool check_type(const conduit::Node &input,
                       const conduit::Node &schema,
                       conduit::Node &info,
                       const std::string &path)
{
    const std::string schema_defined_type = get_type_string(schema);
    if(schema_defined_type.empty()) return true; // schema didn't specify; treat as "accept anything"

    const auto data_type = input.dtype();
    bool ok = true;

    if(schema_defined_type == "object") ok = data_type.is_object();
    else if(schema_defined_type == "string") ok = data_type.is_string();
    else if(schema_defined_type == "number") ok = data_type.is_number();
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

// ---------- Object-Specific Validation Helpers ----------
static bool validate_required(const conduit::Node &schema,
                              const conduit::Node &input,
                              conduit::Node &info,
                              const std::string &path)
{
    if(!schema.has_child("required")) return true;
    if(!input.dtype().is_object()) return true; // type error handled elsewhere

    bool ok = true;
    const conduit::Node &req = schema["required"];
    for(conduit::index_t i = 0; i < req.number_of_children(); ++i)
    {
        const std::string k = req.child(i).as_string();
        if(!input.has_child(k))
        {
            add_error(info, "Missing required field '" + conduit::utils::join_file_path(path, k) + "'");
            ok = false;
        }
    }
    return ok;
}

static bool validate_forbid(const conduit::Node &schema,
                            const conduit::Node &input,
                            conduit::Node &info,
                            const std::string &path)
{
    if(!schema.has_path("constraints/forbid")) return true;
    if(!input.dtype().is_object()) return true;

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

static bool validate_properties(const conduit::Node &schema,
                                const conduit::Node &input,
                                conduit::Node &info,
                                const std::string &path)
{
    if(!schema.has_child("properties")) return true;
    if(!input.dtype().is_object()) return true;

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

static bool validate_additional_properties(const conduit::Node &schema,
                                           const conduit::Node &input,
                                           conduit::Node &info,
                                           const std::string &path)
{
    if(!input.dtype().is_object()) return true;

    bool allow_additional = true;
    if(schema.has_child("additionalProperties"))
    {
        allow_additional = schema["additionalProperties"].to_int() != 0;
    }

    if(allow_additional) return true;

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

static bool validate_dependencies(const conduit::Node &schema,
                                  const conduit::Node &input,
                                  conduit::Node &info,
                                  const std::string &path)
{
    if(!schema.has_path("constraints/dependencies")) return true;
    if(!input.dtype().is_object()) return true;

    bool ok = true;
    const conduit::Node &deps = schema["constraints/dependencies"];

    for(conduit::index_t i = 0; i < deps.number_of_children(); ++i)
    {
        const std::string trigger = deps[i].name();
        if(!input.has_child(trigger)) continue;

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

static bool validate_exclusive_children(const conduit::Node &schema,
                                        const conduit::Node &input,
                                        conduit::Node &info,
                                        const std::string &path)
{
    if(!schema.has_path("constraints/exclusiveChildren")) return true;
    if(!input.dtype().is_object()) return true;

    const conduit::Node &keys = schema["constraints/exclusiveChildren"];
    const bool allow_none = schema.has_path("constraints/allowNoneInExclusiveGroup")
                            ? (schema["constraints/allowNoneInExclusiveGroup"].to_int() != 0)
                            : true;

    std::vector<std::string> present;
    present.reserve((size_t)keys.number_of_children());

    for(conduit::index_t i = 0; i < keys.number_of_children(); ++i)
    {
        const std::string k = keys.child(i).as_string();
        if(input.has_child(k)) present.push_back(k);
    }

    const int count = (int)present.size();
    const bool ok = allow_none ? (count <= 1) : (count == 1);

    if(ok) return true;

    std::ostringstream oss;
    oss << "Exclusive-children violation at '"
        << (path.empty() ? "<root>" : path) << "': expected "
        << (allow_none ? "zero or one" : "exactly one")
        << " of {";

    for(conduit::index_t i = 0; i < keys.number_of_children(); ++i)
    {
        if(i) oss << ", ";
        oss << keys.child(i).as_string();
    }
    oss << "}";

    if(count > 0)
    {
        oss << ", but found: {";
        for(size_t i = 0; i < present.size(); ++i)
        {
            if(i) oss << ", ";
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

static bool validate_one_of(const conduit::Node &schema,
                            const conduit::Node &input,
                            conduit::Node &info,
                            const std::string &path)
{
    if(!schema.has_child("oneOf")) return true;

    const conduit::Node &opts = schema["oneOf"];
    int matches = 0;

    // keep at most one representative failure per option for clarity
    std::vector<std::string> option_msgs;

    for(conduit::index_t i = 0; i < opts.number_of_children(); ++i)
    {
        const conduit::Node &opt = opts.child(i);

        conduit::Node tmp;
        tmp.reset();

        bool ok = true;
        ok = check_type(input, opt, tmp, path) && ok;
        ok = validate_required(opt, input, tmp, path) && ok;
        ok = validate_forbid(opt, input, tmp, path) && ok;
        ok = validate_dependencies(opt, input, tmp, path) && ok;
        ok = validate_exclusive_children(opt, input, tmp, path) && ok;

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

    if(matches == 1) return true;

    std::ostringstream oss;
    oss << "oneOf violation at '" << (path.empty() ? "<root>" : path) << "': ";
    if(matches == 0) oss << "input did not match any supported schemas";
    else oss << "input matched " << matches << " options (ambiguous)";
    add_error(info, oss.str());

    // give a couple of hints
    for(size_t i = 0; i < option_msgs.size() && i < 2; ++i)
    {
        add_error(info, std::string("  hint: ") + option_msgs[i]);
    }

    return false;
}

static bool validate_object(const conduit::Node &schema,
                            const conduit::Node &input,
                            conduit::Node &info,
                            const std::string &path)
{
    bool ok = true;

    // Base checks first
    ok = validate_required(schema, input, info, path) && ok;
    ok = validate_forbid(schema, input, info, path) && ok;
    ok = validate_dependencies(schema, input, info, path) && ok;
    ok = validate_exclusive_children(schema, input, info, path) && ok;

    // Enforce unknown fields after we know properties
    ok = validate_additional_properties(schema, input, info, path) && ok;

    // Recurse into declared properties that exist in input
    ok = validate_properties(schema, input, info, path) && ok;

    // Finally, enforce oneOf (treating options as extra constraints on this same object)
    ok = validate_one_of(schema, input, info, path) && ok;

    return ok;
}

static bool validate_node(const conduit::Node &schema,
                          const conduit::Node &input,
                          conduit::Node &info,
                          const std::string &path)
{
    bool ok = true;

    ok = check_type(input, schema, info, path) && ok;
    if(!ok) return false; // type mismatch stops recursion

    const std::string schema_defined_type = get_type_string(schema);
    if(schema_defined_type == "object")
    {
        return validate_object(schema, input, info, path);
    }

    ok = validate_one_of(schema, input, info, path) && ok;

    return ok;
}

// ---------- Schema Validation Entry-Point ----------
bool validate(const conduit::Node &schema,
              const conduit::Node &input,
              conduit::Node &info)
{
    info.reset();
    bool ok = validate_node(schema, input, info, "");

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