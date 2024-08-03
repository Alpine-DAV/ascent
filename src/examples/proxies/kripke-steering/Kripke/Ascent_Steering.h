#include <ascent.hpp>
#include <conduit_blueprint.hpp>

// Helpers
void register_callbacks()
{
    ascent::register_callback("test_callback", void_callback_1);
}

// Callbacks
void void_callback_1(conduit::Node &params, conduit::Node &output)
{
    output["param_was_passed"] = false;
    if (params.has_path("example_param"))
    {
        output["param_was_passed"] = true;
    }
}
