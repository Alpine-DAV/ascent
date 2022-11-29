// This file contains the javascript that is run when the notebook is loaded.
// It contains some requirejs configuration and the `load_ipython_extension`
// which is required for any notebook extension.

// Configure requirejs
define(function () {
    if (window.require) {
        window.require.config({
            map: {
                '*' : {
                    'ascent_widgets': 'nbextensions/ascent_widgets/index',
                }
            }
        });
    } else {
        console.log('RequireJS was not found');
    }

    function load_ipython_extension() {

    }
    // Export the required load_ipython_extension
    return {
        load_ipython_extension: load_ipython_extension
    };
});
