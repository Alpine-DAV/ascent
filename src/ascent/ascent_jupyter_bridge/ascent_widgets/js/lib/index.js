define(function (require) {
    var exports = {};

    var loadedModules = [
        require("./trackball/trackball"),
    ];

    for (var i in loadedModules) {
        if (loadedModules.hasOwnProperty(i)) {
            var loadedModule = loadedModules[i];
            for (var target_name in loadedModule) {
                if (loadedModule.hasOwnProperty(target_name)) {
                    exports[target_name] = loadedModule[target_name];
                }
            }
        }
    }

    return exports;
});

