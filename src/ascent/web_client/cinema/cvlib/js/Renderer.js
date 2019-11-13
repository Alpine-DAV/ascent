'use strict'; /* globals $ */
(/** @lends module:CVLIB */function(){

/**
 * Renderer for ResultSets
 * @constructor
 * @abstract
 */
function Renderer(){
    if (this.constructor === Renderer) {
        throw new Error("Can't instantiate abstract class!");
    }
}

/**
 * Render an element to a canvas
 * @param {object} element - element of a resultSet
 * @param {canvas} canvasJQ - canvas in JQ representation
 */
Renderer.prototype.render = function(element, canvasJQ){
    throw new Error("Abstract method!");
};

if(!window.CVLIB) window.CVLIB = {};
window.CVLIB.Renderer = Renderer;

})();