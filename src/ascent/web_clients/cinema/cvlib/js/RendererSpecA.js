'use strict'; /* globals $ CVLIB*/
(/** @lends module:CVLIB */function(){

/**
 * Renderer for Spec A ResultSets
 * @constructor
 */
function RendererSpecA(){
    CVLIB.Renderer.call( this );

    this.imageLoader = {};
}
RendererSpecA.prototype = Object.create( CVLIB.Renderer.prototype );
RendererSpecA.prototype.constructor = RendererSpecA;

/**
 * Render an element to a canvas and abortOld render process if requested
 * @param {object} element - element of a spec A ResultSet
 * @param {canvas} canvasJQ - canvas in JQ representation
 * @param {bool} abortOld - if true abort old render process
 */
RendererSpecA.prototype.render = function(element, canvasJQ, abortOld, callback){
    var canvas = canvasJQ[0];
    switch (element.type) {
        case 'image':
            if(element.hasOwnProperty('img')) {
                canvas.width = element.img.width;
                canvas.height = element.img.height;
                canvas.getContext('2d').drawImage(element.img, 0, 0);
                canvas.img = element.img;
            }
            else {
                this.imageLoader.ignore = abortOld;
                this.imageLoader = document.createElement('img');
                this.imageLoader.onload = function(){
                    if(this.ignore) return;

                    element.img = this;
                    if(this.width !== canvas.width || this.height !== canvas.height){
                        canvas.width = this.width;
                        canvas.height = this.height;
                        canvasJQ.trigger('resized');
                    }
                    canvas.getContext('2d').drawImage(this, 0, 0);
                    canvas.img = element.img;

                    if (callback)
                        callback();
                };
                this.imageLoader.src = element.src;
            }
            break;

        default:
            console.error('Renderer doesn\'t support element type', element.type, element);
    }
};

if(!window.CVLIB) window.CVLIB = {};
window.CVLIB.RendererSpecA = RendererSpecA;

})();