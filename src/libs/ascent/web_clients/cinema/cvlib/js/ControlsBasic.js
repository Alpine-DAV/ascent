'use strict'; /* globals $ */
(/** @lends module:CVLIB */function(){

/**
 * Augments a viewport with mouse-controls to zoom and pan.
 * With these controls, left-click-and-dragging pans the image and scrolling zooms the image.
 * @constructor
 * @param {viewport} viewport - viewport which is target of the mouse interaction
 */
function ControlsBasic(viewport){
    var canvas = viewport.canvas;
    var zoomFactor = 0.1;

    var x0=0, y0=0;

    var onMouseDown = function(e){
        if(e.which !== 1) return;
        e.preventDefault();
        x0 = e.clientX;
        y0 = e.clientY;

        viewport.off('mousedown', onMouseDown);
        viewport.on('mousemove', onMouseMove);
        viewport.on('mouseout', onMouseUp);
        viewport.on('mouseup', onMouseUp);
    };

    var onMouseMove = function(e){
        e.preventDefault();
        e.stopPropagation();

        var dx = (e.clientX-x0);
        var dy = (y0-e.clientY);

        var pos = canvas.position();
        canvas.css({
            'left': pos.left+dx,
            'top': pos.top-dy
        });
        canvas.trigger('transformed', canvas);

        x0 = e.clientX;
        y0 = e.clientY;

    };

    var onMouseUp = function(e){
        viewport.on('mousedown', onMouseDown);
        viewport.off('mousemove', onMouseMove);
        viewport.off('mouseout', onMouseUp);
        viewport.off('mouseup', onMouseUp);
    };

    var onMouseWheel = function(e){
        e.preventDefault();
        e.stopPropagation();

        // Zoom in our out
        var delta = 1 + (e.originalEvent.deltaY<0 ? zoomFactor : -zoomFactor);

        // Position relative to canvas
        var xPos = e.pageX - canvas.offset().left;
        var yPos = e.pageY - canvas.offset().top;

        // Normalize cursor position relative to canvas
        var xRel = xPos/canvas.width();
        var yRel = yPos/canvas.height();

        // Compute new canvas size
        var dx = delta*canvas.width();
        var dy = delta*canvas.height();

        // Compute projection of old cursor position on new canvas size
        var xPosNew = xRel*dx;
        var yPosNew = yRel*dy;

        var pos = canvas.position();
        canvas.css({
            'width': dx,
            'height': dy,
            'left': pos.left - (xPosNew-xPos),
            'top': pos.top - (yPosNew-yPos)
        });
        canvas.trigger('transformed', canvas);
    };

    viewport.on('mousedown', onMouseDown);
    viewport.on('wheel', onMouseWheel);
    viewport.on('contextmenu', function(e){
        e.preventDefault();
        e.stopPropagation();
        return false;
    });
}

if(!window.CVLIB) window.CVLIB = {};
window.CVLIB.ControlsBasic = ControlsBasic;

})();