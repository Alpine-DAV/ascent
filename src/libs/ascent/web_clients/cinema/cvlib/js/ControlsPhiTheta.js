'use strict'; /* globals $ CVLIB */
(/** @lends module:CVLIB */function(){

/**
 * Augments a viewport with mouse-controls to zoom, pan, and rotate according to a phi-theta camera model.
 * With these controls, left-click-and-dragging on the image rotates the view, scrolling resizes the image
 * and shift-click-and-dragging pans the image.
 * @constructor
 * @param {viewport} viewport - viewport which is target of the mouse interaction
 * @param {QuerySet} querySet - is updated after view is rotated
 */
function ControlsPhiTheta(viewport, querySet){
    var j;
    var theta;
    var phi;
    var temp;

    var tMin, tMax, pMin, pMax, t, p, tNum, pNum, tOld, pOld;

    var epsX = 0.4;
    var epsY = 0.8;

    var canvas = viewport.canvas;
    var zoomFactor = 0.1;

    var x0=0, y0=0, mode=0;

    // Check if rotation is possible
    if(querySet.parameters.hasOwnProperty('theta') && querySet.parameters.hasOwnProperty('phi')){
        theta = querySet.parameters.theta;
        phi = querySet.parameters.phi;

        tMin = Infinity;
        tMax = -Infinity;
        for(j in theta.values){
            temp = parseFloat(theta.values[j]);
            if(tMin > temp) tMin = temp;
            if(tMax < temp) tMax = temp;
        }

        pMin = Infinity;
        pMax = -Infinity;
        for(j in phi.values){
            temp = parseFloat(phi.values[j]);
            if(pMin > temp) pMin = temp;
            if(pMax < temp) pMax = temp;
        }

        t = theta.query;
        p = phi.query;
        tNum = parseFloat(t);
        pNum = parseFloat(p);
    } else {
        console.error('Unable to create OrbitControls: QuerySet does not have a "theta" or "phi" parameter');
        return;
    }

    var onMouseDown = function(e){
        if (e.which !== 1) return;
        e.preventDefault();
        x0 = e.clientX;
        y0 = e.clientY;

        mode = -1;
        if (e.shiftKey) {
            // Panning
            mode = 1;
        }
        else {
            // Rotating
            mode = 0;
            t = theta.query;
            p = phi.query;
            tNum = parseFloat(t);
            pNum = parseFloat(p);
            tOld = t;
            pOld = p;
        }

        viewport.off('mousedown', onMouseDown);
        viewport.on('mousemove', onMouseMove);
        viewport.on('mouseout', onMouseUp);
        viewport.on('mouseup', onMouseUp);
    };

    var snapToValue = function(v, values){
        var min = Infinity, temp, temp2, snappedValue;
        for(var i in values){
            temp = parseFloat(values[i]);
            temp2 = Math.abs(v-temp);
            if(temp2<min){
                min = temp2;
                snappedValue = values[i];
            }
        }
        return snappedValue;
    };

    var onMouseMove = function(e){
        e.preventDefault();
        e.stopPropagation();

        var dx = (e.clientX-x0);
        var dy = (y0-e.clientY);
        if(mode===0){
            // Rotate
            tNum = Math.max(tMin, Math.min(tMax, tNum + dy*epsX));
            pNum = Math.max(pMin, Math.min(pMax, pNum + dx*epsX));

            var tNew = snapToValue(tNum, theta.values);
            var pNew = snapToValue(pNum, phi.values);

            if(tOld!==tNew || pOld!==pNew){
                tOld = tNew;
                pOld = pNew;
                theta.setValue(tNew);
                phi.setValue(pNew);
            }
        }else if(mode===1){
            var pos = canvas.position();
            canvas.css({
                'left': pos.left+dx,
                'top': pos.top-dy
            });
            canvas.trigger('transformed', canvas);
        }
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
window.CVLIB.ControlsPhiTheta = ControlsPhiTheta;

})();