'use strict'; /* globals $ */
(/** @lends module:CVLIB */function(){

/**
 * Represents a set of parameters which can be queried
 * @constructor
 * @param {object} parameters - list of paramters to query
 * @prop {div} emitter     - div element used to trigger and listen to query related events
 * @prop {JSON} parameters - object representing all queryable parameters
 * @prop {JSON} info       - generic information about the QuerySet
 */
function QuerySet(parameters){

    this.emitter = $('<div></div>');
    this.parameters = parameters;

    var self = this;
    for(var i in this.parameters){
        var p = this.parameters[i];
        this.emitter.append(p.emitter);
    }

    this.info = {
        type: 'single'
    };
}

/**
 * Returns a simplified JSON version of the object
 * @return {JSON} simplified JSON representation of the QuerySet
 */
QuerySet.prototype.serialize = function(){
    var parameters = {};
    for(var i in this.parameters){
        var p = this.parameters[i];
        parameters[i] = p.query;
    }

    return {
        info: this.info,
        parameters: parameters
    };
};

QuerySet.prototype.deserialize = function(json){
    if (this.info.type == json.info.type) {
        for (var i in json.parameters) {
            if (this.parameters[i])
                this.parameters[i].setValue(json.parameters[i]);
            else
                console.log("Error: Could not load parameter " + i);
        }
    }
    else
        console.log("Error: Could not load query of different type")
}

/**
 * Creates a clone of the QuerySet
 * @return {QuerySet}
 */
QuerySet.prototype.clone = function(){
    var parameters = {};
    for(var i in this.parameters){
        parameters[i] = this.parameters[i].clone();
    }
    return new QuerySet( parameters );
};

if(!window.CVLIB) window.CVLIB = {};
window.CVLIB.QuerySet = QuerySet;

})();