'use strict'; /* globals $ */
(/** @lends module:CVLIB */function(){

/**
 * Represents a parameter which can be queried
 * @constructor
 * @param {string}    label   - parameter label
 * @param {string}    type    - parameter type
 * @param {Array.<T>} values  - list of paramters values
 * @param {T}         query   - a subset of values
 * @prop  {string}    label   - parameter label
 * @prop  {string}    type    - parameter type
 * @prop  {Array.<T>} values  - list of paramters values
 * @prop  {T}         query   - a subset of values
 * @prop  {div}       emitter - div element used to trigger when parameter was updated
 */
function Parameter(label, type, values, query){
    this.label = label;
    this.type = type;
    this.values = values;
    this.query = query;
    this.emitter = $('<div></div>');
}

/**
 * Sets the query value of a paramter and triggers a change event unless suppressChangeEvent is true
 * @param {T}    query               - target value
 * @param {bool} suppressChangeEvent - if true the change event will not be fired
 */
Parameter.prototype.setValue = function(query, suppressChangeEvent){
    this.query = query;
    if(!suppressChangeEvent)
        this.emitter.trigger('change', [this]);
};

/**
 * Creates a clone of the parameter
 * @return {Parameter}
 */
Parameter.prototype.clone = function(){
    return new Parameter(this.label, this.type, this.values, this.query);
};

if(!window.CVLIB) window.CVLIB = {};
window.CVLIB.Parameter = Parameter;

})();