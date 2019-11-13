'use strict'; /* globals $ */
(/** @lends module:CVLIB */function(){

/**
 * Represents a set of results for a QuerySet
 * @constructor
 * @param {JSON} serializedQuerySet - serialized version of the requested QuerySet
 * @param {JSON} data               - data object containg the response to the requested QuerySet
 * @prop {JSON} querySet - serialized version of the requested QuerySet
 * @prop {JSON} data     - data object containg the response to the requested QuerySet
 */
function ResultSet(serializedQuerySet, data){
    this.querySet = serializedQuerySet;
    this.data = data;
}

if(!window.CVLIB) window.CVLIB = {};
window.CVLIB.ResultSet = ResultSet;

})();