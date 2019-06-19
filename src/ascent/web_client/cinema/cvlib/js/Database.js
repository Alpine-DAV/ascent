'use strict'; /* globals $ CVLIB */
(/** @lends module:CVLIB */function(){

/**
 * @constructor
 * @abstract
 */
function Database(url, callback){
    if (this.constructor === Database) {
        throw new Error("Can't instantiate abstract class!");
    }
}

/**
 * Validate if JSON file is compliant to a Spec
 * @param {JSON} json - JSON object
 * @return {bool}
 */
Database.prototype.validateJSON = function(json){
    throw new Error("Abstract method!");
};

/**
 * Asynchronously processes a querySet and passes the resultSet to a callback
 * @param {QuerySet} querySet - database request
 * @param {function} callback - will be called after request has been proceed with ResultSet as its first argument
 */
Database.prototype.processQuery = function(querySet, callback){
    throw new Error("Abstract method!");
};

if(!window.CVLIB) window.CVLIB = {};
window.CVLIB.Database = Database;

})();