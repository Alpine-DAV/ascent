'use strict'; /* globals $ */
(function(){
/**
 * Factory for UI Elements
 * @exports UIFactory
 */
var UIFactory = {

    /**
     * Creates a DIV element containing a CANVAS
     * @return {div}
     */
    createViewport: function(){
            var canvas = $('<canvas class="cvlib_canvas"></canvas>');

            var label = $('<div class="cvlib_label"></div>');

            var container = $('<div class="cvlib_canvasContainer"></div>');
            container.append(canvas);
            container.append(label);

            /*canvas.on('resized', function(canvas, container){
                container.width(canvas.width());
                container.height(canvas.height());
            }.bind(null, canvas, container));*/

            container.canvas = canvas;
            container.label = label;

            return container;
    },

    /**
     * Helper function which returns a symbol for each parameter type
     * @param {string} type - parameter object
     * @return {string} symbol in html encoding
     */
    getTypeSymbol: function(type){
        switch(type){
            case 'range':
                return '&xharr;';
            case 'set':
                return '&#9863;';
            case 'boolean':
                return '&check;';
            default:
                return type;
        }
    },

    /**
     * Creates a TR element with multiple-value input widgets for a parameter
     * @param {parameter} p - parameter object
     * @return {tr}
     */
    createVariableRow: function(p){
        var tr = $('<tr id="cvlib_row_'+p.label.replace(' ','_')+'" class="cvlib_variableRow"></tr>');

        // Add Label
        var label = $('<td>' + p.label + '</td>');
        label.on('click', function(){
            console.log(this.label, this);
        }.bind(p));
        tr.append(label);

        tr.append('<td>'+this.getTypeSymbol(p.type)+'</td>');

        switch (p.type) {
            case 'range':
                if(p.values.length<2){
                    tr.append('<td>'+p.values[0]+' - '+p.values[0]+'</td><td></td>');
                } else {
                    var minMaxInput = this.createMinMaxNumberInput(p);
                    tr.append(
                        $('<td colspan="2"></td>')
                            .append(minMaxInput[0])
                            .append(minMaxInput[1])
                            .append(' - ')
                            .append(minMaxInput[2])
                            .append(minMaxInput[3])
                    );
                }
                break;
            case 'set':
                if(p.values.length<2){
                    tr.append('<td>'+p.values[0]+'</td><td></td>');
                } else {
                    tr.append( $('<td colspan=2></td>').append( this.createSetInput(p) ) );
                }
                break;
            case 'boolean':
                tr.append('<td></td><td></td>');
                break;
        }

        return tr;
    },

    /**
     * Creates a TR element with multiple-value input widgets for a parameter
     * Allows the selection of num different values on the parameter
     * @param {parameter} p - parameter object
     * @param {int} num - Number of selectors to have
     * @return {tr}
     */
    createMultiValueRow: function(p, num) {
        var tr = $('<tr id="cvlib_row_'+p.label.replace(' ','_')+'"></tr>');

        // Add Label
        var label = $('<td>' + p.label + '</td>');
        tr.append(label);

        tr.append('<td>'+this.getTypeSymbol(p.type)+'</td>');

        tr.append( $('<td colspan=2></td>').append(this.createMultiValueInput(p, num)));

        return tr;
    },

    /**
     * Creates a TR element with single-value input widgets for a parameter
     * @param {parameter} p - parameter object
     * @return {tr}
     */
    createFixedRow: function(p){
        var tr = $('<tr id="cvlib_row_'+p.label.replace(' ','_')+'"></tr>');

        // Add Label
        var label = $('<td>' + p.label + '</td>');
        label.on('click', function(){
            console.log(this.label, this);
        }.bind(p));
        tr.append(label);

        tr.append('<td>'+this.getTypeSymbol(p.type)+'</td>');

        switch (p.type) {
            case 'range':
                if(p.values.length<2){
                    tr.append('<td>'+p.query+'</td><td></td>');
                } else {
                    var sliderInput = this.createSliderInput(p);
                    var numberLabel = this.createNumberLabel(p);
                    tr.append(
                        $('<td></td>').append(sliderInput),
                        $('<td></td>').append(numberLabel)
                    );
                }
                break;
            case 'set':
                if(p.values.length<2){
                    tr.append('<td>'+p.query+'</td><td></td>');
                } else {
                    var selectInput = this.createSelectInput(p);
                    tr.append(
                        $('<td></td>').append(selectInput),
                        '<td></td>'
                    );
                }
                break;
            case 'boolean':
                tr.append('<td></td><td></td>');
                break;
        }

        return tr;
    },

    /**
     * Creates a TABLE element with cells containing viewports based on a resultSet
     * @param {ResultSet} resultSet - used to determine matrix size
     * @return {table}
     */
    createMatrix: function(resultSet){
        var table = $('<table class="cvlib_matrix"></table>');
        var map = {};

        for(var i in resultSet.data){
            var tr = $('<tr></tr>');
            map[i] = {};
            for(var j in resultSet.data[i]){
                var viewport = this.createViewport();
                /*if (!(resultSet.data[i][j].hasOwnProperty('label')))
                    viewport.append('<span>('+i+', '+j+')</span>');    */
                tr.append( $('<td></td>').append(viewport) );
                map[i][j] = viewport;
            }
            table.append(tr);
        }

        table.viewportMap = map;

        table.on('transformed', function(e){
            var style = e.target.style;
            this
                .css({
                    left: style.left,
                    top: style.top,
                })
                .width(style.width)
                .height(style.height);
        }.bind(table.find('canvas')));

        return table;
    },

    /**
     * Creates a TABLE element with multiple-value input widgets for two parameters and single-value widgets for the others based on a querySet
     * @param {QuerySet} querySet - contains parameters used to generate the table
     * @return {table}
     */
    createVersusQueryTable: function(querySet){
        var table = this.createSimpleQueryTable(querySet);

        var parameters = Object.keys(querySet.parameters);
        if(parameters.length<2){
            console.error('Cannot create Versus Table for querySets with less then 2 parameters.');
            return;
        }

        var sel1 = this.createRawParameterSelect(querySet);
        sel1.attr('class','p1_select');
        sel1.oldValue = parameters[0];
        sel1.val(parameters[0]);
        var sel2 = this.createRawParameterSelect(querySet);
        sel2.attr('class','p2_select');
        sel2.oldValue = parameters[1];
        sel2.val(parameters[1]);

        var updateTable = function(){
            var p1 = sel1.val();
            var p2 = sel2.val();

            var oldP1 = querySet.info.p1;
            var oldP2 = querySet.info.p2;

            if(p1 === p2){
                sel1.val(oldP1);
                sel2.val(oldP2);
                return;
            }

            querySet.info.p1 = p1;
            querySet.info.p2 = p2;

            var oldRow, newRow, p;

            var replaceVariableRow = function(id){
                oldRow = table.find('#cvlib_row_'+querySet.parameters[id].label.replace(' ','_'));
                p = querySet.parameters[id];
                p.query = p.values[0];
                newRow = UIFactory.createFixedRow(p);
                oldRow.after( newRow );
                oldRow.remove();
            };

            var replaceFixedRow = function(id){
                oldRow = table.find('#cvlib_row_'+querySet.parameters[id].label.replace(' ','_'));
                p = querySet.parameters[id];
                p.query = [p.values[0]];
                newRow = UIFactory.createVariableRow(p);
                oldRow.after( newRow );
                oldRow.remove();
            };

            if(oldP1 !== p1){
                if(oldP1) replaceVariableRow(oldP1);
                replaceFixedRow(p1);
                table.trigger('resized');
            }
            if(oldP2 !== p2){
                if(oldP2) replaceVariableRow(oldP2);
                replaceFixedRow(p2);
                table.trigger('resized');
            }
        };

        sel1.on('change', updateTable);
        sel2.on('change', updateTable);

        table.prepend(
            $('<tr></tr>')
                .append(
                    $('<td colspan=4></td>')
                        .append(sel1)
                        .append(' VS ')
                        .append(sel2)
                )
        );

        querySet.info = {
            type: 'matrix',
            p1: null,
            p2: null,
        };

        updateTable();

        return table;
    },

    /**
     * Creates a TABLE element with multiple-value input widgets for one parameter and single-value widgets for the others based on a querySet
     * @param {QuerySet} querySet - contains parameters used to generate the table
     * @param {int} num - The number of values that can be selected for the multiple-value input
     * @return {table}
     */
    createCompareQueryTable: function(querySet, num){
        var table = this.createSimpleQueryTable(querySet);

        var parameters = Object.keys(querySet.parameters);

        var sel = this.createRawParameterSelect(querySet);
        sel.attr('class','p_select');
        sel.val(parameters[0]);

        var updateTable = function(){
            var selectedP = sel.val();
            var oldP = querySet.info.p;

            querySet.info.p = selectedP;

            var oldRow, newRow, p;

            var replaceMultiValueRow = function(id){
                oldRow = table.find('#cvlib_row_'+querySet.parameters[id].label.replace(' ','_'));
                p = querySet.parameters[id];
                p.query = p.values[0];
                newRow = UIFactory.createFixedRow(p);
                oldRow.after( newRow );
                oldRow.remove();
            };

            var replaceFixedRow = function(id){
                oldRow = table.find('#cvlib_row_'+querySet.parameters[id].label.replace(' ','_'));
                p = querySet.parameters[id];
                p.query = [p.values[0]];
                newRow = UIFactory.createMultiValueRow(p, num);
                oldRow.after( newRow );
                oldRow.remove();
            };

            if(oldP !== selectedP){
                if(oldP) replaceMultiValueRow(oldP);
                replaceFixedRow(selectedP);
                table.trigger('resized');

                if (oldP) {
                    querySet.parameters[oldP].emitter.trigger('change', querySet.parameters[oldP]);
                }
            }
        };

        sel.on('change', updateTable);

        table.prepend(
            $('<tr></tr>')
                .append(
                    $('<td colspan=4></td>')
                        .append("Compare ")
                        .append(sel)
                )
        );

        querySet.info = {
            type: 'compare',
            p: null
        };

        updateTable();

        return table;
    },

    /**
     * Creates a TABLE element with multiple-value input widgets for all parameters
     * @param {QuerySet} querySet - Its parameters will be the targets of the input widgets
     * @return {table}
     */
    createSearchQueryTable: function(querySet) {
        var tr, table = $('<table class="cvlib_queryTable"></table>');

        var p,i;
        for (i in querySet.parameters) {
            p = querySet.parameters[i];
            p.query = [p.values[0]];
            table.append(this.createVariableRow(p));
        }

        table.prepend(
            $('<tr></tr>').append('<td colspan=4></td>')
        );

        querySet.info = {type : 'search'};

        return table;
    },

    /**
     * Creates a TABLE element with single-value input widgets based on a querySet
     * @param {QuerySet} querySet - its parameters will be the targets of the input widgets
     * @return {table}
     */
    createSimpleQueryTable: function(querySet){
        var tr, table = $('<table class="cvlib_queryTable"></table>');

        var p,i;
        for(i in querySet.parameters){
            p = querySet.parameters[i];
            table.append( this.createFixedRow(p) );
        }

        return table;
    },

    /**
     * Creates a DIV element with a searchBar and message to enter text to search querySet
     * @param {QuerySet} querySet
     * @return {div}
     */
    createSearchBar: function(querySet) {
        var searchContainer = $('<div class="cvlib_searchContainer"></div>');
        var searchBar = $('<input type="text" class="cvlib_searchBar"></div>');
        var searchMessage = $('<div class="cvlib_searchMessage"></div>');

        searchContainer.append(searchBar);
        searchContainer.append(searchMessage);

        searchContainer.bar = searchBar;
        searchContainer.message = searchMessage;

        //Parse text in search bar and set parameter queries accordingly
        var updateSearch = function() {
            //Reset parameter queries
            var usedParams = {}; //keeps track of which parameters have been specified in the search
            for (var i in querySet.parameters) {
                p = querySet.parameters[i];
                p.query = {};
                usedParams[p.label] = false;
            }

            if (searchBar.val().length === 0) {
                searchMessage.text('Please enter search terms separated by a semicolon \';\'.');
                searchMessage.attr('mode', 'waiting');
                return;
            }
            //Split into search terms and parse one-by-one
            var terms = searchBar.val().split(';');
            for (var term in terms) {
                term = terms[term];
                term = term.trim();
                //Parse parameter
                var foundParam = null;
                for (var i in querySet.parameters) {
                    if (term.startsWith(querySet.parameters[i].label))
                        foundParam = querySet.parameters[i];
                }
                if (!foundParam) { //Error message if text does not match any parameters
                    var message = "Unrecognized parameter. Allowed values are: ";
                    for (var i in querySet.parameters)
                        message += ("\'"+querySet.parameters[i].label+"\', ");
                    searchMessage.text(message);
                    searchMessage.attr('mode', 'error');
                    return;
                }
                usedParams[foundParam.label] = true;
                //Parse operator and values
                term = term.substring(foundParam.label.length,term.length);
                term = term.trim();
                var operator = term.substr(0,2); //operator is always two characters
                term = term.substring(2,term.length);
                term = term.trim();
                switch (operator) {
                    //Equals. Iterates through comma separated values and adds them to the parameter's query
                    case '==' :
                        var values = term.split(',');
                        var valuesCount = 0;
                        //Iterate through listed values
                        for (var val in values) {
                            val = values[val];
                            val = val.trim();
                            var foundVal = false;
                            //Iterate through parameter values until one matching the text is found
                            for (var i in foundParam.values) {
                                if (val == foundParam.values[i].toString()) {
                                    foundParam.query[valuesCount] = foundParam.values[i];
                                    valuesCount++;
                                    foundVal = true;
                                }
                            }
                            if (!foundVal) { //Error if no matching value is found
                                var message = "Unrecognized value: \'"+val+"\'. Allowed values are: ";
                                for (var i in foundParam.values)
                                    message += ("\'"+foundParam.values[i]+"\', ");
                                message += ("(comma separated)");
                                searchMessage.text(message);
                                searchMessage.attr('mode', 'error');
                                return;
                            }
                        }
                        break;
                    //Greater than. Adds given value and all values after it to parameter's query
                    case '>=' :
                        var foundVal = null;
                        var foundValIndex;
                        //Iterate through parameter values until one matching the text is found
                        for (var i in foundParam.values) {
                            if (term == foundParam.values[i].toString()) {
                                foundVal = foundParam.values[i];
                                foundValIndex = i;
                            }
                        }
                        if (foundVal == null) { //Error if no matching value is found
                            var message = "Unrecognized value: \'"+term+"\'. Allowed values are: ";
                            for (var i in foundParam.values)
                                message += ("\'"+foundParam.values[i]+"\', ");
                            searchMessage.text(message);
                            searchMessage.attr('mode', 'error');
                            return;
                        }
                        //Add to query
                        var query = [];
                        for (var i = foundParam.values.length-1; i >= foundValIndex; i--)
                            query.push(foundParam.values[i]);
                        foundParam.setValue(query);
                        break;
                    //Less than. Adds given value and all values before it to parameter's query
                    case '<=' :
                        var foundVal = null;
                        var foundValIndex;
                        //Iterate through parameter values until one matching the text is found
                        for (var i in foundParam.values) {
                            if (term == foundParam.values[i].toString()) {
                                foundVal = foundParam.values[i];
                                foundValIndex = i;
                            }
                        }
                        if (foundVal == null) { //Error if no matching value is found
                            var message = "Unrecognized value: \'"+term+"\'. Allowed values are: ";
                            for (var i in foundParam.values)
                                message += ("\'"+foundParam.values[i]+"\', ");
                            searchMessage.text(message);
                            searchMessage.attr('mode', 'error');
                            return;
                        }
                        //Add to query
                        var query = [];
                        for (var i = 0; i <= foundValIndex; i++)
                            query.push(foundParam.values[i]);
                        foundParam.setValue(query);
                        break;
                    default:
                        searchMessage.text("Unrecognized operator: \'"+operator+"\'. Allowed values are: \'==\', \'>=\', \'<=\'");
                        searchMessage.attr('mode', 'error');
                    return;
                } //End switch
            } //End for loop (terms)
            //Fill queries for unspecified parameters
            for (var i in querySet.parameters) {
                var p = querySet.parameters[i];
                if (!usedParams[p.label]) {
                    for (var j in p.values)
                        p.query[j] = p.values[j];
                }
            }
            //Estimate number of results
                var resultsCount = 1;
                for (var i in querySet.parameters)
                    resultsCount *= Object.keys(querySet.parameters[i].query).length;
                searchMessage.text("Estimated number of results: " + resultsCount);
                searchMessage.attr('mode', 'valid');
        }; //End updateSearch

        querySet.info = {type : 'search'};

        searchBar.on('input', updateSearch);

        updateSearch();

        return searchContainer;
    },

    /**
     * Creates a number label element for a parameter
     * @param {parameter} p - target of the input widget
     * @return {input}
     */
    createNumberLabel: function(p) {
        var label = $('<span class="cvlib_numberLabel">'+p.query+'</span>');
        label.oldValue = p.query;

        p.emitter.on('change', function(input, e, p) {
            input.oldValue = p.query;
            input.text(p.query);
        }.bind(null, label));

        return label;
    },

    /**
     * Creates a range INPUT element for a min, max, and default value
     * @param {number} min - min
     * @param {number} max - max
     * @param {number} value - default value
     * @return {input}
     */
    createRawSliderInput: function(min, max, value){
        return $('<input type="range"  min="'+min+'" max="'+max+'" value="'+value+'">');
    },

    /**
     * Creates a range INPUT element for a parameter
     * @param {parameter} p - target of the input widget
     * @return {input}
     */
    createSliderInput: function(p) {
        var input = this.createRawSliderInput(0, p.values.length-1, p.values.indexOf(p.query));

        p.emitter.on('change', function(input, e, p){
            input.val( parseInt(p.values.indexOf(p.query)) );
        }.bind(null, input));

        input.on('input', function(input, p){
            p.setValue( p.values[input.val()] );
        }.bind(null, input, p));

        return input;
    },

    /**
     * Creates a SELECT element for a set of labels and values
     * @param {string[]} labels
     * @param {object[]} values
     * @return {select}
     */
    createRawSelectInput: function(labels, values){
        var select = $('<select></select>');
        for(var i in labels)
            select.append('<option'+ (values ? ' value="'+values[i]+'"' : '') +'>'+labels[i]+'</option>');

        return select;
    },

    /**
     * Creates a SELECT element for selecting a parameter of the given querySet
     * @param {querySet} querySet
     * @return {select}
     */
    createRawParameterSelect: function(querySet) {
        var keys = [];
        var labels = [];
        for (var p in querySet.parameters) {
            keys.push(p)
            labels.push(querySet.parameters[p].label);
        }
        return this.createRawSelectInput(labels,keys);
    },

    /**
     * Creates a SELECT element for a parameter
     * @param {parameter} p - target of the input widget
     * @return {input}
     */
    createSelectInput: function(p){
        var select = this.createRawSelectInput(p.values);

        p.emitter.on('change', function(select, e, p){
            select.val(p.query);
        }.bind(null, select));

        select.on('change', function(select, p){
            p.setValue(select.val());
        }.bind(null, select, p));

        return select;
    },

    /**
     * Creates a checkbox INPUT element
     * @param {bool} checked - default
     * @return {input}
     */
    createRawCheckboxInput: function(checked){
        return $('<input type="checkbox" ' + (checked ? 'checked' : '') + '>');
    },

    /**
     * Creates a checkbox INPUT element for a parameter
     * @param {parameter} p - target of the input widget
     * @return {input}
     */
    createCheckboxInput: function(p){
        var select = this.createRawSelectInput(p);

        p.emitter.on('change', function(select, e, p){
            select.val(p.query);
        }.bind(null, select));

        select.on('change', function(select, p){
            p.setValue(select.val());
        }.bind(null, select, p));

        return select;
    },

    /**
     * Creates a pair of synchronized number INPUT elements for a parameter representing a min and max value
     * @param {parameter} p - target of the input widgets
     * @return {Array.<input>} the min and max element
     */
    createMinMaxNumberInput: function(p) {
        var minInput = $('<input type="number" value="0" min="0" max="'+(p.values.length-1)+'" step="1"></input>');
        //var minInputVis = this.createRawNumberInput(0,0,p.query[0]);
        var minInputVis = $('<span class="cvlib_numberLabel">'+p.query[0]+'</span>');

        var maxInput = $('<input type="number" value="0" min="0" max="'+(p.values.length-1)+'" step="1"></input>');
        //var maxInputVis = this.createRawNumberInput(0,0,p.query[0]);
        var maxInputVis = $('<span class="cvlib_numberLabel">'+p.query[0]+'</span>');

        minInput.on('input', function(){
            var v1 = parseInt(minInput.val());
            minInputVis.text(p.values[v1]);

            var v2 = parseInt(maxInput.val());
            if(v1>v2){
                maxInput.val(v1);
                maxInputVis.text(p.values[v1]);
                v2=v1;
            }

            var query = [];
            for(var i=v1; i<=v2; i++)
                query.push(p.values[i]);
            p.setValue(query);
        });

        maxInput.on('input', function(){
            var v2 = parseInt(maxInput.val());
            maxInputVis.text(p.values[v2]);

            var v1 = parseInt(minInput.val());
            if(v1>v2){
                minInput.val(v2);
                minInputVis.text(p.values[v2]);
                v1=v2;
            }

            var query = [];
            for(var i=v1; i<=v2; i++)
                query.push(p.values[i]);
            p.setValue(query);
        });

        p.emitter.on('change', function(e) {
            var max = p.values.indexOf(p.query[p.query.length-1]);
            var min = p.values.indexOf(p.query[0]);
            maxInput.val(max);
            minInput.val(min);
            maxInputVis.text(p.values[max]);
            minInputVis.text(p.values[min]);
        });

        return [minInput, minInputVis, maxInput, maxInputVis];
    },

    /**
     * Creates a container containing two interactive lists based on parameter. The first list contains all available options and the other all selected options.
     * @param {parameter} p - target of the input widgets
     * @return {div}
     */
    createSetInput: function(p){
        var container = $('<div class="cvlib_setInput"></div>');
        var sourceList = $('<ul></ul>');
        var targetList = $('<ul></ul>');

        var dragElement = null;

        var updateSetInput = function(){
            var values = [];
            var temp = targetList.find('li');
            for(var i=0, j=temp.length-1; i<j; i++)
                values.push(temp[i].innerHTML);
            p.setValue(values);
        };

        var addDragListeners = function(li){
            li.on('dragenter', function(){
                $(this).addClass('dragOver');
            });
            li.on('dragover', function(e){
                e.preventDefault();
            });
            li.on('dragleave', function(){
                $(this).removeClass('dragOver');
            });
            li.on('dragstart', function(e){
                dragElement = this;
            });
            li.on('drop', function(e){
                $(this).removeClass('dragOver');
                if(dragElement!=this){
                    $(this).before(dragElement);
                    updateSetInput();
                }
            });
        };

        var ghost = $('<li></li>');
        addDragListeners(ghost);
        targetList.append(ghost);
        ghost.hide();

        var createTargetLi = function(v){
            var li = $('<li draggable="true">'+v+'</li>');
            addDragListeners(li);
            li.on('click', function(e){
                if(e.which === 1 && targetList.find('li').length>2){
                    $(this).remove();
                    updateSetInput();
                }
            });
            return li;
        };

        var addTarget = function(e){
            if(!targetList.find('li:contains('+this.v+')').length){
                ghost.before( createTargetLi(this.v) );
                updateSetInput();
            }
        };

        p.emitter.on('change', function(e){
            var i,dej;
            var temp = ghost.prev();
            while(temp.length){
                temp.remove();
                temp = ghost.prev();
            }
            for(i in p.query){
                ghost.before( createTargetLi(p.query[i]) );
            }
        });

        var i;

        for(i in p.query){
            ghost.before( createTargetLi(p.query[i]) );
        }

        for(i in p.values){
            var source = $('<li>'+p.values[i]+'</li>');
            source[0].v = p.values[i];
            source.on('click', addTarget);
            sourceList.append(source);
        }
        container.append(sourceList, targetList);

        return container;
    },

    /**
     * Creates a container containing num selections based on the parameter
     * @param {parameter} p - target of the input widgets
     * @param {int} num - The number of selectors to have
     * @return {div}
     */
    createMultiValueInput: function(p, num) {
        var container = $('<div class="cvlib_multiValueInput"></div>');

        var i = 0;
        for (i = 0; i < num; i++) {
            (function (){ //nested into a function because javascript doesn't do block scope otherwise
                            //coming back to this function months later, I realize how much I've improved with javascript since then
                            //and seeing hackiness like this makes me cringe, but I don't have time to rewrite it now so I must live with it :(
                var index = i;
                p.query[index] = p.values[0];
                var sel = CVLIB.UIFactory.createRawSelectInput(p.values);
                sel.attr('class','p_'+index);
                sel.on('change', function(){
                    p.query[index] = sel.val();
                    p.emitter.trigger('change', p);
                });
                container.append(sel);
                if (i != num-1) {
                    container.append(' , ');
                }
            })();
        }

        p.emitter.on('change',function() {
            for(var i in p.query) {
                var sel = container.find('.p_'+i);
                sel.val(p.query[i]);
            }
        });

        return container;
    },

    /**
     * Creates a container containing a sidebar and content container
     * @return {div}
     */
    createSidebarLayout: function(){
        var container = $('<div></div>');
        var sidebar = $('<div class="cvlib_sidebarLayoutSidebar"></div>');
        var content = $('<div class="cvlib_sidebarLayoutContent"></div>');
        var oldAppend = content.append;
        content.append = function(dom){
            content.css({left: sidebar[0].getBoundingClientRect().right});
            oldAppend.apply(this, dom);
        };

        sidebar.on('resized', function(){
            content.css({left: sidebar[0].getBoundingClientRect().right});
        });

        container.append(sidebar, content);

        container.sidebar = sidebar;
        container.content = content;

        return container;
    },

    /**
     * Creates a DIV element representing a loading bar with the functions 'setSteps' and 'progress'.
     * @return {div}
     */
    createLoadingBar: function(){
        var bar = $('<div class="cvlib_loadingBar"></div>');
        bar.setSteps = function(steps){
            this.width(0);
            this.fadeIn(0);
            this.step = 0;
            this.steps = steps;
        };
        bar.progress = function(){
            this.step++;
            this.width( this.step/this.steps * $(window).width() );
            if(this.step === this.steps)
                this.fadeOut(300);
        };
        return bar;
    },

    /**
     * Creates a DIV element representing a loading symbol
     * @return {div}
     */
    createLoadingSymbol: function(){
        return $('<div class="cvlib_loadingSymbol">Loading...</div>');
    }
};

if(!window.CVLIB) window.CVLIB = {};
window.CVLIB.UIFactory = UIFactory;

})();
