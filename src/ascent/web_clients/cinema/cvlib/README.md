# CVLIB - Cinema Viewer Library

## NOTE: CVLIB is for Spec-A Cinema Databases ONLY!

CVLIB is a library which provides a JavaScript API to Paraview Cinema databases for visualization in the Browser.

In contrast to other Cinema viewers this library is focusing on modular expandability. Most applications have additional viewer requirements such as specialized data visualizations, aggregations, interactions, and other features a generic all-in-one viewer can not provide. Hence, CVLIB provides functions to access Cinema databases, process queries, create viewer UI elements, render results, and interact with the visualization so that users can build their own custom viewer.

The **Demos** folder contains many pre-built viewers and sample databases to study how to use cvlib in various ways. It is recommended to first look at the viewers marked with a **BEG** (for beginner) before examining the intermediate (**INT**) and advanced (**ADV**) viewers.

### Architecture
CVLIB follows a simple system architecture. It provides the following classes:
* **Parameter**: a parameter which can be queryied (e.g. camera angle, iso-value, time, ...)
* **QuerySet**: a set of Parameters
* **ResultSet**: a response to a QuerySet (e.g. images, graphs, numbers, ...)
* **Database**: provides an interface to the Cinema database and processes QuerySets
* **Renderer**: renders ResultSets
* **UIFactory**: generates HTML widgets to request QuerySets and display rendered ResultSets
* **Controls**: augments HTML widgets elements with mouse interactions such as panning and zooming

A Cinema database is interfaced via the Database class which can process QuerySets. The database will respond with ResultSets which can be rendered with the Renderer. The UIFactory can generate HTML widgets to adjust QuerySets via input elements and display output from the Renderer with viewports. The different Control classes enable users to zoom, pan, and rotate viewports.

### Examples
To create viewers one simply has to use the modules and interconnect them.

```javascript
// Load JSON file decribing a Cinema SpecA Database and return the database Scheme as a QuerySet
var db = new CVLIB.DatabaseSpecA('data/volume-render/info.json', function(querySet){

    // Create a sidebar layout
    var sidebarLayout = CVLIB.UIFactory.createSidebarLayout();

    // Add the layout to the body
    $('body').append(sidebarLayout);

    // Add a headline to the sidebar
    sidebarLayout.sidebar.append( '<h2>Cinema Viewer<br>Spec A - Single</h2>' );

    // Create a table with input widgets to modify the QuerySet
    var queryTable = CVLIB.UIFactory.createSimpleQueryTable( querySet );

    // Add the QueryTable to the sidebar
    sidebarLayout.sidebar.append(queryTable);

    // Create a viewport for the images
    var viewport = CVLIB.UIFactory.createViewport();

    // Add the Viewport to the content
    sidebarLayout.content.append(viewport);

    // Augment the viewport with controls which can also modify the QuerySet
    new CVLIB.ControlsPhiTheta(viewport, querySet);

    // Create Renderer for SpecA images
    var renderer = new CVLIB.RendererSpecA();

    // Function which renders a resultSet to a canvas
    var renderFunction = function(resultSet){
        renderer.render(resultSet.data, viewport.canvas, true);
    };

    // Function which requests a ResultSet for the current QuerySet
    var updateFunction = function(){
        db.processQuery( querySet, renderFunction );
    };

    // On change of the QuerySet call updateFunction
    querySet.emitter.on(
        'change',
        updateFunction
    );

    // Request image for initial QuerySet
    updateFunction();
});
```
